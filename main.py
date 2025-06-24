import logging
import sys
from typing import Dict, Any, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
import json
import operator
from pathlib import Path

# Core dependencies
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Rich for beautiful CLI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import box

class Logger:
    """Centralized logging configuration"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = 'logs/classification_log.jsonl') -> logging.Logger:
        """Setup structured logging to file only"""
        
        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # File handler for structured logs
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(message)s')  # JSON logs don't need timestamp prefix
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger

@dataclass
class ClassificationResult:
    """Structure for classification results"""
    label: str
    confidence: float
    raw_scores: Dict[str, float]
    timestamp: str
    input_text: str
    corrected_by_user: bool = False

class WorkflowState(TypedDict):
    """State management for the LangGraph workflow"""
    input_text: str
    prediction: Optional[ClassificationResult]
    confidence_threshold: float
    needs_clarification: bool
    user_feedback: Optional[str]
    final_result: Optional[ClassificationResult]
    messages: list[BaseMessage]
    session_id: str
    suggested_label: Optional[str]

class TextClassificationWorkflow:
    """LangGraph workflow for text classification with confidence-based fallback"""
    
    def __init__(self, 
                 model_name: str = "multiheadattn/my_awesome_model",
                 confidence_threshold: float = 0.75):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.classifier = None
        self.tokenizer = None
        self.model = None
        self.logger = Logger.setup_logger(__name__)
        self.console = Console()
        self._load_model()
        self.workflow = self._build_workflow()
        
    def _load_model(self):
        """Load the Hugging Face model and tokenizer"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task(f"Loading model: {self.model_name}", total=None)
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.classifier = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    top_k=None
                )
                
            self.logger.info(json.dumps({
                "event": "model_loaded",
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            self.logger.error(json.dumps({
                "event": "model_load_failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }))
            raise

    def inference_node(self, state: WorkflowState) -> WorkflowState:
        """InferenceNode: Runs classification using the trained model"""
        input_text = state["input_text"]
        
        try:
            # get the prediction with all scores
            results = self.classifier(input_text)

            # extract the top prediciton
            top_prediction = max(results[0], key=lambda x: x['score'])

            # create structured result
            raw_scores = {result['label']: result['score'] for result in results[0]}
            
            prediction = ClassificationResult(
                label=top_prediction['label'],
                confidence=top_prediction['score'],
                raw_scores=raw_scores,
                timestamp=datetime.now().isoformat(),
                input_text=input_text
            )
            
            log_data = {
                "event": "initial_prediction",
                "input": input_text,
                "predicted_label": prediction.label,
                "confidence": prediction.confidence,
                "all_scores": raw_scores,
                "session_id": state["session_id"],
                "timestamp": datetime.now().isoformat()
            }
            self.logger.info(json.dumps(log_data))
            
            state["prediction"] = prediction
            state["messages"] = add_messages(
                state["messages"],
                [AIMessage(content=f"[InferenceNode] Predicted label: {prediction.label} | Confidence: {prediction.confidence:.0%}")]
            )
            
        except Exception as e:
            self.logger.error(json.dumps({
                "event": "inference_failed",
                "error": str(e),
                "session_id": state["session_id"],
                "timestamp": datetime.now().isoformat()
            }))
            raise
        
        return state
    
    def confidence_check_node(self, state: WorkflowState) -> WorkflowState:
        """ConfidenceCheckNode: Evaluates confidence and decides on fallback"""
        prediction = state["prediction"]
        threshold = state["confidence_threshold"]
        
        if prediction.confidence < threshold:
            state["needs_clarification"] = True

            # log fallback trigger
            log_data = {
                "event": "fallback_triggered",
                "reason": "low_confidence",
                "confidence": prediction.confidence,
                "threshold": threshold,
                "session_id": state["session_id"],
                "timestamp": datetime.now().isoformat()
            }
            self.logger.info(json.dumps(log_data))
            state["messages"] = add_messages(
                state["messages"],
                [AIMessage(content=f"[ConfidenceCheckNode] Confidence too low. Triggering fallback...")]
            )
        else:
            state["needs_clarification"] = False
            state["final_result"] = prediction
            
        return state
    
    def fallback_node(self, state: WorkflowState) -> WorkflowState:
        """FallbackNode: Asks user for clarification about their intent."""
        prediction = state["prediction"]
        
        # Determine the next most likely label as a suggestion
        sorted_scores = sorted(prediction.raw_scores.items(), key=operator.itemgetter(1), reverse=True)
        suggested_label = sorted_scores[1][0] if len(sorted_scores) > 1 else prediction.label
        
        state["suggested_label"] = suggested_label
        
        clarification_msg = (
            f"[FallbackNode] The model is not confident in its prediction of '{prediction.label}'.\n"
            f"Was this actually a '{suggested_label}' review? (Type 'yes' or 'no'):"
        )
        
        log_data = {
            "event": "clarification_prompted",
            "original_prediction": prediction.label,
            "confidence": prediction.confidence,
            "suggested_label": suggested_label,
            "session_id": state["session_id"],
            "timestamp": datetime.now().isoformat()
        }
        self.logger.info(json.dumps(log_data))

        state["messages"] = add_messages(
            state["messages"],
            [AIMessage(content=clarification_msg)]
        )
        return state

    def process_clarification_node(self, state: WorkflowState) -> WorkflowState:
        """Processes user feedback to correct the label."""
        user_feedback = (state.get("user_feedback") or "").strip().lower()
        original_prediction = state["prediction"]
        suggested_label = state["suggested_label"]
        final_label = ""
        final_confidence = original_prediction.confidence  # original confidence

        if user_feedback == 'yes':
            # User confirms the suggested label (opposite of original prediction)
            final_label = suggested_label
        elif user_feedback == 'no':
            # User disagrees with suggestion, keep original prediction
            final_label = original_prediction.label
        else:
            # Assume any other input is a custom label
            final_label = user_feedback.capitalize() if user_feedback else "Unknown"

        # Create a new, corrected classification result with original confidence
        corrected_result = ClassificationResult(
            label=final_label,
            confidence=final_confidence,  # original confidence score
            raw_scores=original_prediction.raw_scores.copy(),  # original scores
            timestamp=datetime.now().isoformat(),
            input_text=original_prediction.input_text,
            corrected_by_user=True
        )

        state["final_result"] = corrected_result
        
        log_data = {
            "event": "user_correction",
            "original_prediction": original_prediction.label,
            "user_input": user_feedback,
            "final_label": final_label,
            "confidence_preserved": final_confidence,
            "session_id": state["session_id"],
            "timestamp": datetime.now().isoformat()
        }
        self.logger.info(json.dumps(log_data))

        state["messages"] = add_messages(
            state["messages"],
            [HumanMessage(content=state["user_feedback"]),
             AIMessage(content=f"Corrected label based on user input.")]
        )
        return state
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with clarification fallback"""
        workflow = StateGraph(WorkflowState)
        
        # add nodes
        workflow.add_node("inference", self.inference_node)
        workflow.add_node("confidence_check", self.confidence_check_node)
        workflow.add_node("fallback", self.fallback_node)
        workflow.add_node("process_clarification", self.process_clarification_node)


        def route_start(state: WorkflowState) -> str:
            if state.get("user_feedback") is not None:
                return "process_clarification"
            return "inference"

        # reference: https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes
        # Define edges
        workflow.set_conditional_entry_point(route_start)
        workflow.add_edge("inference", "confidence_check")
        
        # refrence: https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-entry-point
        # Conditional routing based on confidence
        def should_fallback(state: WorkflowState) -> str:
            if state.get("needs_clarification"):
                return "fallback"
            return END

        workflow.add_conditional_edges(
            "confidence_check",
            should_fallback,
            {"fallback": "fallback", END: END}
        )
        
        workflow.add_edge("fallback", END)
        workflow.add_edge("process_clarification", END)
        
        return workflow.compile()

class CLIInterface:
    """Beautiful CLI interface for the classification workflow"""
    
    def __init__(self, confidence_threshold: float = 0.75):
        self.console = Console()
        self.workflow_manager = TextClassificationWorkflow(
            confidence_threshold=confidence_threshold
        )
        self.session_counter = 0
        self.logger = Logger.setup_logger("cli_interface")
    
    def _display_header(self):
        """Display beautiful header"""
        header = Panel.fit(
            "[bold blue] Text Classification with Self-Healing Fallback[/bold blue]\n"
            f"[dim]Confidence Threshold: {self.workflow_manager.confidence_threshold:.0%}[/dim]\n"
            "[dim]Logs are saved to: logs/classification_log.jsonl[/dim]",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(header)
    
    def _display_result(self, result: ClassificationResult, step_info: Dict):
        """Display classification result in a beautiful format"""
        
        # Create confidence bar
        confidence_bar = "█" * int(result.confidence * 20) + "░" * (20 - int(result.confidence * 20))
        confidence_color = "green" if result.confidence >= 0.75 else "yellow" if result.confidence >= 0.5 else "red"
        
        # Result panel
        if result.corrected_by_user:
            result_text = f"[bold green]✓ Final Label: {result.label}[/bold green]\n"
            result_text += f"[dim]Corrected via user clarification[/dim]"
            panel_style = "green"
        else:
            result_text = f"[bold {confidence_color}]Final Label: {result.label}[/bold {confidence_color}]\n"
            result_text += f"[{confidence_color}]Confidence: {result.confidence:.1%}[/{confidence_color}] "
            result_text += f"[dim]{confidence_bar}[/dim]"
            panel_style = confidence_color
        
        result_panel = Panel(
            result_text,
            title="Classification Result",
            border_style=panel_style,
            padding=(1, 2)
        )
        
        self.console.print(result_panel)
        
        # Show top predictions if available
        if len(result.raw_scores) > 1:
            table = Table(title="All Predictions", box=box.ROUNDED)
            table.add_column("Label", style="cyan", no_wrap=True)
            table.add_column("Confidence", style="magenta")
            table.add_column("Bar", style="blue")
            
            sorted_scores = sorted(result.raw_scores.items(), key=lambda x: x[1], reverse=True)
            for label, score in sorted_scores[:5]:  # Show top 5
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                table.add_row(label, f"{score:.1%}", bar)
            
            self.console.print(table)
    
    def run_classification(self, input_text: str):
        """Run the complete classification workflow with beautiful output"""
        self.session_counter += 1
        session_id = f"session_{self.session_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Log session start
        self.logger.info(json.dumps({
            "event": "session_started",
            "session_id": session_id,
            "input_text": input_text,
            "timestamp": datetime.now().isoformat()
        }))
        
        current_state = WorkflowState(
            input_text=input_text,
            prediction=None,
            confidence_threshold=self.workflow_manager.confidence_threshold,
            needs_clarification=False,
            user_feedback=None,
            final_result=None,
            messages=[HumanMessage(content=f"Input: {input_text}")],
            session_id=session_id
        )
        
        # Display input
        input_panel = Panel(
            f"[bold white]{input_text}[/bold white]",
            title="📝 Input Text",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(input_panel)
        
        step_info = {"steps": [], "fallback_triggered": False}
        
        # Main workflow loop
        while current_state.get("final_result") is None:
            for step in self.workflow_manager.workflow.stream(current_state):
                node_name = list(step.keys())[0]
                
                if node_name != "__end__":
                    current_state = step[node_name]
                    step_info["steps"].append(node_name)
                    
                    # Show processing step
                    if node_name == "inference":
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=self.console,
                            transient=True
                        ) as progress:
                            task = progress.add_task("Running inference...", total=None)
                            # Small delay for visual effect
                            import time
                            time.sleep(0.5)
                    
                    elif node_name == "confidence_check":
                        pred = current_state.get("prediction")
                        if pred:
                            conf_color = "green" if pred.confidence >= 0.75 else "yellow" if pred.confidence >= 0.5 else "red"
                            self.console.print(f"[{conf_color}]Checking confidence: {pred.confidence:.1%}[/{conf_color}]")
            
            # Handle clarification if needed
            if current_state.get("needs_clarification"):
                step_info["fallback_triggered"] = True
                
                # Show fallback panel
                pred = current_state.get("prediction")
                suggested = current_state.get("suggested_label")
                
                fallback_text = f"[yellow] Low confidence prediction:[/yellow] [bold]{pred.label}[/bold] ({pred.confidence:.1%})\n\n"
                fallback_text += f"[dim]Was this actually a '{suggested}' review?[/dim]"
                
                fallback_panel = Panel(
                    fallback_text,
                    title=" Need Clarification",
                    border_style="yellow",
                    padding=(1, 2)
                )
                self.console.print(fallback_panel)
                
                # Get user input with rich prompt
                user_response = Prompt.ask(
                    "[bold cyan]Your response[/bold cyan] [dim](yes/no)[/dim]",
                    choices=["yes", "no"],
                    show_choices=False
                ).strip()
                
                current_state["user_feedback"] = user_response
                current_state["needs_clarification"] = False
            
            # Safety break
            if "__end__" in step:
                if not current_state.get("final_result"):
                    if current_state.get("user_feedback") is not None:
                        continue
                    else:
                        break
        
        result = current_state.get("final_result")
        
        if not result:
            self.console.print("[red]Error: Workflow finished without producing a final result.[/red]")
            raise RuntimeError("Workflow finished without producing a final result.")
        
        # Log session completion
        self.logger.info(json.dumps({
            "event": "session_completed",
            "session_id": session_id,
            "final_label": result.label,
            "confidence": result.confidence,
            "corrected_by_user": result.corrected_by_user,
            "fallback_triggered": step_info["fallback_triggered"],
            "steps_executed": step_info["steps"],
            "timestamp": datetime.now().isoformat()
        }))
        
        # Display beautiful result
        self._display_result(result, step_info)
        
        return result

    def interactive_mode(self):
        """Run interactive CLI mode with beautiful interface"""
        self._display_header()
        
        self.console.print("\n[dim]Type 'quit', 'exit', or 'q' to exit[/dim]\n")
        
        while True:
            try:
                user_input = Prompt.ask("[bold green]Enter text to classify[/bold green]").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.console.print("\n[bold blue]Goodbye![/bold blue]")
                    break
                
                if not user_input:
                    continue
                
                self.console.print()  # Add spacing
                self.run_classification(user_input)
                
                # Separator
                self.console.print("\n" + "─" * 60 + "\n")
                
            except KeyboardInterrupt:
                self.console.print("\n[bold blue] Goodbye![/bold blue]")
                break
            except Exception as e:
                self.console.print(f"[red] An error occurred: {e}[/red]")
                self.logger.error(json.dumps({
                    "event": "cli_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }))

def main():
    """Main entry point for the CLI."""
    console = Console()
    
    try:
        cli = CLIInterface(confidence_threshold=0.75)
        
        # Run demonstration
        console.print(Panel.fit(
            "[bold yellow] Demonstration Mode[/bold yellow]\n"
            "[dim]Running example: 'It was okay I guess.'[/dim]",
            border_style="yellow"
        ))
        
        example_text = "It was okay I guess."
        cli.run_classification(example_text)
        
        console.print("\n" + "═" * 80 + "\n")
        
        # Start interactive session
        cli.interactive_mode()
        
    except Exception as e:
        console.print(f"[red] Fatal error: {e}[/red]")
        Logger.setup_logger("main").error(json.dumps({
            "event": "fatal_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))

if __name__ == "__main__":
    main()