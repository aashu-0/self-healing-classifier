"""CLI interface for the classification system."""
from models import ClassificationResult, WorkflowState
from logger import Logger
from nodes import TextClassificationWorkflow

from datetime import datetime
from typing import Dict
import json
from langchain_core.messages import HumanMessage

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich import box


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
            title="Input Text",
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