"""LangGraph nodes for the classification workflow."""

from models import ClassificationResult, WorkflowState
from logger import Logger

from datetime import datetime
import json
import operator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


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