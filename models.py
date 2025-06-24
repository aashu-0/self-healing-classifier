from typing import Dict, Optional, TypedDict
from dataclasses import dataclass
from langchain_core.messages import BaseMessage

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

