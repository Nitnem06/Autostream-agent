from typing import TypedDict, Optional, List, Annotated
from langgraph.graph.message import add_messages


class LeadInfo(TypedDict):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[str]
    lead_info: LeadInfo
    pending_field: Optional[str]
    lead_captured: bool
    retrieved_context: Optional[str]
    last_asked_field: Optional[str]


def initial_state() -> AgentState:
    return {
        "messages": [],
        "intent": None,
        "lead_info": {"name": None, "email": None, "platform": None},
        "pending_field": None,
        "lead_captured": False,
        "retrieved_context": None,
        "last_asked_field": None,
    }