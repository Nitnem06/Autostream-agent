import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.state import AgentState
from agent.prompts import (
    INTENT_DETECTION_PROMPT,
    GREETING_PROMPT,
    RAG_SYSTEM_PROMPT,
    LEAD_INIT_PROMPT,
    FIELD_EXTRACTION_PROMPT,
    ASK_FIELD_PROMPT,
    REPROMPT_FIELD_PROMPT,
    LEAD_CAPTURED_CONFIRMATION,
)
from rag.retriever import retrieve_context
from tools.lead_capture import mock_lead_capture, is_valid_email

load_dotenv()

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0.3,
    max_tokens=1024,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _recent_history(state: AgentState, n: int = 4):
    return state["messages"][-n:] if len(state["messages"]) >= n else state["messages"]


# ---------- ROUTING ----------

def route_entry(state: AgentState) -> str:
    """Entry router: if mid-collection, go straight to field extraction."""
    if state.get("lead_captured"):
        # After capture, everything is a normal query
        return "detect_intent"
    if state.get("pending_field"):
        return "extract_field"
    return "detect_intent"


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "product_query")
    if intent == "greeting":
        return "handle_greeting"
    if intent == "high_intent" and not state.get("lead_captured"):
        return "handle_lead_init"
    return "handle_rag"


def route_after_extraction(state: AgentState) -> str:
    """After extracting a field, decide next step."""
    info = state["lead_info"]
    if all([info.get("name"), info.get("email"), info.get("platform")]):
        return "execute_lead_capture"
    return "ask_next_field"


# ---------- NODES ----------

def detect_intent(state: AgentState) -> dict:
    user_msg = _last_user_message(state)
    prompt = INTENT_DETECTION_PROMPT.format(message=user_msg)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()

    # Sanitize
    for valid in ["high_intent", "product_query", "greeting"]:
        if valid in raw:
            return {"intent": valid}
    return {"intent": "product_query"}


def handle_greeting(state: AgentState) -> dict:
    user_msg = _last_user_message(state)
    prompt = GREETING_PROMPT.format(message=user_msg)
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}


def handle_rag(state: AgentState) -> dict:
    user_msg = _last_user_message(state)
    context = retrieve_context(user_msg, k=3)
    system = RAG_SYSTEM_PROMPT.format(context=context)

    history = _recent_history(state, 4)
    messages = [SystemMessage(content=system)] + history
    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content)],
        "retrieved_context": context,
    }


def handle_lead_init(state: AgentState) -> dict:
    # Guard: don't re-trigger if already collecting
    if state.get("pending_field"):
        return {}

    response = llm.invoke([HumanMessage(content=LEAD_INIT_PROMPT)])
    return {
        "messages": [AIMessage(content=response.content)],
        "pending_field": "name",
        "last_asked_field": "name",
        "lead_info": {"name": None, "email": None, "platform": None},
    }


def extract_field(state: AgentState) -> dict:
    field = state["pending_field"]
    user_msg = _last_user_message(state)
    prompt = FIELD_EXTRACTION_PROMPT.format(field=field, message=user_msg)
    response = llm.invoke([HumanMessage(content=prompt)])
    extracted = response.content.strip()

    # Clean up common LLM artifacts
    extracted = extracted.strip("\"'.,!? \n\t")

    if not extracted or "NOT_FOUND" in extracted.upper() or len(extracted) > 100:
        # Extraction failed — keep pending_field to re-ask
        return {}

    # Validate email specifically
    if field == "email" and not is_valid_email(extracted):
        return {}

    lead_info = dict(state["lead_info"])
    lead_info[field] = extracted

    return {
        "lead_info": lead_info,
        "pending_field": None,
    }


def ask_next_field(state: AgentState) -> dict:
    info = state["lead_info"]
    order = ["name", "email", "platform"]

    next_field = None
    for f in order:
        if not info.get(f):
            next_field = f
            break

    if not next_field:
        return {}

    # Was this a re-ask (extraction failed on same field)?
    is_reprompt = state.get("last_asked_field") == next_field

    if is_reprompt:
        prompt = REPROMPT_FIELD_PROMPT.format(field=next_field)
    else:
        collected = {k: v for k, v in info.items() if v}
        collected_str = ", ".join([f"{k}={v}" for k, v in collected.items()]) or "nothing yet"
        prompt = ASK_FIELD_PROMPT.format(collected=collected_str, next_field=next_field)

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=response.content)],
        "pending_field": next_field,
        "last_asked_field": next_field,
    }


def execute_lead_capture(state: AgentState) -> dict:
    info = state["lead_info"]
    result = mock_lead_capture(info["name"], info["email"], info["platform"])

    if not result.get("success"):
        # Email invalid — clear it and re-collect
        if result.get("error") == "Invalid email":
            lead_info = dict(info)
            lead_info["email"] = None
            return {
                "lead_info": lead_info,
                "pending_field": "email",
                "last_asked_field": None,  # force fresh ask
                "messages": [AIMessage(content="That email doesn't look quite right. Could you double-check it?")],
            }
        return {
            "messages": [AIMessage(content="Hmm, something went wrong saving your info. Let's try again shortly.")],
        }

    prompt = LEAD_CAPTURED_CONFIRMATION.format(
        name=result["name"],
        email=result["email"],
        platform=result["platform"],
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=response.content)],
        "lead_captured": True,
        "pending_field": None,
    }