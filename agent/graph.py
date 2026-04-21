from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    detect_intent,
    handle_greeting,
    handle_rag,
    handle_lead_init,
    extract_field,
    ask_next_field,
    execute_lead_capture,
    route_entry,
    route_after_intent,
    route_after_extraction,
)


def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_rag", handle_rag)
    graph.add_node("handle_lead_init", handle_lead_init)
    graph.add_node("extract_field", extract_field)
    graph.add_node("ask_next_field", ask_next_field)
    graph.add_node("execute_lead_capture", execute_lead_capture)

    # Entry routing via conditional edge from a virtual start
    graph.set_conditional_entry_point(
        route_entry,
        {
            "detect_intent": "detect_intent",
            "extract_field": "extract_field",
        },
    )

    # After intent detection
    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_rag": "handle_rag",
            "handle_lead_init": "handle_lead_init",
        },
    )

    # Lead init -> ask name
    graph.add_edge("handle_lead_init", "ask_next_field")

    # After extraction, either capture or ask next
    graph.add_conditional_edges(
        "extract_field",
        route_after_extraction,
        {
            "execute_lead_capture": "execute_lead_capture",
            "ask_next_field": "ask_next_field",
        },
    )

    # Terminal nodes
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_rag", END)
    graph.add_edge("ask_next_field", END)
    graph.add_edge("execute_lead_capture", END)

    return graph.compile()


# Singleton
_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph