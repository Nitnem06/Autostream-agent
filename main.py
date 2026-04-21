import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import get_graph
from agent.state import initial_state
from rag.retriever import get_vectorstore

load_dotenv()


def print_banner():
    print("\n" + "=" * 60)
    print("  🎬  AutoStream AI Agent")
    print("  AI-powered video editing for content creators")
    print("=" * 60)
    print("  Type your message. Commands: /reset, /state, /exit")
    print("=" * 60 + "\n")


def print_state_debug(state):
    print("\n--- STATE ---")
    print(f"  Intent         : {state.get('intent')}")
    print(f"  Pending field  : {state.get('pending_field')}")
    print(f"  Lead info      : {state.get('lead_info')}")
    print(f"  Lead captured  : {state.get('lead_captured')}")
    print("-------------\n")


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    print_banner()

    # Pre-build vectorstore so first query isn't slow
    get_vectorstore()

    graph = get_graph()
    state = initial_state()

    print("Agent: Hey! I'm the AutoStream assistant. How can I help you today?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            state = initial_state()
            print("\n[State reset]\n")
            continue

        if user_input.lower() == "/state":
            print_state_debug(state)
            continue

        # Append user message and invoke graph
        state["messages"].append(HumanMessage(content=user_input))

        try:
            new_state = graph.invoke(state)
            state = new_state
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            continue

        # Print the last AI message
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\nAgent: {msg.content}\n")
                break


if __name__ == "__main__":
    main()