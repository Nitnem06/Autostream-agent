"""
AutoStream — FastAPI WebSocket Server
--------------------------------------
Serves the UI at http://localhost:8000
WebSocket endpoint at ws://localhost:8000/ws/{session_id}

Run with:
    python server.py
"""

import json
import os
import sys
import uuid
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("[ERROR] ANTHROPIC_API_KEY not found in .env file.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Import agent — do this after env check so we don't hit API errors early
# ---------------------------------------------------------------------------
from agent.graph import build_graph
from agent.state import AgentState, LeadInfo

# ---------------------------------------------------------------------------
# Build graph once at startup (also initialises RAG index)
# ---------------------------------------------------------------------------
print("[Server] Building agent graph and loading RAG index...")
graph = build_graph()
print("[Server] Agent ready.")

# ---------------------------------------------------------------------------
# In-memory session store: session_id → AgentState
# In production this would be Redis
# ---------------------------------------------------------------------------
sessions: dict[str, AgentState] = {}


def new_state() -> AgentState:
    """Return a fresh AgentState for a new session."""
    return AgentState(
        messages=[],
        intent="",
        lead_info=LeadInfo(name=None, email=None, platform=None),
        pending_field=None,
        lead_captured=False,
        retrieved_context="",
    )


def get_last_ai_message(state: AgentState) -> str:
    """Extract the most recent AIMessage content from state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AutoStream AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "ready"}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Initialise session if new
    if session_id not in sessions:
        sessions[session_id] = new_state()

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            action = payload.get("action", "message")

            # ── Reset session ───────────────────────────────────────────────
            if action == "reset":
                sessions[session_id] = new_state()
                await websocket.send_text(json.dumps({
                    "type": "reset_ack",
                    "message": "Conversation reset.",
                }))
                continue

            # ── Process user message ────────────────────────────────────────
            user_text = payload.get("message", "").strip()
            if not user_text:
                continue

            state = sessions[session_id]

            # Append user message to state
            msgs = list(state.get("messages", []))
            msgs.append(HumanMessage(content=user_text))
            state["messages"] = msgs

            # Notify frontend: typing in progress
            await websocket.send_text(json.dumps({"type": "typing"}))

            # Run LangGraph
            try:
                state = graph.invoke(state)
                sessions[session_id] = state
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Agent error: {str(e)}. Please try again.",
                }))
                # Roll back the user message we appended
                state["messages"] = state["messages"][:-1]
                continue

            # Collect lead_info safely
            raw_info = state.get("lead_info") or {}
            lead_info = {
                "name": raw_info.get("name"),
                "email": raw_info.get("email"),
                "platform": raw_info.get("platform"),
            }

            # Send full response payload
            await websocket.send_text(json.dumps({
                "type": "response",
                "message": get_last_ai_message(state),
                "intent": state.get("intent", ""),
                "lead_info": lead_info,
                "pending_field": state.get("pending_field"),
                "lead_captured": state.get("lead_captured", False),
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Unexpected error for session {session_id}: {e}")


# ---------------------------------------------------------------------------
# Serve static folder (must be after route definitions)
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AutoStream AI Agent — Web UI Mode")
    print("=" * 55)
    print("  Open your browser at: http://localhost:8000")
    print("  Press Ctrl+C to stop.\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")