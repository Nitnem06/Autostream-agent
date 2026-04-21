# 🎬 AutoStream AI Agent

An intelligent conversational AI agent for AutoStream — a content creation platform. Built with **LangGraph**, **FastAPI**, and **RAG** for context-aware responses.

## ✨ Features
- 🧠 Multi-intent detection (greeting, product query, high-intent lead)
- 📚 RAG-powered responses from a knowledge base
- 🎯 Automated lead capture (name, email, platform)
- 🔄 Resilient LLM with Anthropic primary + multi-provider fallback
- 💬 Real-time chat via WebSocket
- 🎨 Modern dashboard-style UI

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, LangGraph, LangChain
- **LLMs:** Anthropic Claude, OpenAI, Gemini, Groq
- **Frontend:** HTML, CSS, Vanilla JS (WebSocket)
- **RAG:** Custom retriever over markdown knowledge base

## 📁 Project Structure
Autostream-agent/

├── agent/           # LangGraph nodes, state, prompts

├── rag/             # Retriever logic

├── tools/           # Lead capture tool

├── knowledge_base/  # Markdown KB

├── static/          # Frontend UI

├── server.py        # FastAPI + WebSocket server

└── main.py          # CLI entry point


## 🚀 Setup

1. Clone the repo:
   git clone https://github.com/YourUsername/Autostream-agent.git
   cd Autostream-agent

## Create a virtual environment and install dependencies:

    python -m venv .venv
    .venv\Scripts\activate   # Windows
    pip install -r requirements.txt

## Create a .env file with your API keys:

    ANTHROPIC_API_KEY=your_key_here
    OPENAI_API_KEY=your_key_here

## Run the server:

    python server.py

## Open http://localhost:8000 in your browser.

    📝 License
    MIT

    Feel free to customize it.
