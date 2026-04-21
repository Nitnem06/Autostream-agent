import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter

KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "Autostream_kb.md"
_vectorstore = None
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _build_vectorstore():
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at: {KB_PATH}")

    loader = TextLoader(str(KB_PATH), encoding="utf-8")
    docs = loader.load()

    splitter = MarkdownTextSplitter(chunk_size=450, chunk_overlap=60)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, _get_embeddings())
    return vs


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("[RAG] Building FAISS vectorstore from knowledge base...")
        _vectorstore = _build_vectorstore()
        print(f"[RAG] Vectorstore ready.")
    return _vectorstore


def retrieve_context(query: str, k: int = 3) -> str:
    vs = get_vectorstore()
    results = vs.similarity_search(query, k=k)
    return "\n\n---\n\n".join([doc.page_content for doc in results])