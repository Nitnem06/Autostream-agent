"""
LLM Provider Resilience Layer.

Implements a 4-tier provider failover chain to ensure zero-downtime
operation even under LLM provider outages, rate limits, or auth failures.

Priority chain:
    1. Anthropic Claude 3 Haiku    (Primary - best structured reasoning)
    2. OpenAI GPT-4o-mini           (Fallback 1 - high reliability)
    3. Google Gemini 1.5 Flash      (Fallback 2 - generous free tier)
    4. Groq Llama 3.1 8B            (Fallback 3 - ultra-fast free tier)

All providers expose a unified LangChain ChatModel interface, so
downstream node logic remains provider-agnostic.
"""

import os
import logging
from typing import List, Union, Optional, Any
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

load_dotenv()

logger = logging.getLogger("autostream.llm")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")


class ResilientLLM:
    """
    Wraps a chain of LLM providers with automatic failover.
    Drop-in replacement for any LangChain ChatModel — exposes .invoke().
    """

    def __init__(self):
        # Ordered chain: (name, init_callable)
        self.providers: List[tuple[str, Any]] = []
        self.active_provider: Optional[str] = None

        self._init_anthropic()
        self._init_openai()
        self._init_gemini()
        self._init_groq()

        if not self.providers:
            raise RuntimeError(
                "No LLM providers available. Set at least one of: "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY."
            )

        logger.info(
            f"Resilient LLM initialized with {len(self.providers)} provider(s): "
            f"{[name for name, _ in self.providers]}"
        )

    # ---------- Provider initializers ----------

    def _init_anthropic(self):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0.3,
                max_tokens=1024,
                api_key=key,
            )
            self.providers.append(("anthropic", llm))
            logger.info("✓ Anthropic Claude 3 Haiku ready (primary)")
        except Exception as e:
            logger.warning(f"✗ Anthropic init failed: {e}")

    def _init_openai(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=1024,
                api_key=key,
            )
            self.providers.append(("openai", llm))
            logger.info("✓ OpenAI GPT-4o-mini ready (fallback 1)")
        except Exception as e:
            logger.warning(f"✗ OpenAI init failed: {e}")

    def _init_gemini(self):
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            return
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                max_output_tokens=1024,
                google_api_key=key,
            )
            self.providers.append(("gemini", llm))
            logger.info("✓ Google Gemini 1.5 Flash ready (fallback 2)")
        except Exception as e:
            logger.warning(f"✗ Gemini init failed: {e}")

    def _init_groq(self):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=1024,
                api_key=key,
            )
            self.providers.append(("groq", llm))
            logger.info("✓ Groq Llama 3.1 8B ready (fallback 3)")
        except Exception as e:
            logger.warning(f"✗ Groq init failed: {e}")

    # ---------- Unified invoke with failover ----------

    def invoke(self, messages: Union[List[BaseMessage], str], **kwargs):
        """
        Try each provider in priority order. On failure, automatically
        failover to the next. Raises only if ALL providers fail.
        """
        last_error: Optional[Exception] = None

        for name, llm in self.providers:
            try:
                result = llm.invoke(messages, **kwargs)

                # Log provider switch
                if self.active_provider != name:
                    if self.active_provider is None:
                        logger.info(f"🟢 Active provider: {name}")
                    else:
                        logger.info(
                            f"🔄 Failover successful: {self.active_provider} → {name}"
                        )
                    self.active_provider = name

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"⚠️  Provider '{name}' failed: "
                    f"{type(e).__name__}: {str(e)[:150]}"
                )
                # Try next provider
                continue

        # All providers exhausted
        logger.error("❌ All LLM providers failed.")
        raise RuntimeError(
            f"All {len(self.providers)} LLM providers failed. "
            f"Last error: {last_error}"
        ) from last_error


# ---------- Module-level singleton ----------

_llm_instance: Optional[ResilientLLM] = None


def get_llm() -> ResilientLLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ResilientLLM()
    return _llm_instance