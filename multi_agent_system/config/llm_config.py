"""LLM provider configuration.

Reads LLM_PROVIDER from the environment and returns the correct:
  - model string for LlmAgent (ADK LiteLLM format)
  - client for direct orchestrator calls (planner / consolidation)

Supported providers
-------------------
  gemini  → google-genai client   + gemini-2.0-flash (default)
  openai  → openai.OpenAI client  + gpt-4o-mini
  groq    → openai.OpenAI client  + groq/llama-3.3-70b-versatile

Set in .env:
  LLM_PROVIDER=openai        # or groq / gemini
  OPENAI_API_KEY=sk-...
  GROQ_API_KEY=gsk_...
  GOOGLE_API_KEY=AIza...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str          # "gemini" | "openai" | "groq"
    agent_model: str       # model string for LlmAgent
    planner_model: str     # model string for direct API calls
    client: Any            # ready-to-use API client
    client_type: str       # "genai" | "openai_sdk"


def get_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    if provider == "openai":
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in .env")
        client = openai.AsyncOpenAI(api_key=api_key)
        return LLMConfig(
            provider="openai",
            agent_model="openai/gpt-4o-mini",       # LiteLLM prefix for LlmAgent
            planner_model="gpt-4o-mini",             # native model name for OpenAI SDK
            client=client,
            client_type="openai_sdk",
        )

    if provider == "groq":
        import openai
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in .env")
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        return LLMConfig(
            provider="groq",
            agent_model="groq/llama-3.3-70b-versatile",   # LiteLLM prefix for LlmAgent
            planner_model="llama-3.3-70b-versatile",       # native model name for Groq
            client=client,
            client_type="openai_sdk",
        )

    # Default: Gemini
    import google.genai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set in .env")
    client = genai.Client(api_key=api_key)
    return LLMConfig(
        provider="gemini",
        agent_model="gemini-2.0-flash",
        planner_model="gemini-2.0-flash",
        client=client,
        client_type="genai",
    )
