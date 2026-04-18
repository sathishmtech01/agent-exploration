"""ADK Web UI entry point.

ADK web discovers this file automatically.
It must expose a module-level `root_agent` variable.

Run with:
  adk web /path/to/multi_agent_system/..  (parent dir)
  or
  adk web --reload /path/to/multi_agent_system/..
"""

from __future__ import annotations

import os
import sys

# Ensure all local modules (models, agents, config, etc.) are importable
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from agents.root_orchestrator import build_root_orchestrator
from config.llm_config import get_llm_config
from context.context_manager import ContextManager
from reward.reward_engine import RewardEngine
from security.security_layer import SecurityLayer
from trace.tracer import AgentTracer

# ── Bootstrap ──────────────────────────────────────────────────────────────
_llm_config = get_llm_config()
_context_manager = ContextManager()
_tracer = AgentTracer()
_security = SecurityLayer()
_reward_engine = RewardEngine()

# ── root_agent — this is what ADK web exposes in the chat UI ───────────────
root_agent = build_root_orchestrator(
    context_manager=_context_manager,
    tracer=_tracer,
    llm_config=_llm_config,
    security=_security,
    reward_engine=_reward_engine,
)
