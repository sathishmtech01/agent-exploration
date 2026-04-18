"""RootOrchestrator — the top-level planner and coordinator.

This is the entry point for all user queries. It:
  1. Knows about ALL agents — both specialists and the DomainOrchestrator.
  2. Plans which agents to call (LLM-powered, with full reasoning + context).
  3. Routes: may send a query to one specialist directly, or delegate the
     entire domain sub-problem to the DomainOrchestrator (which has its own
     nested planning loop).
  4. Executes agents sequentially or indicates parallel paths.
  5. Consolidates all results via summary_agent.
  6. Records learnings — success and failure — so future plans improve.

The RootOrchestrator extends BaseOrchestrator, so the planner + coordinator
loop is inherited. The difference from DomainOrchestrator is:
  - Broader agent registry (includes domain_orchestrator itself)
  - Cross-domain routing (can combine tech + language + data etc.)
  - Top-level session ownership
"""

from __future__ import annotations

from agents.base_orchestrator import BaseOrchestrator
from agents.domain_orchestrator import build_domain_orchestrator
from agents.registry import AgentCapability, AgentRegistry
from agents.specialist_agents import (
    build_language_agent,
    build_research_agent,
    build_summary_agent,
)
from config.llm_config import LLMConfig, get_llm_config
from context.context_manager import ContextManager
from reward.reward_engine import RewardEngine
from security.security_layer import SecurityLayer
from trace.tracer import AgentTracer


class RootOrchestrator(BaseOrchestrator):
    """
    Top-level orchestrator — the system entry point.

    Registered agents:
      - domain_orchestrator  : handles all tech-domain queries (research+code+data)
      - language_agent       : translation, content generation
      - research_agent       : direct research bypass for simple lookups
      - summary_agent        : cross-domain consolidation
    """

    pass  # Full implementation in BaseOrchestrator


def build_root_orchestrator(
    context_manager: ContextManager,
    tracer: AgentTracer,
    llm_config: LLMConfig | None = None,
    security: SecurityLayer | None = None,
    reward_engine: RewardEngine | None = None,
) -> RootOrchestrator:
    """
    Build and return a fully wired RootOrchestrator.

    The registry includes:
      - domain_orchestrator  (Agent 2 — a full orchestrator for tech domain)
      - language_agent       (Agent 3 — translation / content)
      - research_agent       (Agent 4 — direct research)
      - summary_agent        (Agent 5 — consolidation)
    """
    cfg = llm_config or get_llm_config()
    # Shared security + reward — created once, passed to all orchestrators
    sec = security or SecurityLayer()
    reward = reward_engine or RewardEngine()
    registry = AgentRegistry()

    # ── Agent 2: DomainOrchestrator ─────────────────────────────────────────
    # Re-uses the same shared context_manager, tracer, AND llm_config so all
    # nested traces appear in the same session using the same provider.
    domain_orch = build_domain_orchestrator(
        context_manager, tracer, llm_config=cfg, security=sec, reward_engine=reward
    )

    registry.register(
        domain_orch,
        AgentCapability(
            name="domain_orchestrator",
            description=(
                "Handles ALL tech-domain tasks by internally planning and "
                "coordinating research_agent, code_agent, and data_agent. "
                "Route here when the query involves coding, algorithms, "
                "data analysis, technical research, or any combination thereof."
            ),
            domain="tech",
            capabilities=[
                "code generation", "technical research", "data analysis",
                "algorithm design", "debugging", "system design", "statistics",
            ],
            input_types=["tech questions", "coding tasks", "data problems", "mixed tech queries"],
            output_types=["integrated tech responses"],
            is_orchestrator=True,
            priority=9,
        ),
    )

    # ── Agent 3: language_agent ──────────────────────────────────────────────
    language = build_language_agent(model=cfg.agent_model)
    registry.register(
        language,
        AgentCapability(
            name="language_agent",
            description="Handles translation, content generation, and language tasks",
            domain="language",
            capabilities=["translation", "content writing", "summarisation", "paraphrasing"],
            input_types=["text to translate", "content requests", "language tasks"],
            output_types=["translated text", "written content"],
            priority=8,
        ),
    )

    # ── Agent 4: research_agent (direct, no tech orchestration) ─────────────
    research = build_research_agent(model=cfg.agent_model)
    registry.register(
        research,
        AgentCapability(
            name="research_agent",
            description=(
                "Direct general-knowledge research — use for factual lookups, "
                "explanations, and background context that do NOT require code or data analysis."
            ),
            domain="general",
            capabilities=["fact retrieval", "general research", "explanations", "comparisons"],
            input_types=["factual questions", "concept explanations", "background research"],
            output_types=["research summaries", "fact sheets"],
            priority=6,
        ),
    )

    # ── Agent 5: summary_agent ──────────────────────────────────────────────
    summary = build_summary_agent(model=cfg.agent_model)
    registry.register(
        summary,
        AgentCapability(
            name="summary_agent",
            description="Synthesises multi-domain results into a single coherent answer",
            domain="general",
            capabilities=["consolidation", "synthesis", "cross-domain merging"],
            input_types=["multiple agent outputs"],
            output_types=["unified response"],
            priority=5,
        ),
    )

    orchestrator = RootOrchestrator(
        name="root_orchestrator",
        description=(
            "Top-level planner and coordinator. Understands the full query, "
            "decides routing across all available agents and domain orchestrators, "
            "and consolidates a final answer."
        ),
        orchestrator_name=f"RootOrchestrator[{cfg.provider}]",
        domain="root",
        specialist_agent_names=[
            "domain_orchestrator",
            "language_agent",
            "research_agent",
            "summary_agent",
        ],
        registry=registry,
        context_manager=context_manager,
        tracer=tracer,
        llm_config=cfg,
        security=sec,
        reward_engine=reward,
    )

    return orchestrator
