"""DomainOrchestrator — Agent 2.

This is a domain-scoped orchestrator that MIMICS the RootOrchestrator's
planner+coordinator pattern but operates within a single domain (e.g. "tech").

It has its own:
  - AgentRegistry (only domain-relevant specialists)
  - Planning step (LLM produces an ExecutionPlan for the domain)
  - Routing + execution loop
  - Consolidation via summary_agent
  - Learning store contributions

The RootOrchestrator can route an entire query to the DomainOrchestrator,
which then handles the intra-domain planning and execution transparently.
When the RootOrchestrator does this, the trace shows two nested plan trees:
  Root plan → DomainOrchestrator → Domain plan → specialist agents

Factory function `build_domain_orchestrator` wires everything together.
"""

from __future__ import annotations

from agents.base_orchestrator import BaseOrchestrator
from agents.registry import AgentCapability, AgentRegistry
from agents.specialist_agents import (
    build_code_agent,
    build_data_agent,
    build_research_agent,
    build_summary_agent,
)
from config.llm_config import LLMConfig, get_llm_config
from context.context_manager import ContextManager
from trace.tracer import AgentTracer


class DomainOrchestrator(BaseOrchestrator):
    """
    Tech-domain orchestrator that plans and coordinates:
      research_agent → code_agent → data_agent

    This class inherits the full planner + coordinator loop from BaseOrchestrator,
    making it a first-class orchestrator that can be called by the RootOrchestrator
    or invoked directly.
    """

    pass  # All logic lives in BaseOrchestrator — domain config injected at build time


def build_domain_orchestrator(
    context_manager: ContextManager,
    tracer: AgentTracer,
    llm_config: LLMConfig | None = None,
) -> DomainOrchestrator:
    """
    Build and return a fully wired DomainOrchestrator for the 'tech' domain.

    Registers:
      - research_agent
      - code_agent
      - data_agent
      - summary_agent (consolidation)
    """
    cfg = llm_config or get_llm_config()
    registry = AgentRegistry()

    # ── Specialist agents ───────────────────────────────────────────────────
    research = build_research_agent(model=cfg.agent_model)
    code = build_code_agent(model=cfg.agent_model)
    data = build_data_agent(model=cfg.agent_model)
    summary = build_summary_agent(model=cfg.agent_model)

    registry.register(
        research,
        AgentCapability(
            name="research_agent",
            description="Researches topics, retrieves facts, synthesises knowledge",
            domain="tech",
            capabilities=["research", "knowledge retrieval", "fact finding", "literature review"],
            input_types=["questions", "topics", "technical concepts"],
            output_types=["research summaries", "fact sheets", "explanations"],
            priority=7,
        ),
    )
    registry.register(
        code,
        AgentCapability(
            name="code_agent",
            description="Writes, reviews, explains, and debugs code in any language",
            domain="tech",
            capabilities=["code generation", "debugging", "code review", "algorithms", "API design"],
            input_types=["coding tasks", "algorithms", "system design requirements"],
            output_types=["code", "documentation", "design patterns"],
            priority=9,
        ),
    )
    registry.register(
        data,
        AgentCapability(
            name="data_agent",
            description="Analyses data, finds patterns, performs statistical analysis",
            domain="tech",
            capabilities=["data analysis", "statistics", "pattern detection", "visualisation"],
            input_types=["datasets", "metrics", "analytical questions"],
            output_types=["analysis reports", "statistical summaries", "chart suggestions"],
            priority=8,
        ),
    )
    registry.register(
        summary,
        AgentCapability(
            name="summary_agent",
            description="Consolidates outputs from multiple agents into a unified response",
            domain="tech",
            capabilities=["summarisation", "synthesis", "consolidation"],
            input_types=["multiple agent outputs"],
            output_types=["unified response"],
            priority=5,
        ),
    )

    orchestrator = DomainOrchestrator(
        name="domain_orchestrator",
        description=(
            "Tech-domain orchestrator. Plans and coordinates research, coding, "
            "and data analysis tasks. Mimics the root orchestrator pattern "
            "within the tech domain."
        ),
        orchestrator_name=f"DomainOrchestrator[tech/{cfg.provider}]",
        domain="tech",
        specialist_agent_names=["research_agent", "code_agent", "data_agent", "summary_agent"],
        registry=registry,
        context_manager=context_manager,
        tracer=tracer,
        llm_config=cfg,
    )

    return orchestrator
