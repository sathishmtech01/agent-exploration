"""Agent registry — the orchestrator's map of the entire agent ecosystem.

The registry gives the planner a structured description of every available
agent so it can make informed routing decisions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field


class AgentCapability(BaseModel):
    """Describes what an agent can do — used by the planner when building the execution plan."""

    name: str
    description: str
    domain: str
    capabilities: List[str]
    input_types: List[str]
    output_types: List[str]
    is_orchestrator: bool = False
    priority: int = Field(default=5, ge=1, le=10)


class AgentRegistry:
    """Central registry: every agent + its capability profile."""

    def __init__(self):
        self._agents: Dict[str, LlmAgent] = {}
        self._caps: Dict[str, AgentCapability] = {}

    # ---------------------------------------------------------------- register

    def register(self, agent: LlmAgent, capability: AgentCapability) -> None:
        self._agents[agent.name] = agent
        self._caps[agent.name] = capability

    # ---------------------------------------------------------------- access

    def get_agent(self, name: str) -> Optional[LlmAgent]:
        return self._agents.get(name)

    def get_capability(self, name: str) -> Optional[AgentCapability]:
        return self._caps.get(name)

    def list_agent_names(self) -> List[str]:
        return list(self._agents.keys())

    def list_non_orchestrators(self) -> List[str]:
        return [n for n, c in self._caps.items() if not c.is_orchestrator]

    # ------------------------------------------ context description for planner

    def context_for_planner(self) -> str:
        """Full agent map injected into the planner prompt."""
        lines = [
            "=== AVAILABLE AGENTS ===",
            "(Use this information to build the execution plan)\n",
        ]
        for name, cap in self._caps.items():
            orch_tag = " [ORCHESTRATOR]" if cap.is_orchestrator else ""
            lines.append(f"Agent: {name}{orch_tag}")
            lines.append(f"  Domain      : {cap.domain}")
            lines.append(f"  Description : {cap.description}")
            lines.append(f"  Capabilities: {', '.join(cap.capabilities)}")
            lines.append(f"  Handles     : {', '.join(cap.input_types)}")
            lines.append(f"  Produces    : {', '.join(cap.output_types)}")
            lines.append("")
        lines.append("=== END AGENTS ===")
        return "\n".join(lines)

    # --------------------------------------------------- keyword fallback search

    def find_agents_for_task(self, task: str, top_k: int = 3) -> List[str]:
        """Keyword-based fallback when the LLM planner is unavailable."""
        words = task.lower().split()
        scores: Dict[str, int] = {}
        for name, cap in self._caps.items():
            blob = " ".join(
                [cap.description, cap.domain] + cap.capabilities + cap.input_types
            ).lower()
            score = sum(1 for w in words if w in blob)
            if score:
                scores[name] = score
        return sorted(scores, key=lambda n: scores[n], reverse=True)[:top_k]
