"""Structured JSON contract between the orchestrator and every specialist agent.

Every agent call goes through exactly two objects:

  AgentRequest  — orchestrator → agent   (what to do + full context)
  AgentResponse — agent → orchestrator   (structured result)

These are the canonical I/O format for all agent boundaries in the system.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────── Request


class AgentRequest(BaseModel):
    """Structured input the orchestrator sends to a specialist agent."""

    request_id: str = Field(
        default_factory=lambda: f"req_{uuid.uuid4().hex[:10]}"
    )
    trace_id: str
    session_id: str
    step: int = 1

    orchestrator: str          # which orchestrator is calling
    agent_name: str            # target agent name
    task: str                  # the exact task for this agent

    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Prior agent results and user-provided context available to this agent",
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional constraints e.g. language, max_length, output_format",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Routing metadata: confidence, parallel flag, depends_on etc.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_prompt_context(self) -> str:
        """Formats the request as a context block injected into the agent instruction."""
        lines = [
            "=== ORCHESTRATOR REQUEST ===",
            f"Request ID  : {self.request_id}",
            f"Trace ID    : {self.trace_id}",
            f"Step        : {self.step}",
            f"From        : {self.orchestrator}",
            f"Task        : {self.task}",
        ]
        if self.constraints:
            lines.append(f"Constraints : {self.constraints}")
        if self.context:
            lines.append("\nContext from prior agents:")
            for k, v in self.context.items():
                preview = str(v)[:300] + "..." if len(str(v)) > 300 else str(v)
                lines.append(f"  [{k}]: {preview}")
        lines.append("=== END REQUEST ===")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────── Response


class AgentResponse(BaseModel):
    """Structured JSON output every specialist agent must return.

    Used as output_schema on LlmAgent — ADK enforces this format automatically.
    """

    request_id: str = Field(
        default="",
        description="Echo the request_id from the incoming request",
    )
    agent_name: str = Field(
        description="Name of the agent producing this response",
    )
    status: Literal["success", "partial", "failed"] = "success"

    # ── Core output ───────────────────────────────────────────────────────
    output: str = Field(
        description="Main response content — the full answer, code, analysis, or translation",
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="3-5 bullet-point summary of the most important findings",
    )
    structured_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any structured artefacts: code snippets, tables, metrics",
    )

    # ── Quality signals ───────────────────────────────────────────────────
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence in the response quality (0.0–1.0)",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known gaps, caveats, or things not covered",
    )

    # ── Routing hints (optional) ──────────────────────────────────────────
    suggested_next_agent: Optional[str] = Field(
        default=None,
        description="Optional: name of agent that should process this output next",
    )

    # ── Meta ──────────────────────────────────────────────────────────────
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def short_summary(self) -> str:
        return (
            f"[{self.status.upper()}] {self.agent_name} "
            f"(confidence={self.confidence:.0%}) — "
            f"{self.output[:120]}{'...' if len(self.output) > 120 else ''}"
        )
