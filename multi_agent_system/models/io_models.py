"""Structured input/output models for the multi-agent system."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    """Standard structured input for any agent in the system."""

    query: str = Field(..., description="The user query or task")
    user_id: str = Field(default_factory=lambda: f"user_{uuid.uuid4().hex[:8]}")
    session_id: Optional[str] = Field(default=None)
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context to pass to agents",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request metadata (source, priority, etc.)",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentTaskResult(BaseModel):
    """Result from a single agent execution."""

    agent_name: str
    task: str
    output: str
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """Standard structured output from the orchestrator."""

    query: str
    response: str = ""
    status: Literal["success", "partial", "failed"] = "success"

    agents_used: List[str] = Field(default_factory=list)
    agent_results: List[AgentTaskResult] = Field(default_factory=list)

    trace_id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    total_duration_ms: float = 0.0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    plan_reasoning: str = ""
    learning_applied: Optional[str] = None

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Query    : {self.query}",
            f"Status   : {self.status.upper()}",
            f"Agents   : {' → '.join(self.agents_used)}",
            f"Duration : {self.total_duration_ms:.0f}ms",
            f"Trace ID : {self.trace_id}",
        ]
        if self.learning_applied:
            lines.append(f"Learning : {self.learning_applied}")
        return "\n".join(lines)
