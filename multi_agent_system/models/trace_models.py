"""Trace, plan, and decision models for full observability."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepType(str, Enum):
    PLANNING = "planning"
    ROUTING = "routing"
    EXECUTION = "execution"
    CONSOLIDATION = "consolidation"
    LEARNING = "learning"
    ERROR = "error"


class AgentDecision(BaseModel):
    """A single routing decision made by an orchestrator."""

    agent_name: str
    task: str
    reasoning: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    expected_output: str = ""
    parallel: bool = False
    depends_on: List[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """Full plan produced by the planner before execution."""

    query_understanding: str
    intent: str
    required_agents: List[str]
    decisions: List[AgentDecision]
    reasoning: str
    alternatives_considered: List[str] = Field(default_factory=list)
    learning_applied: Optional[str] = None
    parallel_possible: bool = False


class PlanStep(BaseModel):
    step_number: int
    agent_name: str
    task: str
    expected_output: str = ""
    depends_on: List[int] = Field(default_factory=list)
    parallel: bool = False


class TraceEntry(BaseModel):
    """A single trace log entry from any agent step."""

    trace_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    orchestrator: str
    step_type: StepType
    step_name: str
    agent: str
    input_data: str
    output_data: str
    reasoning: str
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    context_used: Dict[str, Any] = Field(default_factory=dict)


class TraceSession(BaseModel):
    """Complete trace session for one query invocation."""

    trace_id: str
    query: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    entries: List[TraceEntry] = Field(default_factory=list)
    final_response: str = ""
    agents_used: List[str] = Field(default_factory=list)
    plan: Optional[ExecutionPlan] = None
    success: bool = True
    total_duration_ms: float = 0.0

    def add_entry(self, entry: TraceEntry) -> None:
        self.entries.append(entry)
        if entry.agent not in self.agents_used:
            self.agents_used.append(entry.agent)
