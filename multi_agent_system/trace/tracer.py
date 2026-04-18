"""Complete trace system — every decision, routing, and execution step is logged."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from models.agent_message import AgentRequest, AgentResponse
from models.trace_models import (
    ExecutionPlan,
    StepType,
    TraceEntry,
    TraceSession,
)

console = Console()

_STEP_ICONS = {
    StepType.PLANNING: "🧠",
    StepType.ROUTING: "🔀",
    StepType.EXECUTION: "⚡",
    StepType.CONSOLIDATION: "🔗",
    StepType.LEARNING: "📚",
    StepType.ERROR: "❌",
}

_STEP_COLORS = {
    StepType.PLANNING: "bright_blue",
    StepType.ROUTING: "bright_yellow",
    StepType.EXECUTION: "bright_cyan",
    StepType.CONSOLIDATION: "bright_green",
    StepType.LEARNING: "bright_magenta",
    StepType.ERROR: "bright_red",
}


class AgentTracer:
    """Full observability — records and renders every step of multi-agent execution."""

    def __init__(self):
        self.sessions: Dict[str, TraceSession] = {}
        self._timers: Dict[str, float] = {}

    # ------------------------------------------------------------------ session

    def start_session(self, trace_id: str, query: str) -> TraceSession:
        session = TraceSession(trace_id=trace_id, query=query)
        self.sessions[trace_id] = session
        console.print(
            Panel(
                f"[bold bright_blue]▶  TRACE SESSION STARTED[/bold bright_blue]\n"
                f"[yellow]Trace ID :[/yellow] {trace_id}\n"
                f"[yellow]Query    :[/yellow] {query}",
                border_style="bright_blue",
                expand=False,
            )
        )
        return session

    def end_session(
        self, trace_id: str, final_response: str, success: bool = True
    ) -> Optional[TraceSession]:
        session = self.sessions.get(trace_id)
        if not session:
            return None
        session.end_time = datetime.utcnow()
        session.final_response = final_response
        session.success = success
        session.total_duration_ms = (
            session.end_time - session.start_time
        ).total_seconds() * 1000

        self._print_summary(session)
        return session

    # ---------------------------------------------------------------- step timer

    def start_step(self, trace_id: str, step_name: str) -> str:
        key = f"{trace_id}::{step_name}::{time.time()}"
        self._timers[key] = time.time()
        return key

    def end_step(self, timer_key: str) -> float:
        start = self._timers.pop(timer_key, None)
        return (time.time() - start) * 1000 if start else 0.0

    # -------------------------------------------------------------------- log

    def log(
        self,
        trace_id: str,
        orchestrator: str,
        step_type: StepType,
        step_name: str,
        agent: str,
        input_data: str,
        output_data: str,
        reasoning: str,
        duration_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        context_used: Optional[dict] = None,
    ) -> Optional[TraceEntry]:
        session = self.sessions.get(trace_id)
        if not session:
            return None

        entry = TraceEntry(
            trace_id=trace_id,
            orchestrator=orchestrator,
            step_type=step_type,
            step_name=step_name,
            agent=agent,
            input_data=input_data[:800],
            output_data=output_data[:800],
            reasoning=reasoning,
            duration_ms=duration_ms,
            success=success,
            error=error,
            context_used=context_used or {},
        )
        session.add_entry(entry)

        icon = _STEP_ICONS.get(step_type, "→")
        color = _STEP_COLORS.get(step_type, "white")
        status_tag = "" if success else " [red][FAILED][/red]"

        console.print(
            f"\n[{color}]{icon}  [{step_type.value.upper()}][/{color}]"
            f"  {step_name}{status_tag}"
            f"  │ agent=[bold]{agent}[/bold]"
            f"  │ [dim]{duration_ms:.0f}ms[/dim]"
        )
        if reasoning:
            console.print(f"   [dim]💭 {reasoning[:240]}[/dim]")
        if error:
            console.print(f"   [red]✖  {error}[/red]")

        return entry

    # -------------------------------------------------------------------- plan

    def log_plan(self, trace_id: str, plan: ExecutionPlan) -> None:
        session = self.sessions.get(trace_id)
        if session:
            session.plan = plan
        self._print_plan(plan)

    def _print_plan(self, plan: ExecutionPlan) -> None:
        tree = Tree("\n[bold yellow]📋  EXECUTION PLAN[/bold yellow]")
        tree.add(f"[white]Understanding :[/white] {plan.query_understanding}")
        tree.add(f"[white]Intent        :[/white] {plan.intent}")
        tree.add(f"[white]Agents needed :[/white] {', '.join(plan.required_agents)}")
        tree.add(f"[white]Reasoning     :[/white] {plan.reasoning[:300]}")

        steps_node = tree.add("[white]Steps :[/white]")
        for d in plan.decisions:
            par = " [yellow](⟳ parallel)[/yellow]" if d.parallel else ""
            step = steps_node.add(
                f"[cyan]{d.agent_name}[/cyan]{par} — {d.task}"
            )
            step.add(f"[dim]Why: {d.reasoning}[/dim]")
            if d.depends_on:
                step.add(f"[dim]Depends on: {', '.join(d.depends_on)}[/dim]")

        if plan.alternatives_considered:
            tree.add(
                f"[white]Alternatives  :[/white] {', '.join(plan.alternatives_considered)}"
            )
        if plan.learning_applied:
            tree.add(f"[green]📚 Learning applied :[/green] {plan.learning_applied}")

        console.print(tree)

    # --------------------------------------------------------------- summary

    def _print_summary(self, session: TraceSession) -> None:
        table = Table(
            title="\n[bold]TRACE SUMMARY[/bold]",
            border_style="bright_blue",
            show_lines=True,
        )
        table.add_column("Field", style="yellow", no_wrap=True)
        table.add_column("Value", style="white")

        status_val = (
            "[bright_green]SUCCESS[/bright_green]"
            if session.success
            else "[bright_red]FAILED[/bright_red]"
        )
        table.add_row("Trace ID", session.trace_id)
        table.add_row("Status", status_val)
        table.add_row("Total Duration", f"{session.total_duration_ms:.0f} ms")
        table.add_row("Agents Used", " → ".join(session.agents_used))
        table.add_row("Steps Logged", str(len(session.entries)))
        console.print(table)

        if session.final_response:
            console.print(
                Panel(
                    session.final_response[:1000],
                    title="[bold green]Final Response[/bold green]",
                    border_style="green",
                    expand=False,
                )
            )

    def get_session(self, trace_id: str) -> Optional[TraceSession]:
        return self.sessions.get(trace_id)

    # ------------------------------------------------- JSON I/O pretty print

    def log_agent_request(
        self,
        trace_id: str,
        orchestrator: str,
        agent_name: str,
        request: AgentRequest,
    ) -> None:
        """Print the structured JSON request the orchestrator is sending to an agent."""
        req_dict = request.model_dump()
        # Show context keys without full values to keep output readable
        if req_dict.get("context"):
            req_dict["context"] = {
                k: f"{str(v)[:80]}..." if len(str(v)) > 80 else v
                for k, v in req_dict["context"].items()
            }
        json_str = json.dumps(req_dict, indent=2, default=str)

        console.print(
            f"\n[bright_yellow]📤  REQUEST  [/bright_yellow]"
            f"[bold]{orchestrator}[/bold] → [bold cyan]{agent_name}[/bold cyan]"
            f"  [dim](step {request.step} · {request.request_id})[/dim]"
        )
        console.print(
            Syntax(json_str, "json", theme="monokai", line_numbers=False)
        )

    def log_agent_response(
        self,
        trace_id: str,
        orchestrator: str,
        agent_name: str,
        response: AgentResponse,
    ) -> None:
        """Print the structured JSON response received from an agent."""
        resp_dict = response.model_dump()
        # Truncate long output for readability
        if resp_dict.get("output") and len(resp_dict["output"]) > 300:
            resp_dict["output"] = resp_dict["output"][:300] + "... [truncated]"

        json_str = json.dumps(resp_dict, indent=2, default=str)

        status_color = {
            "success": "bright_green",
            "partial": "yellow",
            "failed": "bright_red",
        }.get(response.status, "white")

        console.print(
            f"\n[{status_color}]📥  RESPONSE [/{status_color}]"
            f"[bold cyan]{agent_name}[/bold cyan] → [bold]{orchestrator}[/bold]"
            f"  [dim](confidence={response.confidence:.0%} · {response.request_id})[/dim]"
        )
        console.print(
            Syntax(json_str, "json", theme="monokai", line_numbers=False)
        )
