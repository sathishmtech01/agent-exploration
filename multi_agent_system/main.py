"""
Multi-Agent System — Entry Point
=================================
Demonstrates a 3-level agent hierarchy:

  User Query
      │
  RootOrchestrator          ← Level 1: planner + coordinator
      ├── domain_orchestrator    ← Level 2: tech-domain planner + coordinator
      │       ├── research_agent      ← Level 3: specialist
      │       ├── code_agent          ← Level 3: specialist
      │       └── data_agent          ← Level 3: specialist
      ├── language_agent         ← Level 2 (direct): translation / content
      ├── research_agent         ← Level 2 (direct): general research
      └── summary_agent          ← Level 2 (direct): consolidation

Every decision, routing choice, and agent call is traced to stdout in real time.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.root_orchestrator import build_root_orchestrator
from config.llm_config import get_llm_config
from context.context_manager import ContextManager
from models.io_models import AgentInput, AgentOutput, AgentTaskResult
from trace.tracer import AgentTracer

console = Console()

APP_NAME = "multi_agent_system"


# --------------------------------------------------------------------------- #
# System builder
# --------------------------------------------------------------------------- #


def build_system() -> tuple[Runner, ContextManager, AgentTracer, InMemorySessionService]:
    """Initialise and wire up the entire multi-agent system."""
    llm_cfg = get_llm_config()
    context_manager = ContextManager()
    tracer = AgentTracer()
    root = build_root_orchestrator(context_manager, tracer, llm_config=llm_cfg)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root,
        app_name=APP_NAME,
        session_service=session_service,
    )
    return runner, context_manager, tracer, session_service


# --------------------------------------------------------------------------- #
# Core run function
# --------------------------------------------------------------------------- #


async def run_query(
    agent_input: AgentInput,
    runner: Runner,
    session_service: InMemorySessionService,
    tracer: AgentTracer,
) -> AgentOutput:
    """
    Send a structured AgentInput through the system and return a structured AgentOutput.

    Prints the full trace to stdout as it runs.
    """
    console.print(Rule(f"[bold bright_white]  NEW QUERY  [/bold bright_white]", style="bright_white"))
    console.print(Panel(
        f"[bold yellow]Query  :[/bold yellow] {agent_input.query}\n"
        f"[bold yellow]User   :[/bold yellow] {agent_input.user_id}\n"
        f"[bold yellow]Context:[/bold yellow] {agent_input.context or 'none'}",
        title="[bold]Agent Input[/bold]",
        border_style="yellow",
        expand=False,
    ))

    # Create or reuse session
    session_id = agent_input.session_id
    if not session_id:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=agent_input.user_id
        )
        session_id = session.id
    else:
        # Try to get existing session
        existing = await session_service.get_session(
            app_name=APP_NAME, user_id=agent_input.user_id, session_id=session_id
        )
        if not existing:
            session = await session_service.create_session(
                app_name=APP_NAME, user_id=agent_input.user_id
            )
            session_id = session.id

    # Inject any input context into session state via a pre-seeded message
    user_message = agent_input.query
    if agent_input.context:
        context_note = "\n".join(f"{k}: {v}" for k, v in agent_input.context.items())
        user_message = f"{agent_input.query}\n\n[Context provided by user]\n{context_note}"

    t_start = time.time()
    final_text = ""
    agents_used: list[str] = []
    trace_id = f"trace_{uuid.uuid4().hex[:10]}"

    try:
        async for event in runner.run_async(
            user_id=agent_input.user_id,
            session_id=session_id,
            new_message=genai_types.Content(
                parts=[genai_types.Part(text=user_message)], role="user"
            ),
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text = part.text

        total_ms = (time.time() - t_start) * 1000

        # Gather trace info
        for session_key, session_obj in tracer.sessions.items():
            if session_obj.query in [agent_input.query, user_message]:
                trace_id = session_key
                agents_used = session_obj.agents_used
                break

        # Build structured output
        output = AgentOutput(
            query=agent_input.query,
            response=final_text,
            status="success" if final_text else "failed",
            agents_used=agents_used,
            trace_id=trace_id,
            total_duration_ms=total_ms,
        )

    except Exception as exc:
        total_ms = (time.time() - t_start) * 1000
        console.print(f"\n[red]System error: {exc}[/red]")
        output = AgentOutput(
            query=agent_input.query,
            response=f"System error: {exc}",
            status="failed",
            agents_used=[],
            trace_id=trace_id,
            total_duration_ms=total_ms,
        )

    console.print(Rule(style="bright_white"))
    return output


# --------------------------------------------------------------------------- #
# Demo scenarios
# --------------------------------------------------------------------------- #

DEMO_QUERIES = [
    AgentInput(
        query=(
            "Explain how transformer neural networks work "
            "and write a minimal Python example that demonstrates "
            "scaled dot-product attention."
        ),
        user_id="demo_user",
        context={"expertise_level": "intermediate", "preferred_language": "Python"},
    ),
    AgentInput(
        query=(
            "Analyse the trend in global EV adoption over the last 5 years "
            "and translate the key findings into French."
        ),
        user_id="demo_user",
        context={"output_language": "French"},
    ),
    AgentInput(
        query=(
            "What are the best practices for building a production-ready "
            "REST API with FastAPI, including authentication and rate limiting?"
        ),
        user_id="demo_user",
    ),
]


async def main() -> None:
    llm_cfg = get_llm_config()
    console.print(
        Panel(
            "[bold bright_cyan]Google ADK — Custom Multi-Agent System[/bold bright_cyan]\n\n"
            f"[yellow]LLM Provider :[/yellow] {llm_cfg.provider.upper()}\n"
            f"[yellow]Agent model  :[/yellow] {llm_cfg.agent_model}\n"
            f"[yellow]Planner model:[/yellow] {llm_cfg.planner_model}\n\n"
            "Architecture:\n"
            "  RootOrchestrator  [planner + coordinator]\n"
            "  └─ DomainOrchestrator [tech: planner + coordinator]\n"
            "       ├─ research_agent\n"
            "       ├─ code_agent\n"
            "       └─ data_agent\n"
            "  ├─ language_agent\n"
            "  ├─ research_agent  (direct)\n"
            "  └─ summary_agent\n\n"
            "Every decision is traced with full reasoning.",
            border_style="bright_cyan",
        )
    )

    runner, context_manager, tracer, session_service = build_system()

    # Run all demo queries sequentially (same user session to show cross-query learning)
    for i, query_input in enumerate(DEMO_QUERIES, 1):
        console.print(f"\n[bold bright_magenta]══ Demo Query {i} / {len(DEMO_QUERIES)} ══[/bold bright_magenta]\n")
        output = await run_query(query_input, runner, session_service, tracer)

        console.print(
            Panel(
                output.summary(),
                title="[bold]Structured Output[/bold]",
                border_style="bright_green",
                expand=False,
            )
        )
        await asyncio.sleep(1)  # brief pause between queries

    # ── Interactive mode ────────────────────────────────────────────────────
    console.print(
        "\n[bold bright_yellow]═══ Interactive Mode ═══[/bold bright_yellow]\n"
        "Type your query (or 'quit' to exit):\n"
    )
    user_id = "interactive_user"
    session_id = None

    while True:
        try:
            query = input("[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        inp = AgentInput(query=query, user_id=user_id, session_id=session_id)
        output = await run_query(inp, runner, session_service, tracer)
        session_id = output.trace_id  # persist session for follow-up questions

        console.print(
            Panel(
                output.response or "[no response]",
                title="[bold green]Answer[/bold green]",
                border_style="green",
                expand=False,
            )
        )

    console.print("\n[dim]Session ended. Learnings saved to learning_store.json[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
