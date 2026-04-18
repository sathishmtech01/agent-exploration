"""
Quick test runner — run this BEFORE main.py to verify setup.

Tests:
  1. Config / API key check
  2. Single LLM call (planner model)
  3. Single specialist agent (research_agent)
  4. Full orchestration pipeline (one query, full trace)

Usage:
  python test_system.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel

console = Console()


# ─────────────────────────────────────────────────────────────────── Test 1
def test_config() -> bool:
    console.print("\n[bold yellow]TEST 1 — Config & API Key[/bold yellow]")
    try:
        from config.llm_config import get_llm_config
        cfg = get_llm_config()
        console.print(f"  [green]✓[/green] Provider    : {cfg.provider}")
        console.print(f"  [green]✓[/green] Agent model : {cfg.agent_model}")
        console.print(f"  [green]✓[/green] Client type : {cfg.client_type}")
        return True
    except EnvironmentError as e:
        console.print(f"  [red]✗ {e}[/red]")
        console.print(
            "\n  [yellow]Fix:[/yellow] Add to .env:\n"
            "    LLM_PROVIDER=openai\n"
            "    OPENAI_API_KEY=sk-...\n"
            "  or\n"
            "    LLM_PROVIDER=groq\n"
            "    GROQ_API_KEY=gsk_...\n"
            "  or\n"
            "    LLM_PROVIDER=gemini\n"
            "    GOOGLE_API_KEY=AIza..."
        )
        return False


# ─────────────────────────────────────────────────────────────────── Test 2
async def test_llm_call() -> bool:
    console.print("\n[bold yellow]TEST 2 — Direct LLM Call (planner)[/bold yellow]")
    try:
        from config.llm_config import get_llm_config
        cfg = get_llm_config()

        if cfg.client_type == "openai_sdk":
            resp = await cfg.client.chat.completions.create(
                model=cfg.planner_model,
                messages=[{"role": "user", "content": "Reply with just: HELLO"}],
                max_tokens=10,
            )
            text = resp.choices[0].message.content
        else:
            resp = cfg.client.models.generate_content(
                model=cfg.planner_model,
                contents="Reply with just: HELLO",
            )
            text = resp.text

        console.print(f"  [green]✓[/green] LLM responded: {text.strip()[:60]}")
        return True
    except Exception as e:
        console.print(f"  [red]✗ LLM call failed: {e}[/red]")
        return False


# ─────────────────────────────────────────────────────────────────── Test 3
async def test_specialist_agent() -> bool:
    console.print("\n[bold yellow]TEST 3 — Specialist Agent (research_agent)[/bold yellow]")
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types as genai_types

        from agents.specialist_agents import build_research_agent
        from config.llm_config import get_llm_config

        cfg = get_llm_config()
        agent = build_research_agent(model=cfg.agent_model)
        session_service = InMemorySessionService()
        runner = Runner(agent=agent, app_name="test", session_service=session_service)
        session = await session_service.create_session(app_name="test", user_id="test_user")

        # Set task via session state before running
        session.state["research_agent_task"] = "In one sentence, what is Python?"

        response = ""
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=genai_types.Content(
                parts=[genai_types.Part(text="What is Python?")], role="user"
            ),
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response = part.text

        if response:
            console.print(f"  [green]✓[/green] Agent responded ({len(response)} chars)")
            console.print(f"  [dim]{response[:120]}...[/dim]")
            return True
        else:
            console.print("  [red]✗ No response from agent[/red]")
            return False
    except Exception as e:
        console.print(f"  [red]✗ Agent test failed: {e}[/red]")
        return False


# ─────────────────────────────────────────────────────────────────── Test 4
async def test_full_pipeline() -> bool:
    console.print("\n[bold yellow]TEST 4 — Full Orchestration Pipeline[/bold yellow]")
    try:
        from main import build_system, run_query
        from models.io_models import AgentInput

        runner, context_manager, tracer, session_service = build_system()

        result = await run_query(
            AgentInput(
                query="What is a binary search tree? Give a one-paragraph explanation.",
                user_id="test_user",
            ),
            runner,
            session_service,
            tracer,
        )

        if result.status == "success" and result.response:
            console.print(f"  [green]✓[/green] Status  : {result.status}")
            console.print(f"  [green]✓[/green] Agents  : {result.agents_used}")
            console.print(f"  [green]✓[/green] Duration: {result.total_duration_ms:.0f}ms")
            console.print(f"  [dim]{result.response[:200]}...[/dim]")
            return True
        else:
            console.print(f"  [red]✗ Pipeline returned status={result.status}[/red]")
            return False
    except Exception as e:
        console.print(f"  [red]✗ Pipeline test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────── Runner
async def main():
    console.print(Panel(
        "[bold]Multi-Agent System — Test Suite[/bold]\n"
        "Runs 4 progressive checks: config → LLM → agent → full pipeline",
        border_style="bright_cyan",
        expand=False,
    ))

    results = {}

    results["config"] = test_config()
    if not results["config"]:
        console.print("\n[red]Cannot continue — fix config first.[/red]")
        return

    results["llm"] = await test_llm_call()
    if not results["llm"]:
        console.print("\n[red]Cannot continue — LLM call failed.[/red]")
        return

    results["agent"] = await test_specialist_agent()
    results["pipeline"] = await test_full_pipeline()

    # Summary
    console.print("\n")
    passed = sum(results.values())
    total = len(results)
    color = "green" if passed == total else "yellow"
    console.print(Panel(
        "\n".join(
            f"  {'[green]PASS[/green]' if v else '[red]FAIL[/red]'}  {k}"
            for k, v in results.items()
        ) + f"\n\n  [{color}]{passed}/{total} tests passed[/{color}]",
        title="[bold]Test Results[/bold]",
        border_style=color,
        expand=False,
    ))

    if passed == total:
        console.print("\n[green]✓ System ready — run:[/green]  python main.py\n")


if __name__ == "__main__":
    asyncio.run(main())
