"""BaseOrchestrator — shared planner + coordinator logic.

Both the RootOrchestrator and the DomainOrchestrator extend this class.
This mirrors the pattern requested: "mimic Orchestrator agent for specific domain".

Flow for every query:
  1. PLAN   — LLM builds an ExecutionPlan (which agents, why, in what order)
  2. ROUTE  — Trace the routing decision with full context/reasoning
  3. EXECUTE — Call each agent, store result in session state via output_key
  4. CONSOLIDATE — summary_agent merges all results
  5. LEARN  — Record outcome in the learning store for future planners

The planner uses:
  - agent_registry.context_for_planner()  → full agent map
  - context_manager.get_learnings_for_prompt() → past successes/failures

All of this is visible in the trace output.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from pydantic import Field

from context.context_manager import ContextManager
from models.agent_message import AgentRequest, AgentResponse
from models.trace_models import AgentDecision, ExecutionPlan, StepType
from reward.reward_engine import RewardEngine, RewardSignal
from security.security_layer import SecurityLayer, SecurityCheckResult
from trace.tracer import AgentTracer

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _extract_text(event: Event) -> str:
    """Pull plain text from an Event, safely."""
    if not event.content or not event.content.parts:
        return ""
    return " ".join(
        p.text for p in event.content.parts if hasattr(p, "text") and p.text
    )


def _make_text_event(author: str, text: str) -> Event:
    return Event(
        author=author,
        content=genai_types.Content(
            parts=[genai_types.Part(text=text)], role="model"
        ),
    )


# --------------------------------------------------------------------------- #
# BaseOrchestrator
# --------------------------------------------------------------------------- #


class BaseOrchestrator(BaseAgent):
    """
    Shared planner + coordinator base class.

    Subclasses must provide:
      - orchestrator_name     : human-readable label shown in traces
      - domain                : the domain this orchestrator handles ("root" or e.g. "tech")
      - specialist_agent_names: list of agent names this orchestrator can route to
      - registry              : AgentRegistry populated with those agents
      - context_manager       : shared ContextManager
      - tracer                : shared AgentTracer
      - genai_client          : google.genai.Client for planner / consolidator calls
    """

    orchestrator_name: str = "BaseOrchestrator"
    domain: str = "general"
    specialist_agent_names: List[str] = Field(default_factory=list)

    # Injected dependencies — must be set by subclasses
    registry: Any = Field(default=None)
    context_manager: Any = Field(default=None)
    tracer: Any = Field(default=None)
    llm_config: Any = Field(default=None)   # LLMConfig from config.llm_config
    security: Any = Field(default=None)     # SecurityLayer (optional, created if not provided)
    reward_engine: Any = Field(default=None)  # RewardEngine (optional, created if not provided)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------ utils

    def _get_security(self) -> SecurityLayer:
        if self.security is None:
            object.__setattr__(self, "security", SecurityLayer())
        return self.security

    def _get_reward(self) -> RewardEngine:
        if self.reward_engine is None:
            object.__setattr__(self, "reward_engine", RewardEngine())
        return self.reward_engine

    def _get_user_query(self, ctx: InvocationContext) -> str:
        if ctx.user_content and ctx.user_content.parts:
            return " ".join(
                p.text
                for p in ctx.user_content.parts
                if hasattr(p, "text") and p.text
            )
        # Fallback: latest user turn in session
        for event in reversed(ctx.session.events or []):
            if event.author == "user":
                return _extract_text(event)
        return ""

    async def _llm_call(self, prompt: str) -> str:
        """Provider-agnostic LLM call — used for planning and consolidation."""
        cfg = self.llm_config
        if cfg.client_type == "openai_sdk":
            # Works for both OpenAI and Groq (Groq uses OpenAI-compatible API)
            resp = await cfg.client.chat.completions.create(
                model=cfg.planner_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        else:
            # google-genai
            response = cfg.client.models.generate_content(
                model=cfg.planner_model,
                contents=prompt,
            )
            return response.text or ""

    # ------------------------------------------------------------------ plan

    async def _build_plan(
        self,
        query: str,
        trace_id: str,
        session_context: Dict[str, Any],
    ) -> ExecutionPlan:
        """Ask the LLM planner to produce a JSON execution plan."""

        agent_map = self.registry.context_for_planner()
        learnings = self.context_manager.get_learnings_for_prompt()
        performance_hints = self._get_reward().performance_for_prompt()
        session_ctx_str = json.dumps(session_context, indent=2) if session_context else "{}"

        prompt = f"""You are an expert AI orchestrator planner for the "{self.domain}" domain.

{agent_map}

{learnings}
{performance_hints}
SESSION CONTEXT (from prior agent runs in this session):
{session_ctx_str}

USER QUERY:
{query}

YOUR TASK:
Build a precise execution plan. Return ONLY valid JSON matching this schema exactly:

{{
  "query_understanding": "What the user is really asking",
  "intent": "Primary intent (e.g. research, code, analysis, translation, mixed)",
  "required_agents": ["agent_name_1", "agent_name_2"],
  "decisions": [
    {{
      "agent_name": "agent_name",
      "task": "Exact task description for this agent",
      "reasoning": "Why this agent is the right choice",
      "confidence": 0.95,
      "expected_output": "What this agent should return",
      "parallel": false,
      "depends_on": []
    }}
  ],
  "reasoning": "Overall routing strategy and why",
  "alternatives_considered": ["agent that was considered but not chosen — reason"],
  "learning_applied": "Which past learning influenced this plan (null if none)",
  "parallel_possible": false
}}

Rules:
- Only include agents listed in AVAILABLE AGENTS above.
- If the query needs multiple agents, list all of them in order.
- If tasks are independent, mark parallel: true.
- reasoning must explain WHY you chose this routing (not just what).
- alternatives_considered should name at least one agent you chose NOT to use.
"""

        t0 = time.time()
        try:
            raw = await self._llm_call(prompt)
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            data = json.loads(raw)
            plan = ExecutionPlan(**data)
        except Exception as exc:
            # Fallback: keyword-based routing
            fallback_agents = self.registry.find_agents_for_task(query, top_k=2)
            plan = ExecutionPlan(
                query_understanding=query,
                intent="unknown",
                required_agents=fallback_agents or self.specialist_agent_names[:1],
                decisions=[
                    AgentDecision(
                        agent_name=a,
                        task=query,
                        reasoning=f"Fallback routing (planner error: {exc})",
                        confidence=0.5,
                    )
                    for a in (fallback_agents or self.specialist_agent_names[:1])
                ],
                reasoning=f"Fallback routing due to planner error: {exc}",
                alternatives_considered=[],
            )

        duration = (time.time() - t0) * 1000
        self.tracer.log_plan(trace_id, plan)
        self.tracer.log(
            trace_id=trace_id,
            orchestrator=self.orchestrator_name,
            step_type=StepType.PLANNING,
            step_name="build_execution_plan",
            agent=self.name,
            input_data=query,
            output_data=f"Plan: {plan.required_agents}",
            reasoning=plan.reasoning,
            duration_ms=duration,
        )
        return plan

    # --------------------------------------------------------------- execute

    def _build_agent_request(
        self,
        agent_name: str,
        task: str,
        trace_id: str,
        session_id: str,
        step: int,
        ctx: InvocationContext,
    ) -> AgentRequest:
        """Build the structured JSON request the orchestrator sends to an agent."""
        # Collect all prior agent results as context for this agent
        prior_context: Dict[str, Any] = {}
        for key, val in ctx.session.state.items():
            if key.endswith("_result") and key != f"{agent_name}_result":
                prior_context[key] = str(val)[:400]  # truncate for prompt

        return AgentRequest(
            trace_id=trace_id,
            session_id=session_id,
            step=step,
            orchestrator=self.orchestrator_name,
            agent_name=agent_name,
            task=task,
            context=prior_context,
        )

    def _parse_agent_response(
        self, raw: str, agent_name: str, request_id: str
    ) -> AgentResponse:
        """Parse a raw string into AgentResponse, with fallback.

        Handles three formats:
          1. Raw JSON string
          2. JSON wrapped in ```json ... ``` markdown fences
          3. Plain text (wrapped in AgentResponse as partial)
        """
        try:
            import re
            clean = raw.strip()
            # Strip any markdown code fences (```json ... ``` or ``` ... ```)
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean)
            if fence_match:
                clean = fence_match.group(1).strip()
            # Find first { ... } JSON block if there's surrounding text
            if not clean.startswith("{"):
                brace_match = re.search(r"\{[\s\S]*\}", clean)
                if brace_match:
                    clean = brace_match.group(0)
            data = json.loads(clean)
            return AgentResponse(**data)
        except Exception:
            # Fallback: wrap plain text in AgentResponse
            return AgentResponse(
                request_id=request_id,
                agent_name=agent_name,
                status="partial",
                output=raw,
                key_points=[],
                confidence=0.7,
                limitations=["Response was not structured JSON — parsed as plain text"],
            )

    async def _execute_agent(
        self,
        agent_name: str,
        task: str,
        ctx: InvocationContext,
        trace_id: str,
        step: int = 1,
    ) -> AgentResponse:
        """
        Full structured execution of one specialist agent:

          1. Build AgentRequest (JSON) with task + prior context
          2. Inject into session state so the agent's instruction template resolves it
          3. Call the agent via ADK runner
          4. Parse the structured AgentResponse JSON
          5. Log both request and response JSON in the trace
        """
        agent = self.registry.get_agent(agent_name)
        if not agent:
            return AgentResponse(
                request_id="n/a",
                agent_name=agent_name,
                status="failed",
                output=f"Agent '{agent_name}' not found in registry",
                confidence=0.0,
            )

        # ── 1. Build request ────────────────────────────────────────────────
        request = self._build_agent_request(
            agent_name=agent_name,
            task=task,
            trace_id=trace_id,
            session_id=ctx.session.id,
            step=step,
            ctx=ctx,
        )

        # ── 1b. Security check on outgoing request ──────────────────────────
        sec = self._get_security()
        req_check = sec.check_request(request)
        if req_check.sanitized_text:
            request = request.model_copy(update={"task": req_check.sanitized_text})
        if req_check.has_blocking:
            self.tracer.log(
                trace_id=trace_id, orchestrator=self.orchestrator_name,
                step_type=StepType.ERROR, step_name=f"security_block::{agent_name}",
                agent=agent_name, input_data=task,
                output_data=req_check.summary(),
                reasoning=f"SecurityLayer blocked request: {req_check.summary()}",
                success=False, error=req_check.summary(),
            )
            return AgentResponse(
                request_id=request.request_id, agent_name=agent_name,
                status="failed",
                output=f"[SECURITY BLOCKED] {req_check.summary()}",
                confidence=0.0,
                limitations=[req_check.summary()],
            )

        request_json = request.model_dump_json(indent=2)

        # ── 2. Inject into session state (template variables) ───────────────
        ctx.session.state[f"{agent_name}_task"] = request.task
        ctx.session.state[f"{agent_name}_request_context"] = request.to_prompt_context()

        # Log the outgoing request
        self.tracer.log_agent_request(trace_id, self.orchestrator_name, agent_name, request)

        # ── 3. Call agent ────────────────────────────────────────────────────
        t0 = time.time()
        raw_result = ""
        error = None
        try:
            async for event in agent.run_async(ctx):
                if event.is_final_response():
                    raw_result = _extract_text(event)
        except Exception as exc:
            error = str(exc)

        # Also try output_key in session state (ADK stores structured output there)
        if not raw_result:
            raw_result = ctx.session.state.get(f"{agent_name}_result", "")

        duration = (time.time() - t0) * 1000

        # ── 4. Parse AgentResponse ───────────────────────────────────────────
        if error:
            response = AgentResponse(
                request_id=request.request_id,
                agent_name=agent_name,
                status="failed",
                output=f"[ERROR: {error}]",
                confidence=0.0,
                limitations=[error],
            )
        else:
            response = self._parse_agent_response(raw_result, agent_name, request.request_id)
            # Backfill request_id if agent omitted it
            if not response.request_id:
                response.request_id = request.request_id

        # ── 4b. Security check on response ──────────────────────────────────
        resp_check = sec.check_response(response)
        if resp_check.sanitized_text:
            response = response.model_copy(update={"output": resp_check.sanitized_text})
        if resp_check.has_blocking:
            response = response.model_copy(update={
                "status": "failed",
                "output": f"[SECURITY BLOCKED RESPONSE] {resp_check.summary()}",
                "confidence": 0.0,
                "limitations": [resp_check.summary()],
            })

        # ── 4c. Reward scoring ───────────────────────────────────────────────
        reward = self._get_reward()
        reward_signal = reward.score_response(
            response=response,
            duration_ms=duration,
            trace_id=trace_id,
            session_id=ctx.session.id,
            retry_count=step - 1,
        )

        # ── 4d. Auto-retry on low reward (non-summary agents only) ───────────
        if agent_name != "summary_agent" and reward.should_retry(reward_signal):
            retry_count = reward.record_retry(request.request_id)
            refined_task = reward.refine_task_prompt(task, reward_signal, response)
            self.tracer.log(
                trace_id=trace_id, orchestrator=self.orchestrator_name,
                step_type=StepType.EXECUTION,
                step_name=f"retry::{agent_name}::{retry_count}",
                agent=agent_name,
                input_data=f"Low reward score: {reward_signal.composite_score:.0%}",
                output_data=f"Retrying with refined prompt (attempt {retry_count})",
                reasoning=reward_signal.reasoning,
                duration_ms=0,
            )
            ctx.session.state[f"{agent_name}_task"] = refined_task
            ctx.session.state[f"{agent_name}_request_context"] = request.to_prompt_context()
            retry_raw = ""
            try:
                async for event in agent.run_async(ctx):
                    if event.is_final_response():
                        retry_raw = _extract_text(event)
            except Exception:
                pass
            if not retry_raw:
                retry_raw = ctx.session.state.get(f"{agent_name}_result", "")
            if retry_raw:
                retry_response = self._parse_agent_response(retry_raw, agent_name, request.request_id)
                retry_signal = reward.score_response(
                    response=retry_response,
                    duration_ms=duration,
                    trace_id=trace_id,
                    session_id=ctx.session.id,
                    retry_count=retry_count,
                )
                if retry_signal.composite_score > reward_signal.composite_score:
                    response = retry_response
                    reward_signal = retry_signal

        response_json = response.model_dump_json(indent=2)

        # ── 5. Log request + response in trace ───────────────────────────────
        self.tracer.log_agent_response(trace_id, self.orchestrator_name, agent_name, response)
        self.tracer.log(
            trace_id=trace_id,
            orchestrator=self.orchestrator_name,
            step_type=StepType.EXECUTION,
            step_name=f"execute::{agent_name}",
            agent=agent_name,
            input_data=request_json,
            output_data=response_json,
            reasoning=(
                f"Step {step} — task dispatched and structured JSON response received. "
                f"Reward: {reward_signal.composite_score:.0%} ({reward_signal.reasoning})"
            ),
            duration_ms=duration,
            success=error is None,
            error=error,
            context_used={
                "request_id": request.request_id,
                "step": step,
                "reward_score": reward_signal.composite_score,
                "security_violations": len(resp_check.violations),
            },
        )

        # Store result for cross-agent context
        self.context_manager.set_session(
            ctx.session.id, f"{agent_name}_result", response.output
        )
        ctx.session.state[f"{agent_name}_result"] = response.output
        # Store reward signal on session state for consolidation
        ctx.session.state[f"{agent_name}_reward"] = reward_signal.composite_score

        return response

    # ------------------------------------------------------------ consolidate

    async def _consolidate(
        self,
        query: str,
        agent_responses: Dict[str, AgentResponse],
        plan: ExecutionPlan,
        ctx: InvocationContext,
        trace_id: str,
        step: int,
    ) -> AgentResponse:
        """Run summary_agent to merge all AgentResponse objects into one final response."""
        if not agent_responses:
            return AgentResponse(
                request_id="n/a", agent_name=self.name,
                status="failed", output="No results to consolidate.", confidence=0.0,
            )

        # Build rich text block from all responses for the summary agent
        results_text = "\n\n".join(
            f"=== {name.upper()} (confidence={r.confidence:.0%}) ===\n"
            f"Output:\n{r.output}\n"
            f"Key Points:\n" + "\n".join(f"  - {p}" for p in r.key_points)
            for name, r in agent_responses.items()
        )

        summary_task = (
            f"Original user query: {query}\n\n"
            f"Results from {len(agent_responses)} agents:\n{results_text}\n\n"
            f"Synthesise these into a single, clear, comprehensive JSON response."
        )

        summary_resp = await self._execute_agent(
            agent_name="summary_agent",
            task=summary_task,
            ctx=ctx,
            trace_id=trace_id,
            step=step,
        )

        self.tracer.log(
            trace_id=trace_id,
            orchestrator=self.orchestrator_name,
            step_type=StepType.CONSOLIDATION,
            step_name="consolidate_results",
            agent=self.name,
            input_data=f"Agents synthesised: {list(agent_responses.keys())}",
            output_data=summary_resp.model_dump_json(indent=2)[:600],
            reasoning="summary_agent merged all structured agent responses",
            duration_ms=0,
        )
        return summary_resp

    # ------------------------------------------------------- main run loop

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        trace_id = f"trace_{uuid.uuid4().hex[:10]}"
        query = self._get_user_query(ctx)
        session_context = self.context_manager.get_full_session_context(ctx.session.id)

        self.tracer.start_session(trace_id, query)

        # ── 0. INPUT SECURITY CHECK ──────────────────────────────────────────
        sec = self._get_security()
        input_check = sec.check_input(query, session_id=ctx.session.id)
        if input_check.sanitized_text:
            query = input_check.sanitized_text
        if input_check.has_blocking:
            blocked_msg = f"[SECURITY] Request blocked: {input_check.summary()}"
            self.tracer.log(
                trace_id=trace_id, orchestrator=self.orchestrator_name,
                step_type=StepType.ERROR, step_name="input_security_block",
                agent=self.name, input_data=query,
                output_data=blocked_msg,
                reasoning=input_check.summary(),
                success=False, error=blocked_msg,
            )
            self.tracer.end_session(trace_id, blocked_msg, success=False)
            yield _make_text_event(self.name, blocked_msg)
            return

        session_start_ms = time.time() * 1000

        # ── 1. PLAN ──────────────────────────────────────────────────────────
        plan = await self._build_plan(query, trace_id, session_context)

        # ── 2. ROUTE & EXECUTE ───────────────────────────────────────────────
        agent_responses: Dict[str, AgentResponse] = {}
        for step_num, decision in enumerate(plan.decisions, start=1):
            agent_name = decision.agent_name
            if agent_name not in self.specialist_agent_names and \
               agent_name not in (self.registry.list_agent_names()):
                continue

            self.tracer.log(
                trace_id=trace_id,
                orchestrator=self.orchestrator_name,
                step_type=StepType.ROUTING,
                step_name=f"route_to::{agent_name}",
                agent=self.name,
                input_data=query,
                output_data=f"Routing to {agent_name} — task: {decision.task[:100]}",
                reasoning=decision.reasoning,
                context_used={"confidence": decision.confidence, "parallel": decision.parallel},
            )

            agent_resp = await self._execute_agent(
                agent_name=agent_name,
                task=decision.task,
                ctx=ctx,
                trace_id=trace_id,
                step=step_num,
            )
            agent_responses[agent_name] = agent_resp

        # ── 3. CONSOLIDATE ───────────────────────────────────────────────────
        final_resp = await self._consolidate(
            query, agent_responses, plan, ctx, trace_id,
            step=len(agent_responses) + 1,
        )
        final_text = final_resp.output

        # ── 3b. Output security check ────────────────────────────────────────
        output_check = sec.check_output(final_text)
        if output_check.sanitized_text:
            final_text = output_check.sanitized_text
        if output_check.has_blocking:
            final_text = f"[SECURITY] Output blocked: {output_check.summary()}"

        # ── 4. REWARD & LEARN ────────────────────────────────────────────────
        reward = self._get_reward()
        # Collect per-agent reward signals stored during execution
        agent_reward_scores: Dict[str, float] = {
            name: float(ctx.session.state.get(f"{name}_reward", 0.5))
            for name in agent_responses.keys()
        }
        total_duration_ms = time.time() * 1000 - session_start_ms
        exec_reward = reward.score_execution(
            trace_id=trace_id,
            session_id=ctx.session.id,
            query_type=plan.intent,
            agent_signals={},   # signals already scored per-agent above
            total_duration_ms=total_duration_ms,
        )
        # Compute session score from stored per-agent scores
        session_score = (
            sum(agent_reward_scores.values()) / len(agent_reward_scores)
            if agent_reward_scores else 0.5
        )

        self.context_manager.record_success(
            query_summary=query[:120],
            query_type=plan.intent,
            agents_used=list(agent_responses.keys()),
            reasoning=plan.reasoning,
            reward_score=session_score,
            session_score=session_score,
            agent_scores=agent_reward_scores,
        )
        self.tracer.log(
            trace_id=trace_id,
            orchestrator=self.orchestrator_name,
            step_type=StepType.LEARNING,
            step_name="record_learning",
            agent=self.name,
            input_data=query,
            output_data="Recorded successful execution with reward scores",
            reasoning=(
                f"Intent={plan.intent}, agents={list(agent_responses.keys())}, "
                f"avg_confidence={sum(r.confidence for r in agent_responses.values()) / max(len(agent_responses),1):.0%}, "
                f"session_reward={session_score:.0%}, "
                f"agent_rewards={agent_reward_scores}"
            ),
        )

        self.tracer.end_session(trace_id, final_text, success=True)

        # Store structured output in session state for parent orchestrator
        ctx.session.state[f"{self.name}_result"] = final_text
        ctx.session.state[f"{self.name}_trace_id"] = trace_id
        ctx.session.state[f"{self.name}_agents_used"] = list(agent_responses.keys())

        yield _make_text_event(self.name, final_text)
