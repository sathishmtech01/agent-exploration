"""RewardEngine — score every agent execution and use signals to improve routing.

How it works
------------
1. After each AgentResponse is received, `score_response()` computes a
   composite reward (0.0–1.0) from several heuristics.
2. After the full orchestration cycle, `score_execution()` computes a
   session-level reward that accounts for all agent scores + latency.
3. Both scores are stored as `AgentRewardRecord` objects in memory and
   optionally persisted to `reward_store.json`.
4. The `get_agent_performance()` method returns per-agent statistics that
   the planner can use to prefer higher-performing agents.
5. `should_retry()` decides whether a failed/low-confidence response
   warrants an automatic retry with a refined prompt.
6. `refine_task_prompt()` rewrites the task string with context from prior
   failures to guide the next attempt.

Scoring heuristics (all normalised to 0.0–1.0)
-----------------------------------------------
- confidence_score   : raw confidence from AgentResponse (direct signal)
- status_score       : success=1.0, partial=0.6, failed=0.0
- output_quality     : length, key_points count, structured_data richness
- latency_score      : penalises very slow responses (> 10 s)
- limitations_penalty: each limitation reduces score slightly
- retry_penalty      : each retry attempt lowers the score

Final reward = weighted combination of the above.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from models.agent_message import AgentResponse


# ─────────────────────────────────────── constants

REWARD_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "reward_store.json")

# Weights for composite score
_W_CONFIDENCE = 0.30
_W_STATUS = 0.25
_W_OUTPUT_QUALITY = 0.25
_W_LATENCY = 0.10
_W_LIMITATIONS = 0.10

# Thresholds
_RETRY_THRESHOLD = 0.45        # score below this → consider retry
_MAX_RETRIES = 2               # cap on automatic retries per agent
_SLOW_RESPONSE_SEC = 10.0      # latency above this starts penalising
_MIN_OUTPUT_CHARS = 50         # below this → quality drops
_GOOD_OUTPUT_CHARS = 500       # above this → quality plateaus at 1.0


# ─────────────────────────────────────── data classes

@dataclass
class RewardSignal:
    """Decomposed reward breakdown for one agent execution."""
    agent_name: str
    request_id: str
    composite_score: float          # 0.0–1.0 overall
    confidence_score: float
    status_score: float
    output_quality_score: float
    latency_score: float
    limitations_score: float
    retry_count: int
    duration_ms: float
    reasoning: str                  # human-readable explanation
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AgentRewardRecord:
    """Persistent record stored per execution for trend analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    session_id: str = ""
    agent_name: str = ""
    request_id: str = ""
    composite_score: float = 0.0
    status: str = "success"
    retry_count: int = 0
    duration_ms: float = 0.0
    query_type: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ExecutionReward:
    """Session-level reward after full orchestration cycle."""
    trace_id: str
    session_id: str
    query_type: str
    agent_scores: Dict[str, float]   # agent_name → composite_score
    session_score: float             # weighted average
    total_duration_ms: float
    agents_used: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ─────────────────────────────────────── engine

class RewardEngine:
    """
    Central reward computation and performance tracking engine.

    Wire into BaseOrchestrator:
      - After _execute_agent()  → reward_engine.score_response(response, duration_ms)
      - After _consolidate()    → reward_engine.score_execution(trace_id, ...)
      - In _build_plan()        → reward_engine.get_agent_performance() for routing hints
    """

    def __init__(self, persist: bool = True, store_path: str = REWARD_STORE_PATH):
        self.persist = persist
        self.store_path = store_path

        # In-memory indexes
        self._records: List[AgentRewardRecord] = []
        # agent_name → list of composite scores
        self._agent_scores: Dict[str, List[float]] = defaultdict(list)
        # agent_name → retry counts
        self._retry_counts: Dict[str, int] = defaultdict(int)
        # request_id → retry count (for current session)
        self._active_retries: Dict[str, int] = defaultdict(int)

        self._load()

    # ─────────────────────────────── public API

    def score_response(
        self,
        response: AgentResponse,
        duration_ms: float,
        trace_id: str = "",
        session_id: str = "",
        query_type: str = "",
        retry_count: int = 0,
    ) -> RewardSignal:
        """Compute composite reward for one AgentResponse."""

        confidence_score = float(response.confidence)

        status_map = {"success": 1.0, "partial": 0.6, "failed": 0.0}
        status_score = status_map.get(response.status, 0.5)

        output_quality_score = self._score_output_quality(response)

        latency_score = self._score_latency(duration_ms)

        # Each limitation knocks off 0.05, floored at 0.0
        limitations_score = max(0.0, 1.0 - len(response.limitations) * 0.05)

        # Retry penalty: each retry lowers composite by 0.05
        retry_penalty = retry_count * 0.05

        composite = (
            _W_CONFIDENCE * confidence_score
            + _W_STATUS * status_score
            + _W_OUTPUT_QUALITY * output_quality_score
            + _W_LATENCY * latency_score
            + _W_LIMITATIONS * limitations_score
            - retry_penalty
        )
        composite = max(0.0, min(1.0, composite))

        reasoning = (
            f"conf={confidence_score:.2f} status={status_score:.2f} "
            f"quality={output_quality_score:.2f} latency={latency_score:.2f} "
            f"limits={limitations_score:.2f} retry_pen=-{retry_penalty:.2f} "
            f"→ composite={composite:.2f}"
        )

        signal = RewardSignal(
            agent_name=response.agent_name,
            request_id=response.request_id,
            composite_score=composite,
            confidence_score=confidence_score,
            status_score=status_score,
            output_quality_score=output_quality_score,
            latency_score=latency_score,
            limitations_score=limitations_score,
            retry_count=retry_count,
            duration_ms=duration_ms,
            reasoning=reasoning,
        )

        # Persist the record
        record = AgentRewardRecord(
            trace_id=trace_id,
            session_id=session_id,
            agent_name=response.agent_name,
            request_id=response.request_id,
            composite_score=composite,
            status=response.status,
            retry_count=retry_count,
            duration_ms=duration_ms,
            query_type=query_type,
        )
        self._records.append(record)
        self._agent_scores[response.agent_name].append(composite)
        if self.persist:
            self._save()

        return signal

    def score_execution(
        self,
        trace_id: str,
        session_id: str,
        query_type: str,
        agent_signals: Dict[str, RewardSignal],
        total_duration_ms: float,
    ) -> ExecutionReward:
        """Compute session-level reward after full orchestration cycle."""
        if not agent_signals:
            return ExecutionReward(
                trace_id=trace_id,
                session_id=session_id,
                query_type=query_type,
                agent_scores={},
                session_score=0.0,
                total_duration_ms=total_duration_ms,
                agents_used=[],
            )

        agent_scores = {name: sig.composite_score for name, sig in agent_signals.items()}
        session_score = sum(agent_scores.values()) / len(agent_scores)

        return ExecutionReward(
            trace_id=trace_id,
            session_id=session_id,
            query_type=query_type,
            agent_scores=agent_scores,
            session_score=session_score,
            total_duration_ms=total_duration_ms,
            agents_used=list(agent_signals.keys()),
        )

    def should_retry(self, signal: RewardSignal) -> bool:
        """Return True if the agent should be retried with a refined prompt."""
        current_retries = self._active_retries[signal.request_id]
        if current_retries >= _MAX_RETRIES:
            return False
        return signal.composite_score < _RETRY_THRESHOLD

    def record_retry(self, request_id: str) -> int:
        """Increment and return the retry count for a request."""
        self._active_retries[request_id] += 1
        return self._active_retries[request_id]

    def refine_task_prompt(
        self,
        original_task: str,
        signal: RewardSignal,
        response: AgentResponse,
    ) -> str:
        """Rewrite the task with context from the failed attempt to guide retry."""
        limitations_text = ""
        if response.limitations:
            limitations_text = (
                "\n\nPrevious attempt limitations (please address these):\n"
                + "\n".join(f"  - {l}" for l in response.limitations)
            )

        output_preview = response.output[:200] if response.output else "(empty)"

        return (
            f"{original_task}"
            f"{limitations_text}"
            f"\n\nPrevious attempt score: {signal.composite_score:.0%} — "
            f"status={response.status}, confidence={response.confidence:.0%}. "
            f"Previous output (partial): {output_preview}..."
            f"\n\nPlease provide a more complete, higher-quality response."
        )

    def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Return per-agent stats for the planner to use in routing decisions.

        Returns:
            {
                "research_agent": {
                    "avg_score": 0.87,
                    "executions": 12,
                    "recent_score": 0.91,
                    "reliability": "high"
                }, ...
            }
        """
        result: Dict[str, Dict[str, Any]] = {}
        for agent_name, scores in self._agent_scores.items():
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            recent = sum(scores[-5:]) / min(len(scores), 5)
            result[agent_name] = {
                "avg_score": round(avg, 3),
                "executions": len(scores),
                "recent_score": round(recent, 3),
                "reliability": self._reliability_label(avg),
            }
        return result

    def performance_for_prompt(self) -> str:
        """Format agent performance as text for injection into planner prompts."""
        perf = self.get_agent_performance()
        if not perf:
            return ""
        lines = ["\n=== AGENT PERFORMANCE (use to prefer high-reliability agents) ==="]
        for agent, stats in sorted(perf.items(), key=lambda x: -x[1]["avg_score"]):
            lines.append(
                f"  {agent}: avg={stats['avg_score']:.0%}  "
                f"recent={stats['recent_score']:.0%}  "
                f"runs={stats['executions']}  "
                f"reliability={stats['reliability']}"
            )
        lines.append("=== END PERFORMANCE ===\n")
        return "\n".join(lines)

    # ─────────────────────────────── private helpers

    def _score_output_quality(self, response: AgentResponse) -> float:
        score = 0.0

        # Length component (50% weight)
        length = len(response.output)
        if length >= _GOOD_OUTPUT_CHARS:
            length_score = 1.0
        elif length >= _MIN_OUTPUT_CHARS:
            length_score = (length - _MIN_OUTPUT_CHARS) / (_GOOD_OUTPUT_CHARS - _MIN_OUTPUT_CHARS)
        else:
            length_score = 0.1
        score += 0.5 * length_score

        # Key points component (30% weight) — up to 5 points = 1.0
        kp_score = min(1.0, len(response.key_points) / 5.0) if response.key_points else 0.0
        score += 0.3 * kp_score

        # Structured data richness (20% weight)
        sd_score = min(1.0, len(response.structured_data) / 3.0) if response.structured_data else 0.0
        score += 0.2 * sd_score

        return round(score, 3)

    def _score_latency(self, duration_ms: float) -> float:
        duration_sec = duration_ms / 1000.0
        if duration_sec <= 2.0:
            return 1.0
        if duration_sec >= _SLOW_RESPONSE_SEC:
            return 0.2
        # Linear decay between 2s and 10s
        return round(1.0 - 0.8 * (duration_sec - 2.0) / (_SLOW_RESPONSE_SEC - 2.0), 3)

    @staticmethod
    def _reliability_label(avg: float) -> str:
        if avg >= 0.80:
            return "high"
        if avg >= 0.60:
            return "medium"
        return "low"

    def _load(self) -> None:
        if not self.persist:
            return
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path) as f:
                    data = json.load(f)
                for item in data:
                    rec = AgentRewardRecord(**item)
                    self._records.append(rec)
                    self._agent_scores[rec.agent_name].append(rec.composite_score)
            except Exception:
                pass

    def _save(self) -> None:
        try:
            records_data = []
            for r in self._records:
                records_data.append({
                    "id": r.id,
                    "trace_id": r.trace_id,
                    "session_id": r.session_id,
                    "agent_name": r.agent_name,
                    "request_id": r.request_id,
                    "composite_score": r.composite_score,
                    "status": r.status,
                    "retry_count": r.retry_count,
                    "duration_ms": r.duration_ms,
                    "query_type": r.query_type,
                    "timestamp": r.timestamp,
                })
            with open(self.store_path, "w") as f:
                json.dump(records_data, f, indent=2)
        except Exception:
            pass
