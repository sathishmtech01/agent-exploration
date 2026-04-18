"""Context management and learning store for the multi-agent system.

The ContextManager is the memory of the system:
- Propagates context across agent calls
- Stores learnings from past executions (success and failure)
- Informs the planner so it avoids repeating mistakes
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

LEARNING_STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "learning_store.json"
)


class LearningEntry(BaseModel):
    """One recorded learning from a past execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    query_summary: str
    query_type: str
    agents_used: List[str]
    plan_reasoning: str
    success: bool
    feedback: Optional[str] = None
    correction: Optional[str] = None  # what the orchestrator should do differently


class LearningStore:
    """Persistent JSON-backed store for agent execution learnings."""

    def __init__(self, store_path: str = LEARNING_STORE_PATH):
        self.store_path = store_path
        self.entries: List[LearningEntry] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path) as f:
                    data = json.load(f)
                self.entries = [LearningEntry(**e) for e in data]
            except Exception:
                self.entries = []

    def _save(self) -> None:
        with open(self.store_path, "w") as f:
            json.dump([e.model_dump() for e in self.entries], f, indent=2)

    def add(self, entry: LearningEntry) -> None:
        self.entries.append(entry)
        self._save()

    def get_relevant(self, query_type: str, limit: int = 3) -> List[LearningEntry]:
        """Failures first (highest learning value), then successes."""
        failures = [e for e in self.entries if not e.success and e.query_type == query_type]
        successes = [e for e in self.entries if e.success and e.query_type == query_type]
        return (failures + successes)[-limit:]

    def get_recent(self, limit: int = 5) -> List[LearningEntry]:
        return self.entries[-limit:]

    def format_for_prompt(self, query_type: Optional[str] = None) -> str:
        """Format learnings as plain text for injection into planner prompts."""
        entries = self.get_relevant(query_type) if query_type else self.get_recent()
        if not entries:
            return "No previous learnings recorded yet."

        lines = ["=== PAST LEARNINGS (use to improve routing decisions) ==="]
        for e in entries:
            status = "SUCCESS" if e.success else "FAILURE"
            lines.append(f"\n[{status}] {e.query_summary}")
            lines.append(f"  Query type   : {e.query_type}")
            lines.append(f"  Agents used  : {', '.join(e.agents_used)}")
            lines.append(f"  Reasoning    : {e.plan_reasoning}")
            if not e.success and e.correction:
                lines.append(f"  !! CORRECTION: {e.correction}")
        lines.append("=== END OF LEARNINGS ===")
        return "\n".join(lines)


class ContextManager:
    """Central context hub — shared state and learnings across all agents."""

    def __init__(self):
        self.learning_store = LearningStore()
        self._global: Dict[str, Any] = {}
        self._per_session: Dict[str, Dict[str, Any]] = {}

    # --- Global context ---

    def set(self, key: str, value: Any) -> None:
        self._global[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._global.get(key, default)

    # --- Session-scoped context ---

    def set_session(self, session_id: str, key: str, value: Any) -> None:
        self._per_session.setdefault(session_id, {})[key] = value

    def get_session(self, session_id: str, key: str, default: Any = None) -> Any:
        return self._per_session.get(session_id, {}).get(key, default)

    def get_full_session_context(self, session_id: str) -> Dict[str, Any]:
        return dict(self._per_session.get(session_id, {}))

    # --- Learning ---

    def record_success(
        self,
        query_summary: str,
        query_type: str,
        agents_used: List[str],
        reasoning: str,
    ) -> None:
        self.learning_store.add(
            LearningEntry(
                query_summary=query_summary,
                query_type=query_type,
                agents_used=agents_used,
                plan_reasoning=reasoning,
                success=True,
            )
        )

    def record_failure(
        self,
        query_summary: str,
        query_type: str,
        agents_used: List[str],
        reasoning: str,
        feedback: str,
        correction: Optional[str] = None,
    ) -> None:
        self.learning_store.add(
            LearningEntry(
                query_summary=query_summary,
                query_type=query_type,
                agents_used=agents_used,
                plan_reasoning=reasoning,
                success=False,
                feedback=feedback,
                correction=correction,
            )
        )

    def get_learnings_for_prompt(self, query_type: Optional[str] = None) -> str:
        return self.learning_store.format_for_prompt(query_type)
