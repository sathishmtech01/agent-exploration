"""SecurityLayer — defense-in-depth for every boundary of the multi-agent system.

Guards applied at four points:
  1. INPUT   — user query before it enters the orchestrator
  2. REQUEST — AgentRequest before it is sent to a specialist agent
  3. RESPONSE — AgentResponse received from a specialist agent
  4. OUTPUT  — final text returned to the caller

Checks performed:
  - Prompt injection detection (jailbreak / override patterns)
  - PII detection & redaction (email, phone, SSN, credit card, API keys)
  - Secrets / API key leakage scan
  - Malicious content patterns (shell injection, SQL injection, XSS)
  - Response content safety (harmful output patterns)
  - Session rate limiting (per session_id)
  - Max token length enforcement
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.agent_message import AgentRequest, AgentResponse


# ─────────────────────────────────────────────── data classes

@dataclass
class SecurityViolation:
    check: str          # which check triggered
    severity: str       # "low" | "medium" | "high" | "critical"
    detail: str         # human-readable description
    blocked: bool       # True  → request must be blocked
                        # False → warning only, execution continues


@dataclass
class SecurityCheckResult:
    passed: bool
    violations: List[SecurityViolation] = field(default_factory=list)
    sanitized_text: Optional[str] = None   # cleaned version if redaction was applied

    @property
    def has_critical(self) -> bool:
        return any(v.severity == "critical" for v in self.violations)

    @property
    def has_blocking(self) -> bool:
        return any(v.blocked for v in self.violations)

    def summary(self) -> str:
        if self.passed:
            return "OK"
        parts = [f"{v.severity.upper()}:{v.check}" for v in self.violations]
        return " | ".join(parts)


# ─────────────────────────────────────────── compiled regex patterns

# Prompt injection / jailbreak
_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "ignore_instructions"),
    (r"(?i)(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are))\s+.{0,60}(without\s+(restriction|limit|filter)|no\s+(rule|limit|filter))", "persona_override"),
    (r"(?i)system\s*prompt\s*[:=]", "system_prompt_override"),
    (r"(?i)(jailbreak|DAN\s+mode|developer\s+mode|god\s+mode)", "jailbreak_keyword"),
    (r"(?i)disregard\s+(your\s+)?(safety|ethical|content)\s+(guidelines?|filter|policy)", "safety_bypass"),
    (r"(?i)(<\s*system\s*>|<\s*\/\s*system\s*>|\[SYSTEM\]|\[\/SYSTEM\])", "xml_system_tag"),
    (r"(?i)(repeat\s+after\s+me|say\s+exactly|output\s+verbatim).{0,80}(password|secret|key|token)", "extraction_attempt"),
]

# PII patterns — used for redaction
_PII_PATTERNS: List[Tuple[str, str, str]] = [
    # (pattern, replacement, label)
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]", "email"),
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", "phone"),
    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "[SSN]", "ssn"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "[CARD]", "credit_card"),
    (r"\b(?:sk|pk|rk|ak|tok|key)[-_]?(?:live|test|prod)?[-_]?[A-Za-z0-9]{20,}\b", "[API_KEY]", "api_key"),
    (r"(?i)(?:password|passwd|pwd)\s*[:=]\s*\S+", "[PASSWORD]", "password"),
    (r"(?i)(?:api[_\-]?key|secret[_\-]?key|access[_\-]?token)\s*[:=]\s*\S+", "[SECRET]", "secret"),
    (r"\b[0-9a-fA-F]{32,64}\b", "[HASH_OR_KEY]", "hash_or_key"),
]

# Shell / command injection
_SHELL_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)(;\s*rm\s+-rf|&&\s*rm\s+-rf|\|\s*rm\s+-rf)", "shell_rm_rf"),
    (r"(?i)(;\s*(?:cat|curl|wget|nc|netcat|ncat)\s+)", "shell_exfil"),
    (r"(?i)(\$\(|`)[^`$]{0,200}(\)|`)", "shell_substitution"),
    (r"(?i)(eval\s*\(|exec\s*\(|subprocess\.|os\.system\()", "code_exec"),
    (r"(?i)(\/etc\/passwd|\/etc\/shadow|\/proc\/self)", "path_traversal"),
]

# SQL injection
_SQL_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)(\bUNION\s+(?:ALL\s+)?SELECT\b)", "sql_union"),
    (r"(?i)(;\s*DROP\s+TABLE|;\s*DELETE\s+FROM|;\s*TRUNCATE\s+TABLE)", "sql_ddl"),
    (r"(?i)('\s*OR\s*'1'\s*=\s*'1|'\s*OR\s*1\s*=\s*1)", "sql_or_1_eq_1"),
    (r"(?i)(--\s*$|#\s*$)", "sql_comment_terminator"),
]

# XSS
_XSS_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)<\s*script[^>]*>", "xss_script_tag"),
    (r"(?i)javascript\s*:", "xss_javascript_proto"),
    (r"(?i)on(?:load|error|click|mouse\w+)\s*=", "xss_event_handler"),
]

# Harmful output indicators
_HARMFUL_OUTPUT_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)(step[\s\-]*by[\s\-]*step.{0,60}(bomb|weapon|exploit|malware|ransomware))", "harmful_instructions"),
    (r"(?i)(synthesize\s+.{0,40}(drug|nerve\s+agent|explosive))", "harmful_synthesis"),
    (r"(?i)(here\s+is\s+(how\s+to\s+)?(hack|crack|bypass|exploit)\s+)", "harmful_hacking"),
]

# Max input length (characters)
_MAX_INPUT_CHARS = 8_000
_MAX_OUTPUT_CHARS = 50_000

# Rate limit: max requests per session per minute
_RATE_LIMIT_WINDOW_SEC = 60
_RATE_LIMIT_MAX_REQUESTS = 30


def _compile(patterns: List[Tuple[str, ...]]) -> List[Tuple[re.Pattern, ...]]:
    return [(re.compile(p[0]),) + p[1:] for p in patterns]


_INJECTION_RE = _compile(_INJECTION_PATTERNS)
_SHELL_RE = _compile(_SHELL_PATTERNS)
_SQL_RE = _compile(_SQL_PATTERNS)
_XSS_RE = _compile(_XSS_PATTERNS)
_HARMFUL_RE = _compile(_HARMFUL_OUTPUT_PATTERNS)
_PII_RE = [(re.compile(p), repl, label) for p, repl, label in _PII_PATTERNS]


# ─────────────────────────────────────────────────── SecurityLayer


class SecurityLayer:
    """
    Stateful security guard wired into every boundary of the orchestrator.

    Usage:
        security = SecurityLayer()

        # At system entry
        result = security.check_input(user_query, session_id)
        if result.has_blocking:
            raise SecurityError(result.summary())
        safe_query = result.sanitized_text or user_query

        # Before sending to agent
        result = security.check_request(agent_request)

        # After receiving from agent
        result = security.check_response(agent_response)

        # Before returning to caller
        result = security.check_output(final_text)
        safe_output = result.sanitized_text or final_text
    """

    def __init__(
        self,
        max_input_chars: int = _MAX_INPUT_CHARS,
        max_output_chars: int = _MAX_OUTPUT_CHARS,
        rate_limit_max: int = _RATE_LIMIT_MAX_REQUESTS,
        rate_limit_window_sec: int = _RATE_LIMIT_WINDOW_SEC,
        redact_pii: bool = True,
        block_injections: bool = True,
        block_malicious_code: bool = True,
    ):
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.rate_limit_max = rate_limit_max
        self.rate_limit_window_sec = rate_limit_window_sec
        self.redact_pii = redact_pii
        self.block_injections = block_injections
        self.block_malicious_code = block_malicious_code

        # session_id → list of timestamps
        self._rate_buckets: Dict[str, List[float]] = defaultdict(list)
        self._violation_log: List[Dict] = []

    # ─────────────────────────────────────── public API

    def check_input(self, text: str, session_id: str = "default") -> SecurityCheckResult:
        """Gate check on raw user query before orchestrator processing."""
        violations: List[SecurityViolation] = []

        # 1. Rate limiting
        v = self._rate_check(session_id)
        if v:
            violations.append(v)

        # 2. Length check
        if len(text) > self.max_input_chars:
            violations.append(SecurityViolation(
                check="max_input_length",
                severity="medium",
                detail=f"Input length {len(text)} exceeds limit {self.max_input_chars}",
                blocked=True,
            ))
            return SecurityCheckResult(passed=False, violations=violations)

        # 3. Prompt injection
        for pat, name in _INJECTION_RE:
            if pat.search(text):
                violations.append(SecurityViolation(
                    check=f"prompt_injection:{name}",
                    severity="critical",
                    detail=f"Prompt injection pattern detected: {name}",
                    blocked=self.block_injections,
                ))

        # 4. Shell / code injection
        for pat, name in _SHELL_RE:
            if pat.search(text):
                violations.append(SecurityViolation(
                    check=f"shell_injection:{name}",
                    severity="high",
                    detail=f"Shell/code injection pattern: {name}",
                    blocked=self.block_malicious_code,
                ))

        # 5. SQL injection
        for pat, name in _SQL_RE:
            if pat.search(text):
                violations.append(SecurityViolation(
                    check=f"sql_injection:{name}",
                    severity="high",
                    detail=f"SQL injection pattern: {name}",
                    blocked=self.block_malicious_code,
                ))

        # 6. XSS
        for pat, name in _XSS_RE:
            if pat.search(text):
                violations.append(SecurityViolation(
                    check=f"xss:{name}",
                    severity="medium",
                    detail=f"XSS pattern: {name}",
                    blocked=self.block_malicious_code,
                ))

        # 7. PII redaction (non-blocking, but sanitize)
        sanitized, pii_found = self._redact_pii(text)
        for label in pii_found:
            violations.append(SecurityViolation(
                check=f"pii:{label}",
                severity="low",
                detail=f"PII redacted: {label}",
                blocked=False,
            ))

        self._log_violations(violations, source="input", session_id=session_id)
        passed = not any(v.blocked for v in violations)
        return SecurityCheckResult(
            passed=passed,
            violations=violations,
            sanitized_text=sanitized if pii_found else None,
        )

    def check_request(self, request: AgentRequest) -> SecurityCheckResult:
        """Validate AgentRequest before dispatching to a specialist agent."""
        violations: List[SecurityViolation] = []

        # Schema completeness
        if not request.task or len(request.task.strip()) < 3:
            violations.append(SecurityViolation(
                check="empty_task",
                severity="high",
                detail="AgentRequest.task is empty or too short",
                blocked=True,
            ))

        if not request.agent_name:
            violations.append(SecurityViolation(
                check="missing_agent_name",
                severity="critical",
                detail="AgentRequest.agent_name is missing",
                blocked=True,
            ))

        # Scan the task text for injection
        task_result = self._scan_text_for_injection(request.task)
        violations.extend(task_result)

        # Scan context values
        for k, v in request.context.items():
            ctx_violations = self._scan_text_for_injection(str(v))
            for cv in ctx_violations:
                cv.check = f"context[{k}]:{cv.check}"
                cv.severity = "medium"  # downgrade for context (already processed)
                cv.blocked = False       # don't block on prior agent output
            violations.extend(ctx_violations)

        # Sanitize task PII
        sanitized_task, pii_found = self._redact_pii(request.task)
        for label in pii_found:
            violations.append(SecurityViolation(
                check=f"pii:{label}",
                severity="low",
                detail=f"PII in task redacted: {label}",
                blocked=False,
            ))

        self._log_violations(violations, source=f"request:{request.agent_name}")
        passed = not any(v.blocked for v in violations)
        return SecurityCheckResult(
            passed=passed,
            violations=violations,
            sanitized_text=sanitized_task if pii_found else None,
        )

    def check_response(self, response: AgentResponse) -> SecurityCheckResult:
        """Validate AgentResponse received from a specialist agent."""
        violations: List[SecurityViolation] = []

        # Check for failed status with suspicious output
        if response.status == "failed" and len(response.output) > 2000:
            violations.append(SecurityViolation(
                check="oversized_failure",
                severity="low",
                detail="Failed response has unusually large output",
                blocked=False,
            ))

        # Scan output for harmful content
        for pat, name in _HARMFUL_RE:
            if pat.search(response.output):
                violations.append(SecurityViolation(
                    check=f"harmful_output:{name}",
                    severity="critical",
                    detail=f"Harmful content detected in agent output: {name}",
                    blocked=True,
                ))

        # Scan for secrets / API keys leaking in output
        sanitized, pii_found = self._redact_pii(response.output)
        for label in pii_found:
            violations.append(SecurityViolation(
                check=f"pii_leak:{label}",
                severity="medium",
                detail=f"PII/secret in agent response redacted: {label}",
                blocked=False,
            ))

        self._log_violations(violations, source=f"response:{response.agent_name}")
        passed = not any(v.blocked for v in violations)
        return SecurityCheckResult(
            passed=passed,
            violations=violations,
            sanitized_text=sanitized if pii_found else None,
        )

    def check_output(self, text: str) -> SecurityCheckResult:
        """Final gate on text being returned to the user."""
        violations: List[SecurityViolation] = []

        if len(text) > self.max_output_chars:
            violations.append(SecurityViolation(
                check="max_output_length",
                severity="low",
                detail=f"Output truncated from {len(text)} to {self.max_output_chars} chars",
                blocked=False,
            ))
            text = text[: self.max_output_chars] + "\n\n[Output truncated for safety]"

        for pat, name in _HARMFUL_RE:
            if pat.search(text):
                violations.append(SecurityViolation(
                    check=f"harmful_output:{name}",
                    severity="critical",
                    detail=f"Harmful content in final output: {name}",
                    blocked=True,
                ))

        sanitized, pii_found = self._redact_pii(text)
        for label in pii_found:
            violations.append(SecurityViolation(
                check=f"pii_final:{label}",
                severity="medium",
                detail=f"PII in final output redacted: {label}",
                blocked=False,
            ))

        self._log_violations(violations, source="output")
        passed = not any(v.blocked for v in violations)
        return SecurityCheckResult(
            passed=passed,
            violations=violations,
            sanitized_text=sanitized if (pii_found or len(text) > self.max_output_chars) else None,
        )

    def get_violation_log(self) -> List[Dict]:
        """Return all recorded violations for audit / observability."""
        return list(self._violation_log)

    def session_stats(self, session_id: str) -> Dict:
        now = time.time()
        window_start = now - self.rate_limit_window_sec
        recent = [t for t in self._rate_buckets.get(session_id, []) if t >= window_start]
        return {
            "session_id": session_id,
            "requests_in_window": len(recent),
            "rate_limit_max": self.rate_limit_max,
            "window_sec": self.rate_limit_window_sec,
        }

    # ─────────────────────────────────────── private helpers

    def _rate_check(self, session_id: str) -> Optional[SecurityViolation]:
        now = time.time()
        window_start = now - self.rate_limit_window_sec
        bucket = self._rate_buckets[session_id]
        # Evict old entries
        self._rate_buckets[session_id] = [t for t in bucket if t >= window_start]
        self._rate_buckets[session_id].append(now)
        count = len(self._rate_buckets[session_id])
        if count > self.rate_limit_max:
            return SecurityViolation(
                check="rate_limit",
                severity="high",
                detail=f"Session '{session_id}' exceeded rate limit: {count}/{self.rate_limit_max} req/min",
                blocked=True,
            )
        return None

    def _redact_pii(self, text: str) -> Tuple[str, List[str]]:
        """Replace PII with placeholders. Returns (sanitized_text, list_of_found_labels)."""
        if not self.redact_pii:
            return text, []
        found: List[str] = []
        result = text
        for pat, replacement, label in _PII_RE:
            if pat.search(result):
                result = pat.sub(replacement, result)
                found.append(label)
        return result, found

    def _scan_text_for_injection(self, text: str) -> List[SecurityViolation]:
        violations: List[SecurityViolation] = []
        for pat, name in _INJECTION_RE:
            if pat.search(text):
                violations.append(SecurityViolation(
                    check=f"prompt_injection:{name}",
                    severity="critical",
                    detail=f"Injection pattern in text: {name}",
                    blocked=self.block_injections,
                ))
        for pat, name in _SHELL_RE:
            if pat.search(text):
                violations.append(SecurityViolation(
                    check=f"shell:{name}",
                    severity="high",
                    detail=f"Shell pattern: {name}",
                    blocked=self.block_malicious_code,
                ))
        return violations

    def _log_violations(
        self,
        violations: List[SecurityViolation],
        source: str,
        session_id: str = "",
    ) -> None:
        for v in violations:
            self._violation_log.append({
                "timestamp": time.time(),
                "source": source,
                "session_id": session_id,
                "check": v.check,
                "severity": v.severity,
                "detail": v.detail,
                "blocked": v.blocked,
            })
