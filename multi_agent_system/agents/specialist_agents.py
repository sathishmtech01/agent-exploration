"""Specialist LlmAgents — leaf nodes that do the actual domain work.

Every agent:
  - Reads its task from session state via {agent_name_task} template substitution.
  - Reads the full structured request via {agent_name_request_context}.
  - Returns a structured JSON string that the orchestrator parses into AgentResponse.
  - Stores the result in session state via output_key.

NOTE: output_schema is intentionally NOT used here. OpenAI's structured-output API
requires `additionalProperties: false` on every dict field, which conflicts with
our flexible AgentResponse model. Instead we instruct the LLM to emit JSON in the
response text, then parse it ourselves in base_orchestrator._parse_agent_response().
This approach works identically across OpenAI, Groq, and Gemini.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

_DEFAULT_MODEL = "gemini-2.0-flash"

# Shared JSON output instruction appended to every agent
_JSON_OUTPUT_INSTRUCTION = """
## Response Format
You MUST respond with a valid JSON object matching this schema exactly:
{
  "request_id":        "<echo the request_id from the request above>",
  "agent_name":        "<your agent name>",
  "status":            "success" | "partial" | "failed",
  "output":            "<your full answer here>",
  "key_points":        ["point 1", "point 2", "point 3"],
  "structured_data":   {},
  "confidence":        0.0–1.0,
  "limitations":       ["any gaps or caveats"],
  "suggested_next_agent": null,
  "metadata":          {},
  "timestamp":         "<ISO timestamp>"
}
Do NOT wrap in markdown. Return ONLY the raw JSON object.
"""


def build_research_agent(model: str = _DEFAULT_MODEL) -> LlmAgent:
    return LlmAgent(
        name="research_agent",
        model=model,
        description="Deep-dives into any topic, retrieves facts, and synthesises knowledge.",
        output_key="research_agent_result",
        instruction="""You are an expert Research Analyst.

## Your Assigned Task
{research_agent_task}

## Full Request Context
{research_agent_request_context}

## Guidelines
- Provide comprehensive, accurate, well-structured information.
- Cite key facts and figures where applicable.
- Flag uncertainty and knowledge gaps clearly.
- In key_points, list the 3-5 most important findings.
- In structured_data, include any tables, lists, or metrics.
""" + _JSON_OUTPUT_INSTRUCTION,
    )


def build_code_agent(model: str = _DEFAULT_MODEL) -> LlmAgent:
    return LlmAgent(
        name="code_agent",
        model=model,
        description="Writes, reviews, explains, and debugs code across all languages.",
        output_key="code_agent_result",
        instruction="""You are a Senior Software Engineer.

## Your Assigned Task
{code_agent_task}

## Full Request Context
{code_agent_request_context}

## Guidelines
- Write clean, idiomatic, production-quality code.
- Include error handling.
- In output: provide the full code + explanation.
- In structured_data: include {"language": "...", "code": "...", "dependencies": [...]}.
- In key_points: list the key design decisions.
""" + _JSON_OUTPUT_INSTRUCTION,
    )


def build_data_agent(model: str = _DEFAULT_MODEL) -> LlmAgent:
    return LlmAgent(
        name="data_agent",
        model=model,
        description="Analyses data, identifies patterns, and provides statistical insight.",
        output_key="data_agent_result",
        instruction="""You are an Expert Data Analyst.

## Your Assigned Task
{data_agent_task}

## Full Request Context
{data_agent_request_context}

## Guidelines
- Think statistically and analytically.
- Identify trends, anomalies, and actionable patterns.
- In structured_data: include {"metrics": {}, "patterns": [], "visualisations": []}.
- In key_points: the top insights.
- In limitations: data quality issues or missing information.
""" + _JSON_OUTPUT_INSTRUCTION,
    )


def build_language_agent(model: str = _DEFAULT_MODEL) -> LlmAgent:
    return LlmAgent(
        name="language_agent",
        model=model,
        description="Handles translation, language processing, and tailored content generation.",
        output_key="language_agent_result",
        instruction="""You are an Expert Linguist and Content Specialist.

## Your Assigned Task
{language_agent_task}

## Full Request Context
{language_agent_request_context}

## Guidelines
- Translate preserving meaning, tone, and cultural nuance.
- In output: the translated / generated content.
- In structured_data: {"source_language": "...", "target_language": "...", "alternatives": []}.
- In key_points: linguistic notes and choices made.
- In limitations: ambiguities or missing context.
""" + _JSON_OUTPUT_INSTRUCTION,
    )


def build_summary_agent(model: str = _DEFAULT_MODEL) -> LlmAgent:
    return LlmAgent(
        name="summary_agent",
        model=model,
        description="Synthesises outputs from multiple agents into a unified response.",
        output_key="summary_agent_result",
        instruction="""You are a Master Synthesiser.

## Your Assigned Task
{summary_agent_task}

## Full Request Context
{summary_agent_request_context}

## Guidelines
- Merge all provided agent results into one coherent, user-facing response.
- Remove redundancy, highlight connections, preserve depth.
- In output: the complete, unified answer for the end user.
- In key_points: the top 3-5 takeaways across all agent results.
- In structured_data: {"agents_synthesised": [], "word_count": 0}.
""" + _JSON_OUTPUT_INSTRUCTION,
    )
