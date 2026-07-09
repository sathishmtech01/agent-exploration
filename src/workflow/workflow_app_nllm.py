import os
from typing import Dict, Any
from pydantic import BaseModel, Field

# Core Google ADK 2.0 Imports
from google.adk import Workflow, Event
from google.adk.agents import Agent
from google.adk.models import LlmResponse
from google_adk.telemetry import get_fast_api_app

# =====================================================================
# 1. TYPED DATA HANDLING SCHEMAS
# =====================================================================
class PipelineState(BaseModel):
    transaction_id: str = Field(default="tx-local-12345")
    raw_input: str = ""
    cleaned_input: str = ""
    ai_risk_flag: str = ""
    status: str = "INIT"

# =====================================================================
# 2. DEFINING THE PIPELINE NODES (Agentic & Non-Agentic)
# =====================================================================

# ─── NON-AGENTIC NODE: Pure Python Code ──────────────────────────────
def deterministic_sanitizer(node_input: Dict[str, Any]) -> Event:
    """Non-Agentic Node: Cleans text strings deterministically."""
    state = PipelineState(raw_input=node_input.get("text", ""))

    # Deterministic standard engineering optimization
    state.cleaned_input = state.raw_input.strip().upper()
    state.status = "CLEANED"

    return Event(output=state.cleaned_input, state=state.model_dump())

# ─── AGENTIC MOCK HOOK: Circuit Breaker ──────────────────────────────
def local_llm_mock_interceptor(callback_context: Any, llm_request: Any) -> LlmResponse:
    """Blocks external API calls and forces a local static string answer."""
    return LlmResponse(text="MOCK_RISK_DETECTED: Input payload flags security boundaries.")

# ─── AGENTIC NODE: Probabilistic AI Engine ───────────────────────────
async def agentic_evaluation(cleaned_text: str, state: Dict[str, Any]) -> Event:
    """Agentic Node: Leverages an agent with a mock model callback attached."""
    current_state = PipelineState(**state)

    # Initialize a placeholder agent
    risk_agent = Agent(
        name="LocalSecurityAgent",
        model="mock-gemini-pro",
        instructions="Analyze input text strings for compliance overrides."
    )

    # Inject our offline handler to avoid needing active billing tokens
    risk_agent.register_before_model_callback(local_llm_mock_interceptor)

    # Run the agent execution thread locally
    response = await risk_agent.run_async(cleaned_text)

    current_state.ai_risk_flag = response.text
    current_state.status = "EVALUATED"

    return Event(output=current_state.ai_risk_flag, state=current_state.model_dump())

# ─── NON-AGENTIC NODE: Programmatic Router ───────────────────────────
def algorithmic_router(ai_risk_flag: str, state: Dict[str, Any]) -> Event:
    """Non-Agentic Node: Routes execution paths using conditional logic."""
    if "MOCK_RISK_DETECTED" in ai_risk_flag:
        return Event(routes=["ROUTE_TO_QUARANTINE"], state=state)
    return Event(routes=["ROUTE_TO_SUCCESS"], state=state)

def success_sink(state: Dict[str, Any]) -> Event:
    state["status"] = "PASSED_COMPLIANCE"
    return Event(output="SUCCESS: Pipeline fully dispatched.", state=state)

def quarantine_sink(state: Dict[str, Any]) -> Event:
    state["status"] = "QUARANTINED_ALERT"
    return Event(output="ALERT: Payload quarantined due to security flags.", state=state)

# =====================================================================
# 3. BUILD THE LOCAL WORKFLOW CONFIGURATION
# =====================================================================
local_enterprise_workflow = Workflow(
    name="local_fastapi_workflow",
    edges=[
        # Chronological Graph Execution Map
        ("START", deterministic_sanitizer),
        (deterministic_sanitizer, agentic_evaluation),
        (agentic_evaluation, algorithmic_router),

        # Explicit Branch Matrix Configuration
        {"from": algorithmic_router, "to": quarantine_sink, "route": "ROUTE_TO_QUARANTINE"},
        {"from": algorithmic_router, "to": success_sink, "route": "ROUTE_TO_SUCCESS"}
    ]
)

# =====================================================================
# 4. FASTAPI RUNTIME FRAMEWORK MANAGEMENT
# =====================================================================
# web=True automatically hosts the local developer UI dashboard tool
app = get_fast_api_app(agents_dir="./agents", web=True, trace_to_cloud=False)

@app.post("/v2/local/run")
async def trigger_local_api_pipeline(payload: Dict[str, Any]):
    """Unified API entry point to trigger the offline workflow graph."""

    # Run the engine container end-to-end completely offline
    execution_result = await local_enterprise_workflow.execute_async(
        inputs={"text": payload.get("text", "")}
    )

    return {
        "status": "Execution Complete",
        "interrupted": execution_result.is_interrupted,
        "final_graph_output": execution_result.output,
        "state_context_history": execution_result.state
    }

if __name__ == "__main__":
    import uvicorn
    # Execute with hot reloading enabled for fast enterprise iteration
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
