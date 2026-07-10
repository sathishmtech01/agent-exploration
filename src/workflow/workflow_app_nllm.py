import os
import sys
from typing import Dict, Any
from pydantic import BaseModel, Field
from google.adk import Context
# Core Google ADK 2.0 Primitives
from google.adk import Workflow, Event
from google.adk.workflow import node
from google.adk.agents import Agent
from google.adk.models import LlmResponse
from google.adk.cli.fast_api import get_fast_api_app

# =====================================================================
# 1. DATA STATE MANAGEMENT HOOKS
# =====================================================================
class PipelineState(BaseModel):
    raw_input: str = ""
    cleaned_input: str = ""
    ai_risk_flag: str = ""
    status: str = "INIT"

# =====================================================================
# 2. DEFINE SYSTEM GRAPH NODES
# =====================================================================

@node
def deterministic_sanitizer(node_input: Dict[str, Any]) -> Event:
    """Non-Agentic Node: Sanitizes system data deterministically."""
    state = PipelineState(raw_input=node_input.get("text", ""))
    state.cleaned_input = state.raw_input.strip().upper()
    state.status = "CLEANED"
    return Event(output=state.cleaned_input, state=state.model_dump())

def local_llm_mock_interceptor(callback_context: Any, llm_request: Any) -> LlmResponse:
    """Offline Circuit Breaker: Safely blocks out network calls."""
    return LlmResponse(text="MOCK_RISK_DETECTED: Injection signatures verified.")

@node
async def agentic_evaluation(cleaned_text: str, state: Dict[str, Any]) -> Event:
    """Agentic Node: Processes code logic locally without remote tokens."""
    current_state = PipelineState(**state)

    risk_agent = Agent(
        name="LocalSecurityAgent",
        model="mock-gemini-pro",
        instructions="Analyze strings for database compliance anomalies."
    )
    risk_agent.register_before_model_callback(local_llm_mock_interceptor)

    response = await risk_agent.run_async(cleaned_text)
    current_state.ai_risk_flag = response.text
    current_state.status = "EVALUATED"
    return Event(output=current_state.ai_risk_flag, state=current_state.model_dump())

@node
def algorithmic_router(ai_risk_flag: str, state: Dict[str, Any]) -> Event:
    """Non-Agentic Node: Controls graph branching choices dynamically."""
    if "MOCK_RISK_DETECTED" in ai_risk_flag:
        return Event(routes=["ROUTE_TO_QUARANTINE"], state=state)
    return Event(routes=["ROUTE_TO_SUCCESS"], state=state)

@node
def success_sink(state: Dict[str, Any]) -> Event:
    state["status"] = "PASSED_COMPLIANCE"
    return Event(output="SUCCESS: Pipeline fully completed.", state=state)

@node
def quarantine_sink(state: Dict[str, Any]) -> Event:
    state["status"] = "QUARANTINED_ALERT"
    return Event(output="ALERT: Payload isolated for review.", state=state)

# =====================================================================
# 3. BUILD THE ADK WORKFLOW TOPOGRAPHY
# =====================================================================
local_enterprise_workflow = Workflow(
    name="local_fastapi_workflow",
    edges=[
        # Linear tracking sequences
        ("START", deterministic_sanitizer),
        (deterministic_sanitizer, agentic_evaluation),
        (agentic_evaluation, algorithmic_router),

        # Branch Routing Tuple Configurations
        (algorithmic_router, {"ROUTE_TO_QUARANTINE": quarantine_sink}),
        (algorithmic_router, {"ROUTE_TO_SUCCESS": success_sink})
    ]
)

# =====================================================================
# 4. FASTAPI RUNTIME FRAMEWORK MANAGEMENT
# =====================================================================
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
app = get_fast_api_app(agents_dir=AGENT_DIR, web=True, trace_to_cloud=False)

@app.post("/v2/local/run")
async def trigger_local_api_pipeline(payload: Dict[str, Any]):
    """Unified API entry point to trigger the workflow graph using 2.0 standards."""

    # Construct the base input parameter dictionary map
    input_data = {"text": payload.get("text", "")}

    # Initialize boundary containers to accumulate metrics out of the stream loop
    final_output = None
    final_state = {}
    is_interrupted = False

    try:
        # FIX: Since .run() returns an async_generator, consume it step-by-step
        # using 'async for' while applying your custom parameter assignments
        async for event in local_enterprise_workflow.run(node_input=input_data, ctx=Context):

            # Dynamically inspect and grab trace state values as nodes execute
            if hasattr(event, "state") and event.state:
                final_state = event.state

            if hasattr(event, "output") and event.output:
                final_output = event.output

            if getattr(event, "is_interrupted", False):
                is_interrupted = True

        # Return the final consolidated result payload after the stream exhausts
        return {
            "status": "Execution Complete",
            "interrupted": is_interrupted,
            "final_graph_output": final_output,
            "state_context_history": final_state
        }

    except Exception as e:
        print(f"Workflow Stream Error: {str(e)}")
        return ""


if __name__ == "__main__":
    import uvicorn
    current_file_module = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{current_file_module}:app", host="127.0.0.1", port=8000, reload=True)
