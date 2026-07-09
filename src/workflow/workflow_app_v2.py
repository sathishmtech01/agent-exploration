import os
import uuid
from typing import Dict, Any, List
from pydantic import BaseModel, Field

# Core Google ADK New Primitives
from google.adk import Event, RequestInput
from google_adk.core import Agent, Model
from google_adk.workflows import workflow_node, Workflow, workflowagent
from google_adk.telemetry import get_fast_api_app

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "your-enterprise-project")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
gemini_model = Model(name="gemini-2.5-pro")

# =====================================================================
# 1. ENTERPRISE TYPED STATE (Data Handling Guidelines)
# =====================================================================
class PipelineState(BaseModel):
    """The single source of truth passed via Event.state across nodes."""
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str = ""
    sanitized_text: str = ""
    llm_analysis: str = ""
    compliance_passed: bool = True
    human_notes: str = ""

# =====================================================================
# 2. DEFINING THE HYBRID NODES (Agentic & Non-Agentic)
# =====================================================================

# NON-AGENTIC NODE: Pure Python data preparation
@workflow_node
def sanitize_input_node(node_input: Dict[str, Any]) -> Event:
    """Non-Agentic Node: Cleans text deterministically using standard code."""
    # Initialize state from incoming data mapping
    state = PipelineState(raw_text=node_input.get("text", ""))
    state.sanitized_text = state.raw_text.strip().upper() # Safe engineering formatting

    # Emit an Event packaging the structured state for the downstream pipeline
    return Event(output=state.sanitized_text, state=state.model_dump())

# AGENTIC NODE: Probabilistic reasoning node
@workflow_node
async def agentic_llm_analysis_node(node_input: str, state: Dict[str, Any]) -> Event:
    """Agentic Node: Leverages an LLM agent to evaluate the structural text string."""
    current_state = PipelineState(**state)

    analyst = Agent(
        name="ComplianceAnalyst",
        model=gemini_model,
        instructions="Analyze this system input string. If it contains 'EXPLOIT', reply with 'FAIL'."
    )

    response = await analyst.run_async(node_input)
    current_state.llm_analysis = response.text
    current_state.compliance_passed = "FAIL" not in response.text.upper()

    # Pass along state to the routing evaluate step
    return Event(output=current_state.compliance_passed, state=current_state.model_dump())

# NON-AGENTIC ROUTER NODE: Implements explicit graph routing
@workflow_node
def evaluate_routing_node(compliance_passed: bool, state: Dict[str, Any]) -> Event:
    """Non-Agentic Node: Decides the graph path matching adk.dev/graphs/routes/."""
    # Direct conditional dispatch via explicit string routing keys
    if not compliance_passed:
        return Event(routes=["TRIGGER_HUMAN_INTERRUPT"], state=state)
    return Event(routes=["AUTO_DISPATCH"], state=state)

# HUMAN-IN-THE-LOOP NODE: Using the modern Interrupt paradigm
# By using rerun_on_resume=False, ADK skips this node entirely when re-hydrated
@workflow_node(rerun_on_resume=False)
def human_gatekeeper_node(state: Dict[str, Any]):
    """HITL Node: Implements modern yield RequestInput schema."""
    current_state = PipelineState(**state)

    # Yielding a RequestInput event suspends the state engine without locking web threads
    yield RequestInput(
        interrupt_id=f"gatekeeper-{current_state.transaction_id}", # Stable identifier
        message="Vulnerability identified by AI. Human verification required.",
        payload={"analysis": current_state.llm_analysis} # Context for UI rendering
    )

    # When the graph is unblocked via workflow.resume_async(), execution picks up here:
    # ADK automatically injects the reviewer's input directly into the node scope
    return Event(routes=["RESOLVE_REVIEW"], state=state)

@workflow_node
def dispatch_success_node(state: Dict[str, Any]) -> Event:
    return Event(output="SUCCESS: Dispatched to external database records.", state=state)

@workflow_node
def dispatch_failure_node(state: Dict[str, Any]) -> Event:
    return Event(output="TERMINATED: Flagged as threat vector.", state=state)


# =====================================================================
# 3. ENTERPRISE GRAPH BUILDING (adk.dev/graphs/dynamic/)
# =====================================================================
# Use workflow.Chain and Edge maps to declare explicit graph topography
workflow_graph = workflowagent.New(
    edges=[
        # Primary Pipeline Sequence Flow
        ("START", sanitize_input_node),
        (sanitize_input_node, agentic_llm_analysis_node),
        (agentic_llm_analysis_node, evaluate_routing_node),

        # Explicit Routing Branch Matrix
        {"from": evaluate_routing_node, "to": human_gatekeeper_node, "route": "TRIGGER_HUMAN_INTERRUPT"},
        {"from": evaluate_routing_node, "to": dispatch_success_node, "route": "AUTO_DISPATCH"},

        # Human Resolution Branches
        {"from": human_gatekeeper_node, "to": dispatch_failure_node, "route": "RESOLVE_REVIEW"}
    ]
)

# =====================================================================
# 4. FASTAPI STATE MANAGEMENT ENGINE
# =====================================================================
app = get_fast_api_app(agents_dir="./agents", web=True, trace_to_cloud=True)

# In-memory session warehouse. Swap out with Google Cloud Firestore for horizontal scaling.
SESSION_DATABASE: Dict[str, Any] = {}

@app.post("/v2/pipeline/run")
async def execute_enterprise_flow(payload: Dict[str, Any]):
    """Triggers the new graph-based workflow instance."""
    # Run the engine asynchronously
    execution_result = await workflow_graph.execute_async(inputs={"text": payload.get("text", "")})

    # Retrieve trackable state context metadata
    tx_id = execution_result.state.get("transaction_id")
    SESSION_DATABASE[tx_id] = execution_result # Capture complete serializable context record

    return {
        "transaction_id": tx_id,
        "current_status": "PAUSED_AWAITING_HUMAN" if execution_result.is_interrupted else "COMPLETED",
        "output": execution_result.output if not execution_result.is_interrupted else None
    }

@app.post("/v2/pipeline/resume")
async def resume_enterprise_flow(transaction_id: str, allow_action: bool):
    """Processes incoming human responses and unlocks the graph engine."""
    if transaction_id not in SESSION_DATABASE:
        return {"error": "Target transaction trace session not found."}

    # Re-hydrate the frozen session context
    paused_context = SESSION_DATABASE[transaction_id]

    # In the new syntax, human actions are packaged directly as resume data mappings
    human_resolution_payload = {
        "action": "AUTO_DISPATCH" if allow_action else "RESOLVE_REVIEW",
        "notes": "Override processed manually by corporate admin operations."
    }

    # Resume engine timeline using modern execution paradigms
    resumed_result = await workflow_graph.resume_async(
        context=paused_context,
        resume_data=human_resolution_payload
    )

    SESSION_DATABASE[transaction_id] = resumed_result
    return {
        "transaction_id": transaction_id,
        "final_pipeline_output": resumed_result.output,
        "history": [span.get("node_name") for span in resumed_result.history]
    }
