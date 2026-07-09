import os
import uuid
import logging
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends

# Core Google ADK 2.0 Production Imports
from google.adk import Workflow, Event
from google.adk.workflow import node  # Official compact decorator
from google.adk.agents import Agent
from google.adk.models import LlmResponse

# Establish Enterprise Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adk_governance_registry")

# =====================================================================
# 1. GOVERNANCE SCHEMAS & METADATA CONTRACTS
# =====================================================================
class WorkflowMetadata(BaseModel):
    """Governance contract outlining operational ownership of deployed graphs."""
    workflow_id: str
    version: str = "1.0.0"
    environment: str = "production"
    owner_team: str = "SecOps"
    is_active: bool = True

class GovernanceState(BaseModel):
    """The central state schema governing graph compliance checks."""
    execution_id: str = Field(default_factory=lambda: f"exec-{uuid.uuid4().hex[:6]}")
    workflow_name: str = ""
    payload_raw: str = ""
    sanitized_output: str = ""
    audit_trail: List[str] = Field(default_factory=list)

# =====================================================================
# 2. PROGRAMMATIC ADK WORKFLOW REGISTRY (The Governance Center)
# =====================================================================
class EnterpriseWorkflowRegistry:
    """Manages the lifecycle, discoverability, and governance of all ADK graphs."""
    def __init__(self):
        # Maps unique string keys to executable Workflow instances
        self._registry: Dict[str, Workflow] = {}
        # Tracks operational structural metadata profiles
        self._metadata_store: Dict[str, WorkflowMetadata] = {}

    def register_workflow(self, name: str, workflow_instance: Workflow, meta: WorkflowMetadata):
        """Catalog and securely bind a workflow layout to the enterprise register."""
        if not meta.is_active:
            raise ValueError(f"Cannot register deactivated workflow version: {meta.version}")

        self._registry[name] = workflow_instance
        self._metadata_store[name] = meta
        logger.info(f"REGISTRY: Successfully registered governed workflow '{name}' [v{meta.version}]")

    def get_workflow(self, name: str) -> Workflow:
        """Dynamically discover and fetch a validated workflow at runtime."""
        if name not in self._registry:
            raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found in registry catalog.")

        # Verify active governance policy boundaries
        meta = self._metadata_store[name]
        if not meta.is_active:
            raise HTTPException(status_code=403, detail=f"Workflow '{name}' has been deprecated by governance policy.")

        return self._registry[name]

    def list_inventory(self) -> List[Dict[str, Any]]:
        """Provides an inventory breakdown for administrative monitoring tools."""
        return [
            {"workflow_name": k, "meta": v.model_dump()}
            for k, v in self._metadata_store.items()
        ]

# Instantiate global governance registry single source of truth
global_registry = EnterpriseWorkflowRegistry()

# =====================================================================
# 3. COMPONENT IMPLEMENTATIONS (Reusable Logic Nodes)
# =====================================================================

@node
def baseline_security_scrubber(node_input: Dict[str, Any]) -> Event:
    """Non-Agentic Node: Sanitizes system data parameters safely."""
    state = GovernanceState(
        workflow_name=node_input.get("requested_graph", "unknown"),
        payload_raw=node_input.get("text", "")
    )
    state.sanitized_output = state.payload_raw.strip().upper()
    state.audit_trail.append("STEP_01: SANITIZATION_SUCCESS")

    return Event(output=state.sanitized_output, state=state.model_dump())

def local_model_mock_wire(context: Any, request: Any) -> LlmResponse:
    """Local circuit breaker to bypass external billing key requirements."""
    return LlmResponse(text="EVALUATION: PASSED ENTERPRISE REGULATORY MATRIX Compliance check cleared.")

@node
async def governance_agentic_evaluator(sanitized_text: str, state: Dict[str, Any]) -> Event:
    """Agentic Node: Reviews regulatory compliance using an analytical model."""
    current_state = GovernanceState(**state)

    auditor = Agent(
        name="ComplianceAuditorAgent",
        model="mock-gemini-pro",
        instructions="Analyze input parameters for enterprise corporate rule matching."
    )
    auditor.register_before_model_callback(local_model_mock_wire)

    response = await auditor.run_async(sanitized_text)
    current_state.audit_trail.append(f"STEP_02: AGENTIC_ANALYSIS_COMPLETED ({response.text})")

    return Event(output=response.text, state=current_state.model_dump())

# =====================================================================
# 4. ASSEMBLING AND CATALOGING GOVERNED BLUEPRINTS
# =====================================================================

# Blueprint A: Core Transaction Audit Flow Path
financial_audit_graph = Workflow(
    name="financial_ledger_audit",
    edges=[
        ("START", baseline_security_scrubber),
        (baseline_security_scrubber, governance_agentic_evaluator)
    ]
)

# Populate our factory catalog with explicit audit compliance declarations
global_registry.register_workflow(
    name="ledger_compliance_pipeline",
    workflow_instance=financial_audit_graph,
    meta=WorkflowMetadata(
        workflow_id="wf-fin-990",
        version="2.1.0",
        owner_team="FinTech-Auditing"
    )
)

# =====================================================================
# 5. FASTAPI DISPATCH SERVICE WITH ENTERPRISE REPOSITORY LAYER
# =====================================================================
app = FastAPI(title="Google ADK 2.0 Governance Server")

# Decoupled Audit Warehouse Store (Simulating scalable PostgreSQL/BigQuery)
AUDIT_WAREHOUSE_REPORTS: Dict[str, Any] = {}

class ExecutionRequestPayload(BaseModel):
    workflow_target: str = Field(..., example="ledger_compliance_pipeline")
    text_content: str = Field(..., example="  Executing transaction payload balance matrix  ")

@app.post("/v2/governed/execute")
async def trigger_governed_workflow(payload: ExecutionRequestPayload):
    """
    Unified API orchestration gateway. Discovers the target workflow from the registry,
    executes it asynchronously, and commits the state execution ledger history.
    """
    # 1. Fetch the target workflow using the governance registry lookup mechanism
    target_workflow = global_registry.get_workflow(payload.workflow_target)

    # 2. Run the decoupled graph pipeline
    result = await target_workflow.execute_async(
        inputs={
            "requested_graph": payload.workflow_target,
            "text": payload.text_content
        }
    )

    # 3. Extract the tracked state execution ledger history
    compiled_state = result.state
    execution_id = compiled_state.get("execution_id")

    # 4. Commit records to the audit data warehouse repository
    AUDIT_WAREHOUSE_REPORTS[execution_id] = {
        "output": result.output,
        "full_lineage_history": compiled_state.get("audit_trail"),
        "raw_state_snapshot": compiled_state
    }

    return {
        "execution_id": execution_id,
        "status": "PROCESSED_UNDER_GOVERNANCE",
        "graph_output": result.output,
        "audit_logs": AUDIT_WAREHOUSE_REPORTS[execution_id]
    }

@app.get("/v2/governed/inventory")
def view_registry_inventory():
    """Administrative visibility endpoint to inspect current deployed active blueprints."""
    return {"registered_workflow_count": len(global_registry._registry), "catalog": global_registry.list_inventory()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("governance_engine.py:app", host="127.0.0.1", port=8000, reload=True)
