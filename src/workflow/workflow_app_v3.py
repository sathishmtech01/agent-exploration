import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Core Google ADK 2.0 Production Imports
from google.adk import Workflow, Event
from google.adk.events import RequestInput
from google.adk.workflow import node
from google.adk.cli.fast_api import get_fast_api_app

# =====================================================================
# 1. DEFINE YOUR DECOUPLED ADK WORKFLOW NODES
# =====================================================================

@node(rerun_on_resume=False)
async def get_user_approval(node_input: Any):
    """
    Step 1: Yields a RequestInput to pause the workflow.
    ADK injects its own tracking contexts under the hood automatically.
    """
    # Yield breaks out of the execution generator, switching the graph to a paused state
    yield RequestInput(message="Please approve this request (Yes/No)")


@node(rerun_on_resume=True)
async def handle_process(node_input: Any, state: Dict[str, Any]):
    """
    Step 2: Receives the user's manual entry and logs the deployment verdict.
    """
    # When resume() is invoked, the resume_data string lands directly in node_input
    user_response = node_input if isinstance(node_input, str) else str(node_input.get("text", ""))

    if user_response.lower() == "yes":
        return Event(output="Approved", state=state)
    return Event(output="Denied", state=state)

# =====================================================================
# 2. CONSTRUCT THE ADK WORKFLOW TOPOGRAPHY
# =====================================================================
orchestrated_workflow = Workflow(
    name="hitl_orchestrator_pipeline",
    edges=[
        # Kicks off by asking the user, then jumps to handle the evaluation step
        ("START", get_user_approval),
        (get_user_approval, handle_process)
    ]
)

# =====================================================================
# 3. FASTAPI RUNTIME AND SESSION ENGINE WAREHOUSE
# =====================================================================
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
app = get_fast_api_app(agents_dir=AGENT_DIR, web=True, trace_to_cloud=False)

# Local Storage Array to track workflows awaiting response elements
SESSION_WAREHOUSE: Dict[str, Any] = {}

class InitRequest(BaseModel):
    session_id: str
    text_content: str = "Initial execution payload parameters"

class ResumePayload(BaseModel):
    session_id: str
    user_choice: str = "yes"  # Valid options: "yes" or "no"

@app.post("/v2/local/run")
async def trigger_local_api_pipeline(payload: InitRequest):
    """
    Initial Gateway: Offloads execution to the app's hidden runner.
    This provisions the required 'ctx' natively behind the scenes.
    """
    input_data = {"text": payload.text_content}
    final_output = None
    is_paused = False
    final_state = {}

    try:
        # FIX: Delegate execution to app.state.workflow_runner.run()
        # This returns the correct async event generator stream cleanly!
        async for event in app.state.workflow_runner.run(
                node=orchestrated_workflow,
                node_input=input_data
        ):
            if hasattr(event, "state") and event.state:
                final_state = event.state
            if hasattr(event, "output") and event.output:
                final_output = event.output
            if getattr(event, "is_interrupted", False):
                is_paused = True

        # Cache the workflow instance so we can target it on the resume route
        SESSION_WAREHOUSE[payload.session_id] = orchestrated_workflow

        return {
            "session_id": payload.session_id,
            "status": "PAUSED_AWAITING_HUMAN" if is_paused else "COMPLETED",
            "graph_output": final_output,
            "state": final_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Initialization Error: {str(e)}")


@app.post("/v2/local/resume")
async def resume_local_api_pipeline(payload: ResumePayload):
    """
    Resume Gateway: Restarts the frozen graph using the server's runtime.
    """
    if payload.session_id not in SESSION_WAREHOUSE:
        raise HTTPException(status_code=404, detail="Active session context not found.")

    paused_workflow = SESSION_WAREHOUSE[payload.session_id]
    final_output = None
    final_state = {}

    try:
        # FIX: Delegate resume loop tasks to the framework runner too!
        async for event in app.state.workflow_runner.resume(
                node=paused_workflow,
                resume_data=payload.user_choice
        ):
            if hasattr(event, "state") and event.state:
                final_state = event.state
            if hasattr(event, "output") and event.output:
                final_output = event.output

        del SESSION_WAREHOUSE[payload.session_id]

        return {
            "session_id": payload.session_id,
            "status": "SUCCESSFULLY_COMPLETED",
            "final_orchestration_result": final_output,
            "state": final_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Resumption Failure: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    current_file_module = os.path.splitext(os.path.basename(__file__))
    uvicorn.run(f"{current_file_module}:app", host="127.0.0.1", port=8000, reload=True)
