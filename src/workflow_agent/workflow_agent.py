import os
from typing import Dict, Any, List
from pydantic import BaseModel
from google.adk.agents import Agent
from google.adk.workflows import Workflow
from google.adk.context import ToolContext

# --- 1. Define Data Schemas for Format Integrity ---
class AgentInput(BaseModel):
    query: str
    entitlements: Dict[str, bool]
    auth_token: str

class AgentOutput(BaseModel):
    data: str
    reasoning: str  # Mandatory for explainability

# --- 2. Define Domain Agents ---
# Financial Agent: Specialist in fiscal data
financial_agent = Agent(
    name="Financial_Agent",
    model="gemini-2.0-flash",
    instruction="Analyze the fiscal impact of: {query}. Output JSON with 'data' and your 'reasoning'.",
    input_schema=AgentInput,
    output_schema=AgentOutput
)

# Compliance Agent: Specialist in regulatory checks
compliance_agent = Agent(
    name="Compliance_Agent",
    model="gemini-2.0-flash",
    instruction="Check if {query} meets standards. Reference previous data if available. Output 'data' and 'reasoning'.",
    input_schema=AgentInput,
    output_schema=AgentOutput
)

# --- 3. Deterministic Logic & Formatting ---
def prepare_context(node_input: str, tool_context: ToolContext) -> AgentInput:
    """Prepares the formatted input with auth and entitlements from session state."""
    return AgentInput(
        query=node_input,
        entitlements=tool_context.state.get("user_entitlements", {}),
        auth_token=tool_context.state.get("auth_token", "GUEST_TOKEN")
    )

def router_logic(user_query: str):
    """Deterministic routing based on query keywords."""
    if "check" in user_query.lower():
        # Order: Compliance -> Financial
        return ["compliance_agent", "financial_agent"]
    # Default Order: Financial -> Compliance
    return ["financial_agent", "compliance_agent"]

# --- 4. Assemble the Workflow ---
# Using graph-based edges for complete traceability
root_workflow = Workflow(
    name="DeterministicOrchestrator",
    nodes=[financial_agent, compliance_agent],
    edges=[
        # START -> Logic -> Agent 1 -> Agent 2 -> END
        ("START", prepare_context),
        (prepare_context, financial_agent),
        (financial_agent, compliance_agent),
        (compliance_agent, "END")
    ]
)

# --- 5. Execution Wrapper (For Local Testing) ---
if __name__ == "__main__":
    from google.adk.runner import InMemoryRunner

    # Initialize runner with your project ID
    os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"
    runner = InMemoryRunner(root_workflow)

    # Simulate production session with entitlements
    session = runner.session_service().create_session(
        user_id="user_123",
        initial_state={
            "user_entitlements": {"view_finance": True},
            "auth_token": "secure_mcp_token_99"
        }
    )

    # Run the query
    result = runner.run(session.session_key, "Analyze the Q3 budget check")
    for event in result:
        print(f"[{event.author}]: {event.content}")
