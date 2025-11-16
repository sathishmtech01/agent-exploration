import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.cli import adk_web_server
from google.adk.sessions import Session
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.planners import BasePlanner, BuiltInPlanner, PlanReActPlanner
from google.adk.cli import adk_web_server


topic_agent = LlmAgent(
    name="IdeaAgent",  # Unique identifier for the agent
    model="gpt-4o",  # Model spec to use
    description="Brainstorms blog post ideas.",  # Purpose of this agent
    instruction=(
        "You Ops buddy agent "
        "and return only the ideas."
    ),
)

# 2. Initialize the session
# user_id = "user123"
# session_id = "session456"
# session_service = InMemorySessionService()
# session = Session(user_id=user_id, session_id=session_id)
# runner = Runner(agent=topic_agent, app_name="test", session_service=session_service)


# from google.adk.cli.fast_api import get_fast_api_app
# import uvicorn
#
# # agent = your loaded agent in a variable
#
# app = get_fast_api_app(
#     agents_dir=topic_agent,
#     host="0.0.0.0",
#     port=9000,
#     web=False
#     # open_browser=False
# )

# uvicorn.run(app, host="0.0.0.0", port=9000)