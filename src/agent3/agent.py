from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

openai_llm = LiteLlm(
    model="gpt-4o"
)

root_agent = Agent(
    name="IdeaAgent",  # Unique identifier for the agent
    model=openai_llm,  # Model spec to use
    description="Brainstorms blog post ideas.",  # Purpose of this agent
    instruction=(
        "You are an Idea Agent "
        "and return only the ideas."
    ),
)
