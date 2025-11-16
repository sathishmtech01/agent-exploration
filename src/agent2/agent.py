from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
import os
print(os.getenv('GROQ_API_KEY'))
root_agent = Agent(
    model=LiteLlm(model="groq/compound"),
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)
