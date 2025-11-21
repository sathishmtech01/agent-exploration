# agent.py
from google.adk.agents import LlmAgent
from .tools import *
from google.adk.models.lite_llm import LiteLlm

openai_llm = LiteLlm(
    model="gpt-4o"
)
# generic_web_agent.py (continued)
INSTRUCTION = """
You are a generalized web automation agent. You can control a browser to navigate, click, type, and extract information. 
Your goal is to fulfill user requests using the available tools. 
You must decide the correct sequence of actions (start, navigate, interact, quit).
When interacting with elements, you must determine the appropriate XPath selector to use.
Always quit the browser when the task is complete.
"""

root_agent = LlmAgent(
    name="GeneralizedWebAgent",
    model="gemini-2.5-flash",
    description="A flexible agent that performs various web tasks using Selenium tools.",
    instruction=INSTRUCTION,
    tools=[start_browser_adk, navigate_adk, click_adk, type_adk, extract_adk, quit_browser_adk]
)

__all__ = ["root_agent"]


