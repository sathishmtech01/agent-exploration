# code_extractor/agent.py
import logging
import os

from dotenv import load_dotenv
import os
from google.adk.agents import Agent,SequentialAgent,LlmAgent
# from google.adk.io import file_to_str
from pydantic import BaseModel, Field
from google.adk.models.lite_llm import LiteLlm
from .codebase_agent import generate_codebase
from google.adk.tools import FunctionTool
openai_llm = LiteLlm(
    model="gpt-4o"
)
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

load_dotenv()



# Define the main agent
root_agent1 = LlmAgent(
    name="Code_Gen",
    model=openai_llm, # Use the model configured during 'adk create'
    description="An agent that uses structured output to decode unstructured LLM responses into code, path, and filename.",
    instruction="""
    You are a code-generation agent.
    Return ONLY valid JSON describing a project codebase.
    
    FORMAT:
    {
      "files": [
        { "path": "path/to/file", "content": "file contents" }
      ]
    }
    """,
    output_key="generated_code"
    # tools=[generate_codebase] # Register the tool
)

root_agent2 = LlmAgent(
    name="Code_Save",
    model=openai_llm, # Use the model configured during 'adk create'
    description="saving the code",
    instruction="""
    use 
    {generated_code} 
    and 
    call generate_codebase
    """,
    tools=[generate_codebase] # Register the tool
)


root_agent = SequentialAgent(
    name="Code_Extractor",
    # model=openai_llm, # Use the model configured during 'adk create'
    description="saving the code",
    sub_agents=[root_agent1, root_agent2]
    # tools=[generate_codebase] # Register the tool
)

# Optional: Add an example to the prompt for better reliability (optional but recommended)
# example_response = """
# Here is the code you requested.
# The file should be located in 'src/components/'.
# Filename: 'MyComponent.jsx'
#
# ```javascript
# import React from 'react';
# function MyComponent() {
#   return <div>Hello World</div>;
# }
# export default MyComponent;
