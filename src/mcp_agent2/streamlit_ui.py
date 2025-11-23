import streamlit as st
from fastmcp.client import Client
import asyncio
import os # used here for environment variable example

# Define the URL of your running MCP server
# It can be a remote URL or a local one like http://localhost:8000/mcp
MCP_SERVER_URL = "http://localhost:8000/mcp"

st.title("FastMCP Client with Streamlit")

# Function to handle the asynchronous interaction
async def call_mcp_tool(tool_name, parameters):
    """
    Connects to the MCP server and calls a specific tool.
    """
    try:
        # Use a context manager for connection management
        client = Client(MCP_SERVER_URL)
        async with client:
            st.write(f"Connected to MCP server: {client.is_connected()}")
            st.write(f"Calling tool: {tool_name} with params: {parameters}")

            # Call the tool and get the result
            result = await client.call_tool(tool_name, parameters)
            st.success(f"Tool call result: {result}")
            return result
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit UI elements
tool_to_call = st.text_input("Enter tool name (e.g., 'add'):", "add")
params_input = st.text_input("Enter parameters as JSON (e.g., '{\"a\": 1, \"b\": 2}'):", '{"a": 1, "b": 2}')

if st.button("Call MCP Tool"):
    # Convert parameters from string to dictionary (handle potential errors in a real app)
    import json
    try:
        params_dict = json.loads(params_input)

        # Use asyncio.run to execute the async function within Streamlit's sync context
        asyncio.run(call_mcp_tool(tool_to_call, params_dict))

    except json.JSONDecodeError:
        st.error("Invalid JSON format for parameters.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

