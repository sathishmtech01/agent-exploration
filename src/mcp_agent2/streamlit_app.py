import streamlit as st
import subprocess
from openai import OpenAI
import sys
import threading
import time
from dotenv import load_dotenv
import os

load_dotenv()
st.title("FastMCP + LLM Streamlit Demo")

# ----------------------------
# 1. Start the MCP server
# ----------------------------
@st.cache_resource
def start_mcp_server():
    # Run server as a background process
    proc = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    time.sleep(1)  # give server time to start
    return proc

server = start_mcp_server()

st.success("FastMCP server is running!")

# ----------------------------
# 2. LLM Setup
# ----------------------------
client = OpenAI()


user_input = st.text_input("Ask the LLM (it will auto-call the MCP greet tool)", "Say hello to Alice")

if st.button("Run"):
    with st.spinner("Calling LLM + MCP tool…"):
        response = client.responses.create(
            model="gpt-4.1",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "hello",
                    "server_url": "http://127.0.0.1:8000/mcp1",
                    "require_approval": "never",
                },
            ],
            input=user_input,
        )

        st.subheader("LLM Response")
        st.write(response.output_text)

    st.success("Done!")
