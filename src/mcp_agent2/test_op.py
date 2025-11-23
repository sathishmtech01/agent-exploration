import streamlit as st
import subprocess
from openai import OpenAI
import sys
import threading
import time
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()
response = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "mcp",
            # "name":"greet",
            "server_label": "k",
            "server_url": "http://127.0.0.1:8000/mcp",
            "require_approval": "never",
        },
    ],
    input="hi",
)