import requests
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") # Ensure this env var is set
print(api_key)
url = "https://api.groq.com/openai/v1/models"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
print(response.json())


import litellm
import os

# ... (ensure API key is set) ...

response_stream = litellm.completion(
    model="groq/llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Write a short poem about the speed of Groq."}
    ],
    stream=True
)

for chunk in response_stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()
