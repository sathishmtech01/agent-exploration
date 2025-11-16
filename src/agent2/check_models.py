import requests
import os
from dotenv import load_dotenv
api_key = os.environ.get("GROQ_API_KEY") # Ensure this env var is set
print(api_key)
url = "https://api.groq.com/openai/v1/models"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
print(response.json())


#
# curl https://api.groq.com/openai/v1/chat/completions -s \
#              -H "Content-Type: application/json" \
#                 -H "Authorization: Bearer $GROQ_API_KEY" \
#                    -d '{
# "model": "llama-3.3-70b-versatile",
# "messages": [{
#     "role": "user",
#     "content": "Explain the importance of fast language models"
# }]
# }'
