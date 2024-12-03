import chainlit as cl
import openai
import os
from dotenv import load_dotenv


load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


endpoint_url = "https://api.openai.com/v1"
print(api_key)
client = openai.AsyncClient(api_key=api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
    "model": "chatgpt-4o-latest",
    "temperature": 0.3,
    "max_tokens": 500
}

# api_key = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("RUNPOD_API_KEY")
runpod_serverless_id = os.getenv("RUNPOD_SERVERLESS_ID")

# endpoint_url = "https://api.openai.com/v1"
endpoint_url = f"https://api.runpod.ai/v2/{runpod_serverless_id}/openai/v1"

client = openai.AsyncClient(api_key=api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o
# model_kwargs = {
#     "model": "chatgpt-4o-latest",
#     "temperature": 1.2,
#     "max_tokens": 500
# }

model_kwargs = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "temperature": 0.3,
    "max_tokens": 500
}