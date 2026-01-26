import os

from openai import OpenAI

deepseek_url = os.environ["DEEPSEEK_URL"]
deepseek_key = os.environ["DEEPSEEK_KEY"]

deepseek_client = OpenAI(base_url=deepseek_url, api_key=deepseek_key)
