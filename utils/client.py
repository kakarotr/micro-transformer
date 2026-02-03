import os

from openai import OpenAI


def get_deepseek_client():
    deepseek_url = os.environ["DEEPSEEK_URL"]
    deepseek_key = os.environ["DEEPSEEK_KEY"]

    return "deepseek-chat", OpenAI(base_url=deepseek_url, api_key=deepseek_key)


def get_kimi_client():
    kimi_url = os.environ["KIMI_URL"]
    kimi_key = os.environ["KIMI_KEY"]

    return "kimi-k2-0905-preview", OpenAI(base_url=kimi_url, api_key=kimi_key)
