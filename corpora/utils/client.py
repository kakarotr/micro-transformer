import os

from openai import AsyncOpenAI, OpenAI


def get_deepseek_client():
    deepseek_url = os.environ["DEEPSEEK_URL"]
    deepseek_key = os.environ["DEEPSEEK_KEY"]

    return "deepseek-chat", OpenAI(base_url=deepseek_url, api_key=deepseek_key)


def get_kimi_client():
    kimi_url = os.environ["KIMI_URL"]
    kimi_key = os.environ["KIMI_KEY"]

    return "kimi-k2.5", OpenAI(base_url=kimi_url, api_key=kimi_key)


def get_async_kimi_client():
    kimi_url = os.environ["KIMI_URL"]
    kimi_key = os.environ["KIMI_KEY"]

    return "kimi-k2-0905-preview", AsyncOpenAI(base_url=kimi_url, api_key=kimi_key)


def get_async_deepseek_client():
    deepseek_url = os.environ["DEEPSEEK_URL"]
    deepseek_key = os.environ["DEEPSEEK_KEY"]

    return "deepseek-chat", AsyncOpenAI(base_url=deepseek_url, api_key=deepseek_key)


def get_qwen_client():
    qwen_url = os.environ["QWEN_URL"]
    qwen_key = os.environ["QWEN_KEY"]

    return "qwen3.5-plus", OpenAI(base_url=qwen_url, api_key=qwen_key)


def get_openrouter_client():
    url = os.environ["OPENROUTER_URL"]
    key = os.environ["OPENROUTER_KEY"]

    return "google/gemini-3-flash-preview", OpenAI(base_url=url, api_key=key)


def get_bytedance_client():
    url = os.environ["BYTE_URL"]
    key = os.environ["BYTE_KEY"]

    return "doubao-seed-1-8-251228", OpenAI(base_url=url, api_key=key)
