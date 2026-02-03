import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from utils.prompt import wiki_rewrite_prompt
from utils.schemas import WikiListSchema

from .rewrite_example import example_7 as example

load_dotenv()

client = OpenAI(base_url=os.environ["LLM_URL"], api_key=os.environ["LLM_KEY"])
response = client.chat.completions.create(
    model=os.environ["LLM_NAME"],
    messages=[
        {
            "role": "system",
            "content": wiki_rewrite_prompt.format(
                json_schema=json.dumps(WikiListSchema.model_json_schema(), ensure_ascii=False),
                page_title=example["page_title"],
                section_title=example["section_title"],
                list_title=example["list_title"],
            ),
        },
        {"role": "user", "content": example["content"]},
    ],
    response_format={"type": "json_object"},
    temperature=1.2,
)
print(response.choices[0].message.content)
