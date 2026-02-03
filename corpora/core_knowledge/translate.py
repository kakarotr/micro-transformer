import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field, TypeAdapter

from corpora.core_knowledge.wiki.entities import WikiSection
from corpora.core_knowledge.wiki.utils import get_chunks
from utils.client import get_deepseek_client, get_kimi_client
from utils.db import get_db_conn

load_dotenv()


class Result(BaseModel):
    text: Annotated[
        str | list[str],
        Field(
            ...,
            description="The translated Simplified Chinese content based on Japanese Wikipedia input.\n"
            "TYPE SELECTION RULES:\n"
            "1. Return `list[str]` IF the input is a Markdown list:\n"
            "   - Unordered List: Starts with `-` or `*`. (Action: Strip the leading bullet symbol)\n"
            "   - Ordered List: Starts with numbers like `1.`, `2.`, etc. (Action: Strip the leading number and dot, e.g., convert '1. Content' to 'Content')\n"
            "2. Return `str` IF the input is a narrative paragraph, header, or table.\n"
            "   - Always preserve internal Markdown formatting (bold `**`, links `[]`) within the text.",
        ),
    ]


paragraph_prompt = f"""
# Role
你是一位精通日本战国史（Sengoku period）的资深历史学家和专业翻译家。你的任务是将日本维基百科的文本翻译成流畅、准确且学术规范的简体中文。

# Constraints
1. **格式与结构保留（最高优先级）**：
   - **必须严格保留**原文的 Markdown 结构符号，包括标题（#、##、###）、加粗（**...**）、列表符号（- 或 *）以及引用块（>）。
   - **必须保留**段落之间的换行符（\n\n）。不要合并段落。
   - 标题行不要添加任何额外的标点符号。

2. **日语读音与特殊格式处理**：
   - **移除假名注音**：当文中出现“汉字（假名）”的格式时（如 `下克上（げこくじょう）`），请直接删除括号及内部的假名，仅保留汉字（即译为 `下克上`）。
   - **保留关键信息**：如果括号内是**年份、别名或补充说明**（而非纯粹的日语读音），请予以翻译并保留。例如 `本能寺の変（1582年）` 应保留为 `本能寺之变（1582年）`。

3. **历史术语准确性**：
   - 官职、人名、地名、法号必须符合中文历史学界的通用译法。
   - 专有名词（如“石高”、“乐市乐座”、“检地”）直接保留汉字，不要意译。

4. **文风要求**：
   - 译文需为客观、中立的百科全书式简体中文。
   - 在保留 Markdown 列表格式（-）的同时，尽量让列表项内的文字通顺连贯。

# Output Schema
输出必须是一个 JSON 对象，下面是JSON SCHEMA：
{json.dumps(Result.model_json_schema(), ensure_ascii=False)}
"""

title_prompt = f"""
# Role
你是一位专注于日本战国史（Sengoku period）的百科全书编辑。你的任务是将日语维基百科的“条目标题”或“章节标题”翻译成标准的简体中文。

# Constraints
1. **术语标准化（最重要）**：
   - **战役名**：统一翻译为“……之战”。例如 `関ヶ原の戦い` -> `关原之战`（不要用“战役”或“合战”）。
   - **事件名**：例如 `本能寺の変` -> `本能寺之变`。
   - **政策名**：例如 `刀狩り` -> `刀狩`（去掉日语词尾的假名，保留汉字核心）。
   - **人名/地名**：严格转换为简体中文通用写法。例如 `沢` -> `泽`，`関` -> `关`。

2. **符号清洗**：
   - 移除日文中常见的间隔号（・）。如果是并列名词，根据情况改为顿号（、）或直接连接；如果是复合词（如`楽市・楽座`），请翻译为`乐市乐座`。
   - 移除日语特有的角标或消歧义后缀，除非它对区分实体至关重要。
     - 例如 `織田信長 (戦国武将)` -> `织田信长` (去掉显然的消歧义)。
     - 但 `安土城 (近江国)` -> `安土城 (近江国)` (保留地理消歧义，并翻译括号内内容)。

3. **禁止废话**：
   - 绝不要输出“翻译如下”等前缀。
   - 绝不要在标题末尾添加句号。

# Output Schema
输出必须是一个 JSON 对象，下面是JSON SCHEMA：
{json.dumps(Result.model_json_schema(), ensure_ascii=False)}
"""

# model_name, client = get_deepseek_client()
# response = client.chat.completions.create(
#     model=model_name,
#     messages=[
#         {
#             "role": "system",
#             "content": title_prompt,
#         },
#         {
#             "role": "user",
#             "content": """
# 略歴
# """,
#         },
#     ],
#     temperature=1,
#     response_format={"type": "json_object"},
# )
# print(response.choices[0].message.content)


def translate_title():
    model_name, client = get_kimi_client()


def translate(n_threads: int):
    chunks = get_chunks(
        sql="select title, sections from wiki_pages where sections is not null and title = '織田信長'",
        n_threads=n_threads,
    )

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(start_translate, chunks)


def start_translate(chunk):
    conn = get_db_conn()
    cursor = conn.cursor()
    adapter = TypeAdapter(list[WikiSection])
    for item in chunk:
        try:
            sections = adapter.validate_python(item[1])
            for section in sections:
                title = section.title

        except:
            error_stack = traceback.format_exc()
            print(f"{item[0]}错误: err: {error_stack}")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    translate(n_threads=1)
