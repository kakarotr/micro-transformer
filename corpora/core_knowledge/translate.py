import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, TypeAdapter
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from corpora.core_knowledge.wiki.entities import WikiSection
from corpora.core_knowledge.wiki.utils import get_chunks
from utils.client import get_async_kimi_client
from utils.db import get_db_conn

load_dotenv()
global_sem = asyncio.Semaphore(100)


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
   - **必须严格保留**原文的 Markdown 结构符号，包括标题（#、##、###）、加粗（**...**）。
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
#             "content": paragraph_prompt,
#         },
#         {
#             "role": "user",
#             "content": """
# - 天正元年（1572年）冬、陸奥の伊達輝宗から鷹が献上され、信長は伊達氏の分国を「直風」にした。他の奥羽の領主たちも鷹や馬を献上した。,
# - 天正4年（1576年）4月、毛利輝元の叔父・小早川隆景が信長に太刀、馬、銀子1,000枚を献上し、信長は羽柴秀吉を介して謝意を伝えた。,
# - 天正8年（1580年）3月9日、北条氏政は使者を上洛させ、信長に鷹13羽、馬5頭を献上し、北条分国を信長に進上した。,
# - 天正8年（1580年）6月26日、長宗我部元親が鷹16羽を信長に献上した。
# """,
#         },
#     ],
#     temperature=1,
#     response_format={"type": "json_object"},
# )
# print(response.choices[0].message.content)


def translate(n_threads: int):
    chunks = get_chunks(
        sql="select title, sections, id from wiki_pages where sections is not null and lang = 'ja' and title not in ('北庵法印', '石川久智', '太田牛一', '蒲生貞秀', '宮兼信', '石川伊予守', '坂崎直盛', '後醍院宗重', '結城秀康', '前野景定', '生石治家', '斯波義重', '深谷吉次', '松山城風流合戦', '鎌倉幕府', '武士道', '小田原征伐', '畠山宣意', '市川等長', '陰徳太平記', '鶴岡八幡宮の戦い', '豊臣秀頼')",
        n_threads=n_threads,
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        total_items = sum(len(c) for c in chunks)
        overall_task = progress.add_task("[yellow]总进度", total=total_items)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                # 为每个线程创建一个独立的子进度条
                task_id = progress.add_task(f"[cyan]线程-{i + 1}", total=len(chunk))
                futures.append(executor.submit(start_translate, chunk, progress, task_id, overall_task))

            # 等待所有线程完成
            for future in futures:
                future.result()


def start_translate(chunk, progress, task_id, overall_task):
    if not chunk:
        return
    return asyncio.run(thread_coroutine_manager(chunk, progress, task_id, overall_task))


async def thread_coroutine_manager(chunk, progress, task_id, overall_task):
    results = await process_row(chunk, progress, task_id, overall_task)
    if results:
        conn = get_db_conn()
        cursor = conn.cursor()
        for item in results:
            if item:
                id, data = item
                cursor.execute("update wiki_pages set sections = %s, lang='zh' where id = %s", (data, id))
        conn.commit()
        cursor.close()
        conn.close()
        progress.update(task_id, description=f"[green]线程已完成[/green]")


async def process_row(chunk, progress, task_id, overall_task):
    results = []
    for row in chunk:
        page_title, sections, id = row
        adapter = TypeAdapter(list[WikiSection])
        try:
            sections = adapter.validate_python(sections)
            for section in sections:
                tasks = []
                display_title = (page_title[:12] + "...") if len(page_title) > 12 else page_title.ljust(15)
                progress.update(task_id, description=f"正在翻译: {display_title}")
                # 翻译标题
                title = section.title
                tasks.append(("title", section, llm_translate(type="title", text=title)))
                # 翻译内容
                for block in section.blocks:
                    if block.lang != "zh" and block.content:
                        if isinstance(block.content, str):
                            text = block.content
                        else:
                            text = "\n".join([f"- {item}" for item in block.content])
                        tasks.append(("block", block, llm_translate(type="paragraph", text=text)))
                if tasks:
                    task_results = await asyncio.gather(*(t[2] for t in tasks))
                    for (task_type, target_obj, _), result in zip(tasks, task_results):
                        if task_type == "title":
                            target_obj.title = result.text
                        else:
                            target_obj.lang = "zh"
                            target_obj.content = result.text
            # with open("preview/zh.md", mode="w", encoding="utf-8") as f:
            #     f.write(WikiPage(title=page_title, category_name="", lang="", sections=sections).merge_sections())
            results.append((id, adapter.dump_json(sections).decode()))
        except:
            error_stack = traceback.format_exc()
            print(f"{page_title}错误: err: {error_stack}")
        finally:
            progress.update(task_id, advance=1)
            progress.update(overall_task, advance=1)
    return results


async def llm_translate(type: Literal["title", "paragraph"], text: str):
    async with global_sem:
        model_name, client = get_async_kimi_client()
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": title_prompt if type == "title" else paragraph_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        assert result is not None
        return Result.model_validate_json(result)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    translate(n_threads=5)
