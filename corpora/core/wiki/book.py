import asyncio
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests
import zhconv
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from corpora.utils.client import get_async_deepseek_client, get_deepseek_client
from corpora.utils.db import get_cursor


class Paragraph(BaseModel):
    lang: str
    content: str


class BookSection(BaseModel):
    name: str
    paragraphs: list[Paragraph]


class Book(BaseModel):
    title: str
    section: BookSection


prompt = """
# Role
你是一位历史类非虚构作家，正在重写《日本外史》的现代文版本。

# Task
将《日本外史》的原文改写为连贯、详实的现代汉语历史叙事文本。

# Input Data
你将收到两部分输入：
1. **previous_context (前情提要，可能为空)：** 之前段落的“原文-译文”对照列表，用于保持上下文连贯和术语一致（**这部分内容不需要翻译**）。
2. **target_text (待翻译文本)：** 需要你翻译并清洗的核心内容。

# Guidelines (Critical for Model Quality)
1. **时间标准化（Time Normalization）【最高优先级】：**
    * 原文中出现的所有年号，**必须**在译文中转换为“公元年份（年号）”的格式。
    * **隐式时间补全：** 如果原文仅说“三年春”，必须结合前文补全为“1557年（弘治三年）春”。
2. **格式强制（Single Paragraph）【修改点】：**
    * 输出必须是**唯一的、连续的一个段落**。
    * **严禁**在文本中间使用换行符（\n）、段落分隔或列表。
    * 无论原文有多长，都必须将其融合成一段流畅的文字。
3. **加粗去重（First Mention Only）【修改点】：**
    * **仅在实体第一次出现时**使用 Markdown 加粗（**...**）。
    * **后续重复提及**该实体时，**不要**再次加粗。
    * **加粗对象：** 首次出现的人名、地名、特有官职、关键战役名。
    * **时间处理：** 仅加粗关键的时间节点（如：**1184年（寿永三年）五月**），文中普通的“次日”、“随后”不要加粗。
4. **指代清晰：** 原文中省略的主语必须补全。模型无法处理“遂攻之”这样的模糊指代，必须改写为“**织田信长**于是进攻了**美浓国**”。
5. **事实密度：** 保留原文所有的历史事实细节，不要遗漏。
6. **逻辑重组：** 按照正常的时间逻辑组织段落，修正原文的倒叙或插叙。
7. **客观陈述：** 去掉“外史氏曰”等评论，仅保留叙事。

# Output
直接输出重写之后现代文本
"""


def parse(title: str, url: str):
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        },
    )
    soup = BeautifulSoup(response.text, "html.parser")
    content_doc = soup.find("div", class_="mw-parser-output")
    if content_doc:
        paragraphs = content_doc.find_all("p", recursive=False)
        total = len(paragraphs)
        for idx, p in enumerate(paragraphs):
            if idx == 0 or (idx == total - 1):
                continue
            print(p.get_text(strip=True))


def get_url(url: str):
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        },
    )
    if response.ok:
        doc = BeautifulSoup(response.text, "html.parser")
        content_doc = doc.find("div", class_="mw-content-ltr mw-parser-output")
        if content_doc:
            urls = []
            for li in content_doc.find_all("li"):
                urls.append(
                    (
                        zhconv.convert(li.get_text(strip=False), "zh-hans").split("　")[-1],
                        f"https://zh.wikisource.org{li.find('a').attrs.get('href')}",  # type: ignore
                    )
                )


get_url(url="https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2")

urls = [
    ("平氏", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E4%B8%80"),
    ("源氏上", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E4%BA%8C"),
    ("源氏下", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E4%B8%89"),
    ("北条氏", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%9B%9B"),
    ("楠氏", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E4%BA%94"),
    ("新田氏", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%85%AD"),
    ("足利氏上", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E4%B8%83"),
    ("足利氏中", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%85%AB"),
    ("足利氏下", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E4%B9%9D"),
    ("后北条氏", "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81"),
    (
        "武田氏上杉氏",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81%E4%B8%80",
    ),
    (
        "毛利氏",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81%E4%BA%8C",
    ),
    (
        "织田氏上",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81%E4%B8%89",
    ),
    (
        "织田氏下",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81%E5%9B%9B",
    ),
    (
        "丰臣氏上",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81%E4%BA%94",
    ),
    (
        "丰臣氏中",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81%E5%85%AD",
    ),
    (
        "丰臣氏下",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81%E4%B8%83",
    ),
    (
        "德川氏一",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81%E5%85%AB",
    ),
    (
        "德川氏二",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%8D%81%E4%B9%9D",
    ),
    (
        "德川氏三",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E4%BA%8C%E5%8D%81",
    ),
    (
        "德川氏四",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%BB%BF%E4%B8%80",
    ),
    (
        "德川氏五",
        "https://zh.wikisource.org/wiki/%E6%97%A5%E6%9C%AC%E5%A4%96%E5%8F%B2/%E5%8D%B7%E4%B9%8B%E5%BB%BF%E4%BA%8C",
    ),
]


def save(title: str):
    load_dotenv()
    with get_cursor(autocommit=True) as cursor:
        sections = []
        for name, url in urls:
            section = BookSection(name=name, paragraphs=[])
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
                },
            )
            if response.ok:
                doc = BeautifulSoup(response.text, "html.parser")
                content_doc = doc.find("div", class_="mw-content-ltr mw-parser-output")
                if content_doc:
                    for p in content_doc.find_all("p"):
                        text = p.get_text(strip=True)
                        if not text.startswith("日本外史"):
                            section.paragraphs.append(Paragraph(lang="tw", content=text))
                    sections.append(section)
            time.sleep(2)
        for section in sections:
            cursor.execute(
                "insert into book_core_corpus (title, raw_content) values (%s, %s)", (title, section.model_dump_json())
            )


def rewrite(title: str):
    load_dotenv()
    with get_cursor() as cursor:
        cursor.execute(
            "select id, raw_content from book_core_corpus where content is null and title = %s order by id",
            (title,),
        )
        rows = cursor.fetchall()
    console = Console()
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[section_name]}", justify="right"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    with Progress(*progress_columns, console=console) as progress:
        max_workers = len(rows) if len(rows) > 0 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in rows:
                row_id, raw_content = row
                task_id = progress.add_task(
                    "waiting",
                    total=None,  # total 暂时未知，将在子线程中解析后设置
                    start=False,  # 暂时不开始计时
                    section_name=f"ID: {row_id}",
                )

                futures.append(executor.submit(process_rewrite, row_id, raw_content, progress, task_id))

            for future in futures:
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


def process_rewrite(id, raw_content, progress: Progress, task_id):
    try:
        section = BookSection.model_validate(raw_content)
        translated_paragraphs = []
        original_paragraphs = [p.content for p in section.paragraphs]

        total_paragraphs = len(original_paragraphs)
        progress.update(
            task_id,
            total=total_paragraphs,
            section_name=f"Section: {section.name[:15]}...",
        )
        progress.start_task(task_id)

        model_name, client = get_deepseek_client()
        for idx, p in enumerate(section.paragraphs):
            start_index = max(0, idx - 2)
            context_pairs = []
            for i in range(start_index, idx):
                context_pairs.append({"original": original_paragraphs[i], "translation": translated_paragraphs[i]})
            input_data = {"previous_context": context_pairs, "target_text": p.content}
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
            result = response.choices[0].message.content
            assert result is not None
            translated_paragraphs.append(result)
            progress.advance(task_id)
        new_section = BookSection(
            name=section.name, paragraphs=[Paragraph(lang="zh", content=item) for item in translated_paragraphs]
        )
        with get_cursor() as cursor:
            cursor.execute(
                "update book_core_corpus set content = %s where id = %s", (new_section.model_dump_json(), id)
            )
        progress.update(task_id, completed=total_paragraphs)
    except:
        traceback.print_exc()
        progress.update(task_id, description="[red]Error[/]", section_name=f"ID: {id} Failed")
        progress.stop_task(task_id)


rewrite(title="日本外史")
