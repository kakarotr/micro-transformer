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
    title: str | None = None
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

prompt_2 = """
# Role
你是一位精通日本古典文学与中国历史的翻译专家，同时也是一位由于NLP预训练数据清洗的数据工程师。你的任务是将《太平记》（Taiheiki）的古文/日文原文翻译成**现代标准汉语（简体）**。

# Goal
生成高质量、语义准确且格式严格的文本，用于构建Decoder模型的预训练数据集。

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

# Few-Shot Examples

Input:
元弘三年五月、新田義貞は兵を挙げて鎌倉に攻め入った。北条高時は東勝寺において一族と共に自害し、鎌倉幕府は滅亡した。

Output:
**1333年（元弘三年）**五月，**新田义贞**起兵攻入**镰仓**。**北条高时**在**东胜寺**与族人一同自尽，**镰仓幕府**就此灭亡。

Input:
後醍醐天皇は隠岐島を脱出し、船上山に拠点を構えた。

Output:
**后醍醐天皇**逃离**隐岐岛**，在**船上山**构筑了据点。

# Action
请根据以上规则，翻译用户提供的文本：
"""

title_prompt = """
# Role
你是一位专注于日本古典文学标题翻译的专家。你的任务是将《太平记》的章节标题（Title/Header）翻译成**现代标准汉语（简体）**。

# Goal
生成符合现代汉语阅读习惯、同时保留历史厚重感的标题，用于Decoder模型的目录或标题数据训练。

# Constraints & Rules

## 1. 标题结构处理 (Structure)
* **核心规则**：原文标题通常以“……事”结尾。译文应统一处理为“**……之事**”或“**关于……**”，视语感而定，但优先推荐“**……之事**”以保持语料库风格的统一性。
* **动词处理**：将古文动词转换为现代汉语双音节词，以增强标题的稳重感（例如：“没落”->“败逃/沦陷”，“上洛”->“进京”）。

## 2. 实体标记 (Entity Highlighting)
* **原则**：与正文处理一致，使用 Markdown 加粗符号 `**` 标记核心实体。
* **标记对象**：
    * 关键人物（如：**后醍醐天皇**、**将军**）
    * 关键地名（如：**筑紫**、**六波罗**）
* **注意**：标题很短，如果所有词都是实体，可以适当减少非必要的加粗，只突出最核心的一个或两个。

## 3. 格式规范 (Formatting)
* **无标点**：标题末尾**严禁**添加句号或其他标点符号。
* **无年号转换**：标题中极少出现年号。如果出现，保持与正文一样的“公元（年号）”格式；若无，无需强行添加时间。
* **不添加解释**：仅输出译文，不要包裹代码块，不要添加“译文：”前缀。

# Few-Shot Examples

Input:
後醍醐天皇御謀反事

Output:
**后醍醐天皇**企图谋反之事

Input:
楠廷尉京都参内事

Output:
**楠木正成**前往**京都**参拜之事

Input:
主上還御事

Output:
**天皇**回銮之事

Input:
将軍筑紫御開事

Output:
**将军**在**筑紫**开战之事

# Action
请根据以上规则，翻译用户提供的标题：
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
    "https://ja.wikisource.org/wiki/太平記/巻第一",
    "https://ja.wikisource.org/wiki/太平記/巻第二",
    "https://ja.wikisource.org/wiki/太平記/巻第三",
    "https://ja.wikisource.org/wiki/太平記/巻第四",
    "https://ja.wikisource.org/wiki/太平記/巻第五",
    "https://ja.wikisource.org/wiki/太平記/巻第六",
    "https://ja.wikisource.org/wiki/太平記/巻第七",
    "https://ja.wikisource.org/wiki/太平記/巻第八",
    "https://ja.wikisource.org/wiki/太平記/巻第九",
    "https://ja.wikisource.org/wiki/太平記/巻第十",
    "https://ja.wikisource.org/wiki/太平記/巻第十一",
    "https://ja.wikisource.org/wiki/太平記/巻第十二",
    "https://ja.wikisource.org/wiki/太平記/巻第十三",
    "https://ja.wikisource.org/wiki/太平記/巻第十四",
    "https://ja.wikisource.org/wiki/太平記/巻第十五",
    "https://ja.wikisource.org/wiki/太平記/巻第十六",
    "https://ja.wikisource.org/wiki/太平記/巻第十七",
    "https://ja.wikisource.org/wiki/太平記/巻第十八",
    "https://ja.wikisource.org/wiki/太平記/巻第十九",
    "https://ja.wikisource.org/wiki/太平記/巻第二十",
    "https://ja.wikisource.org/wiki/太平記/巻第二十一",
]


def save(title: str):
    load_dotenv()
    with get_cursor(autocommit=True) as cursor:
        sections = []
        for url in urls:
            name = url.split("/")[-1]
            print(f"开始处理 {name}")
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
                    for p in content_doc.find_all("p", recursive=False):
                        text = p.get_text(strip=True)
                        if text not in name and len(text) > 1 and not text.startswith("蒙窃採古今之変化"):
                            segment = text.split(" ")
                            is_title = False
                            try:
                                int(segment[0])
                                is_title = True
                            except:
                                pass
                            if is_title:
                                section_title = segment[-1]
                                section.paragraphs.append(Paragraph(lang="ja", title=section_title, content=""))
                            else:
                                section.paragraphs[-1].content = text
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
            "select id, raw_content from book_core_corpus where content is null and title = %s order by id limit 10",
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
        translated_titles = []
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

            # 翻译标题
            assert p.title is not None
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": title_prompt},
                    {"role": "user", "content": p.title},
                ],
                temperature=0.2,
            )
            result = response.choices[0].message.content
            assert result is not None
            translated_titles.append(result)

            # 翻译正文
            input_data = {"previous_context": context_pairs, "target_text": p.content}
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_2},
                    {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
            result = response.choices[0].message.content
            assert result is not None
            translated_paragraphs.append(result)
            progress.advance(task_id)
        new_section = BookSection(
            name=section.name,
            paragraphs=[
                Paragraph(lang="zh", title=title, content=content)
                for title, content in zip(translated_titles, translated_paragraphs)
            ],
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


rewrite(title="太平记")
