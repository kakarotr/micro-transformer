import base64
import json
import re
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from corpora.core.wiki.entities import SectionBlock, WikiPage, WikiSection
from corpora.utils.client import get_kimi_client, get_openrouter_client, get_qwen_client
from corpora.utils.db import get_cursor


class ContentBlock(BaseModel):
    type: Annotated[
        Literal["title", "paragraph"], Field(description="该文本块的类型。'title'为标题，'paragraph'为正文段落")
    ]
    content: Annotated[str, Field(description="文本内容")]
    start_with_indent: Annotated[
        bool, Field(description="（仅针对paragraph有效）首行是否有明显缩进。标题通常设为 false")
    ]


class Result(BaseModel):
    # 核心变化：不再分开 titles 和 paragraphs，而是合并为一个有序列表
    blocks: Annotated[list[ContentBlock], Field(description="页面内容块列表，必须严格保持从上到下的视觉/阅读顺序")]


prompt = f"""
# Role
你是一个书籍排版分析专家。请严格按照**阅读顺序（从上到下）**提取页面内容。

# Goal
将页面内容解析为一个有序的“内容块（Block）”列表。
**关键目标**：只提取正文和标题，**自动过滤掉图片下方的注释说明文字**。

# Workflow & Rules

1. **阅读顺序 (Sequential Scan)**:
   - 请像人类阅读一样，从页面顶部开始，一行一行向下扫描。
   - 保持原本的段落和标题顺序。

2. **排除规则 (Exclusion - Critical)**:
   - **忽略图片注释 (Skip Captions)**: 
     - 凡是出现在插图正下方、字体明显小于正文、或者采用居中对齐的说明性文字，请**直接丢弃**，不要将其作为 block 输出。
   - **忽略页眉页脚**: 页面最顶端或最底端的页码、书名标记也请忽略。

3. **类型判断 (`type`)**:
   - **标题 (`title`)**: 
     - 字体显著较大、加粗、或者独立成行且带有序号（如“第一节”、“一、”、“二、”）。
     - 例如图片中的“第一节 天皇...”、“一、大化改新...”以及页面中间的“二、义教之死”，都属于 `title`。
   - **段落 (`paragraph`)**: 
     - 仅包含标准的**正文文本**。
     - 必须是页面主体内容的一部分。

4. **缩进检测 (`start_with_indent`)**:
   - 仅对 `paragraph` 有效。
   - 仔细观察文本块的第一行：
     - **True**: 第一行左侧有明显的空白（通常是两个汉字宽的缩进）。这是判断自然段开始的重要标志。
     - **False**: 第一行与左侧边界对齐（顶格）。(注：如果这行字是居中的小字，请检查规则2，它很可能是应当被忽略的图片注释)。

# Output Format
严格遵守 JSON Schema，返回一个包含 `blocks` 的 JSON 对象。
{json.dumps(Result.model_json_schema(), ensure_ascii=False)}
"""

load_dotenv()


def ocr(name):
    images_path = Path(f"preview/pdf_images/{name}")
    output_path = Path(f"preview/jsons/{name}")
    if not output_path.exists():
        output_path.mkdir()
    files = [file for file in images_path.iterdir() if file.is_file() and file.name != ".DS_Store"]
    files.sort(key=lambda p: int(p.stem.split("_")[-1]))
    for file in files:
        if int(file.stem.split("_")[-1]) != 667:
            continue
        print(name, file.stem)
        with open(str(file.absolute()), "rb") as f:
            image_data = f.read()
            base64_bytes = base64.b64encode(image_data)
            base64_str = base64_bytes.decode("utf-8")

        model_name, client = get_openrouter_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}],
                },
            ],
            response_format={"type": "json_object"},
            temperature=1,
        )
        result = response.choices[0].message.content
        if result:
            with open(f"preview/jsons/{name}/{file.stem}.json", mode="w", encoding="utf-8") as f:
                f.write(result)


def parse_header(text):
    """
    解析标题，返回 (level, title_content)。
    如果没有匹配任何模式，返回 (None, text) 或者根据需求自定义。
    """
    # 预处理：去除首尾可能存在的空白字符，但在正则匹配时保留中间的换行
    text = text.strip()

    # 定义正则逻辑
    # re.DOTALL (re.S) 让 '.' 也可以匹配换行符，防止标题内容本身包含换行
    # [零一二三四五六七八九十百千]+ 用于匹配中文数字
    # \s* 用于吃掉前缀后的空格、换行符等

    patterns = [
        # 模式 1: 第一章 XXX -> Level 2
        # 匹配以 "第" 开头，中间是中文数字，以 "章" 结尾
        (r"^第[零一二三四五六七八九十百千\d]+章\s*(.*)", 2),
        # 模式 2: 第一节\nXXXX -> Level 3
        # 匹配以 "第" 开头，中间是中文数字，以 "节" 结尾
        (r"^第[零一二三四五六七八九十百千\d]+节\s*(.*)", 3),
        # 模式 3: 一、XXXX -> Level 4
        # 匹配以中文数字开头，紧跟着顿号 "、"
        (r"^[零一二三四五六七八九十百千\d]+、\s*(.*)", 4),
    ]

    for pattern, level in patterns:
        # 使用 re.DOTALL 模式以支持跨行匹配
        match = re.match(pattern, text, re.DOTALL)
        if match:
            # group(1) 是去掉前缀后的实际标题内容
            title_content = match.group(1).strip()
            return level, title_content

    # 如果都不匹配，默认处理（例如返回 level 0 或者 None）
    return 0, text


def merge(name):
    jsons_path = Path(f"preview/jsons/{name}")
    files = [file for file in jsons_path.iterdir() if file.is_file() and file.name != ".DS_Store"]
    files.sort(key=lambda p: int(p.stem.split("_")[-1]))
    pages: list[WikiPage] = []
    for file in files:
        print(file.stem)
        with open(file, mode="r", encoding="utf-8") as f:
            result = Result.model_validate_json(f.read())
            for block in result.blocks:
                if block.type == "title":
                    if " | " in block.content:
                        continue
                    # pages[-1].sections.append(WikiSection(title=block.content, level=2, blocks=[]))
                    level, title = parse_header(text=block.content)
                    if level == 2:
                        pages.append(
                            WikiPage(
                                title=name,
                                category_name="",
                                lang="zh",
                                sections=[WikiSection(title=title, level=level, blocks=[])],
                            )
                        )
                    else:
                        pages[-1].sections.append(WikiSection(title=title, level=3, blocks=[]))
                elif block.type == "paragraph":
                    content = block.content
                    if block.start_with_indent:
                        pages[-1].sections[-1].blocks.append(SectionBlock(type="text", content=content))
                    else:
                        pages[-1].sections[-1].blocks[-1].content += block.content

    with get_cursor() as cursor:
        for idx, page in enumerate(pages, start=1):
            with open(f"preview/markdown/{name}_{idx}.md", mode="w", encoding="utf-8") as f:
                f.write(page.merge_sections())
            cursor.execute(
                "insert into book_core_corpus (title, raw_content, content) values (%s, %s, %s)",
                (name, page.model_dump_json(), page.model_dump_json()),
            )


def a():
    prefix = "preview/pdf_images/日本战国风云录（上）/page_"
    test_images = [
        f"{prefix}28.png",
    ]

    for idx, item in enumerate(test_images, start=1):
        with open(item, "rb") as f:
            print(item)
            image_data = f.read()
            base64_bytes = base64.b64encode(image_data)
            base64_str = base64_bytes.decode("utf-8")

            model_name, client = get_openrouter_client()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}],
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                extra_body={
                    "provider": {
                        "order": ["deepinfra", "together"],
                        "allow_fallbacks": True,
                    }
                },
            )
            result = response.choices[0].message.content
            if result:
                with open(f"preview/jsons/{idx}.json", mode="w", encoding="utf-8") as f:
                    f.write(result)


for name in [
    "上杉谦信传",
]:
    print(name)
    merge(name=name)
    # ocr(name=name)
