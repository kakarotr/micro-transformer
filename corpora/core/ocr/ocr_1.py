import array
import base64
import json
import re
from pathlib import Path
from typing import Annotated, Literal

import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from corpora.core.wiki.entities import SectionBlock, WikiPage, WikiSection
from corpora.utils.client import get_qwen_client
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
   - **剔除行内注释序号 (Clean Inline Citations)**: 
     - 在提取 `paragraph` 或 `title` 的具体文字时，如果遇到穿插在正文句子中的上标数字、带有圆圈的脚注/尾注序号（如 ①、②、③ 等），请**直接过滤掉**。
     - 绝对不要将这些符号输出到最终提取的文本中，**更不要**因为这些符号的存在而将一个完整连贯的自然段强行拆分成多个不同的 block。确保提取出的文本是纯净且语义连续的。

3. **类型判断 (`type`)**:
   - **全局格式化：中文年份转换 (Chinese Year Conversion)**: 在提取任何文本（标题或正文）时，如果遇到使用中文数字表示的年份（例如“一五八二”、“二〇〇六”、“一九九九”等），**必须**将其强制转换为对应的阿拉伯数字（如“1582”、“2006”、“1999”）。保留原有的“年”字等前后文。
   - **标题 (`title`)**: 
     - 字体显著较大、加粗、或者独立成行且带有序号（如“第一节”、“一、”、“二、”）。
     - 例如图片中的“第一节 天皇...”、“一、大化改新...”以及页面中间的“二、义教之死”，都属于 `title`。
     - **跨行标题合并 (Force Title Merging)**: 如果遇到标题内容被排版在多行（例如章号和章名连续出现，如“第1章”换行后是“皇统分立”），**必须**将其合并为一个完整的标题字符串，中间用一个空格分隔。例如，应当提取为 `"第1章 皇统分立"`。
   - **段落 (`paragraph`)**: 
     - 仅包含标准的**正文文本**。
     - 必须是页面主体内容的一部分。

4. **严格缩进检测 (`start_with_indent` - Critical)**:
   - 仅对 `paragraph` 类型的 block 有效。
   - **核心判断标准（相对位置对比）**：请不要依赖视觉上的绝对距离，而是严格对比当前段落**第一行的首字符**与**第二行的首字符**的垂直对齐关系：
     - **True (有缩进)**：第一行的起始位置明显位于第二行起始位置的**右侧**。
     - **False (无缩进/顶格)**：第一行的起始位置与第二行左侧**严格平齐**。
   - **单行段落的判断（全局基准线）**：如果该段落只有一行，请寻找页面的**左侧正文边界基准线**（其他顶格段落或正文的最左侧边缘）。如果该行向右偏离基准线，则为 True；如果紧贴基准线，则为 False。
   - **特殊情况与抗干扰（Edge Cases）**：
     - **居中并非缩进**：如果一行文字处于页面正中间，左右两侧都有大量留白，说明它是居中对齐，应判定为 False（并请复核规则2，这极有可能是应丢弃的图片注释或副标题）。
     - **忽略标点符号偏移**：如果段首是前引号（如 `“` 或 `「`），请忽略引号本身由于字体排版造成的微小留白，主要观察**第一个实际汉字**的位置是否缩进了常规宽度。

5. **列表合并与强制格式化 (List Merging & Forced Markdown)**: 
  - 如果在正文中遇到连续多行的列表结构，**绝不能**单独拆分 block，必须将整个列表合并提取为一个完整的 `type: "paragraph"`。
  - **强制 Markdown 转换 (极其重要)**：你必须将原文的列表序号强制转换为标准的 Markdown 格式（有序使用 `1. `, `2. `，无序使用 `- `）。
  - **中文序号替换规则**：如果原文使用的是中文序号（例如图中的“第一，”、“第二，”，或者“一、”、“（一）”等），你必须**删除这些原文字符**，并严格替换为对应的阿拉伯数字 Markdown 序号（如 `1. `、`2. `）。绝对不要在最终的字符串中保留原文的中文序号字眼。
  - **单字符串输出**：列表的所有项必须包含在同一个字符串内，各项之间使用换行符（在 JSON 中转义为 `\n`）进行分隔。

# Output Format
严格遵守 JSON Schema，返回一个包含 `blocks` 的 JSON 对象。
{json.dumps(Result.model_json_schema(), ensure_ascii=False)}
"""

load_dotenv()


def test():
    model_name, client = get_qwen_client()
    image = Path("preview/pdf_images/早稻田大学日本史（安土桃山时代）/page_1.png")
    with open(image, mode="rb") as f:
        base64_bytes = base64.b64encode(f.read())
        base64_str = base64_bytes.decode("utf-8")
    response = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    result = response.choices[0].message.content
    assert result is not None
    print(result)


@click.command()
@click.argument("name")
@click.option("--start", help="开始的页数")
def ocr(name, start):
    images_path = Path(f"preview/pdf_images/{name}")
    output_path = Path(f"preview/jsons/{name}")
    if not output_path.exists():
        output_path.mkdir()
    files = [file for file in images_path.iterdir() if file.is_file() and file.name != ".DS_Store"]
    files.sort(key=lambda p: int(p.stem.split("_")[-1]))
    start = int(start)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[yellow]当前文件: {task.fields[item_name]}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        if start != 0:
            files_no = [int(file.stem.split("_")[-1]) for file in files]
            index = [idx for idx, no in enumerate(files_no) if no == int(start)]
            count = len(files[index[0] :])
        else:
            count = len(files)
        task_id = progress.add_task(description=f"识别{name}", total=count, item_name="...")

        for file in files:
            progress.update(task_id, item_name=file.stem)
            if start and int(file.stem.split("_")[-1]) < int(start):
                continue
            with open(str(file.absolute()), "rb") as f:
                image_data = f.read()
                base64_bytes = base64.b64encode(image_data)
                base64_str = base64_bytes.decode("utf-8")

            model_name, client = get_qwen_client()
            response = client.chat.completions.create(
                model="qwen3.5-flash",
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}],
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            result = response.choices[0].message.content
            assert result is not None
            result = clean_llm_json(result)
            # print(result)
            # break
            # result = Result.model_validate_json(result.replace("\n", ""))
            if result:
                with open(f"preview/jsons/{name}/{file.stem}.json", mode="w", encoding="utf-8") as f:
                    f.write(result)

            progress.update(task_id, advance=1)


def clean_llm_json(broken_json_str):
    no_newline_str = re.sub(r"[\r\n]+", " ", broken_json_str)
    pattern = re.compile(r'("(?:\\.|[^"\\])*")|\s+')
    minified_str = pattern.sub(lambda m: m.group(1) if m.group(1) else "", no_newline_str)
    return minified_str


def parse_header(text):
    """
    解析标题，返回 (level, title_content)。
    如果没有匹配任何模式，返回 (None, text) 或者根据需求自定义。
    """
    # patterns = [
    #     (r"^[零一二三四五六七八九十百千\d]+章\s*(.*)", 2),
    #     (r"^第[零一二三四五六七八九十百千\d]+节\s*(.*)", 3),
    #     (r"^\d+[ \t·\s]+(.+)$", 3),
    #     (r"^[零一二三四五六七八九十百千\d]+、\s*(.*)", 4),
    # ]
    if text == "序言" or text == "概要":
        return 2, text

    # patterns = [
    #     (r"^第[零一二三四五六七八九十百千\d]+章\s*(.*)", 2),
    #     (r"^[零一二三四五六七八九十百千\d]+、\s*(.*)", 3),
    #     (r"^（[零一二三四五六七八九十百千\d]+）\s*(.*)", 4),
    # ]

    patterns = [
        (r"^第[零一二三四五六七八九十百千\d]+章\s*(.*)", 2),
        (r"^第[零一二三四五六七八九十百千\d]+节\s*(.*)", 3),
        (r"^(.+)$", 4),
    ]

    # patterns = [
    #     (r"^第[零一二三四五六七八九十百千\d]+章\s*(.*)", 2),
    #     (r"^(.+)$", 3),
    # ]

    # patterns = [
    #     (r"^第[零一二三四五六七八九十百千\d]+部\s*(.*)", 2),
    #     (r"^第\s*[零一二三四五六七八九十百千\d]+\s*章\s*(.*)", 3),
    #     (r"^(.+)$", 4),
    # ]

    # patterns = [
    #     (r"^第[零一二三四五六七八九十百千\d]+章\s*(.*)", 2),
    #     (r"^\d+\.\s*(.+)", 3),
    #     (r"^(.+)$", 4),
    # ]

    for pattern, level in patterns:
        match = re.match(pattern, text, re.DOTALL)
        if match:
            title_content = match.group(1).strip()
            return level, title_content
    return 0, text


def merge(name):
    jsons_path = Path(f"preview/jsons/{name}")
    files = [file for file in jsons_path.iterdir() if file.is_file() and file.name != ".DS_Store"]
    files.sort(key=lambda p: int(p.stem.split("_")[-1]))
    pages: list[WikiPage] = []
    for file in files:
        print(file.stem)
        with open(file, mode="r", encoding="utf-8") as f:
            try:
                content = f.read()
                result = Result.model_validate_json(content)
            except:
                data = json.loads(content)
                if isinstance(data, list):
                    if data[0].get("blocks"):
                        result = Result.model_validate(data[0])
                    else:
                        result = Result.model_validate({"blocks": data})
            for idx, block in enumerate(result.blocks):
                if block.type == "title":
                    if " | " in block.content:
                        continue
                    # pages[-1].sections.append(WikiSection(title=block.content, level=2, blocks=[]))
                    level, title = parse_header(text=block.content)
                    if title.startswith("·"):
                        title = title[1:]
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
                        pages[-1].sections.append(WikiSection(title=title, level=level, blocks=[]))
                elif block.type == "paragraph":
                    content = re.sub(r"\[\d+\]", "", block.content)
                    if idx > 0 or block.start_with_indent:
                        pages[-1].sections[-1].blocks.append(SectionBlock(type="text", content=content))
                    else:
                        pages[-1].sections[-1].blocks[-1].content += content

    with get_cursor() as cursor:
        p = Path(f"preview/markdown/{name}")
        if not p.exists():
            p.mkdir()

        for idx, page in enumerate(pages, start=1):
            with open(f"preview/markdown/{name}/{name}_{idx}.md", mode="w", encoding="utf-8") as f:
                f.write(page.merge_sections())
            cursor.execute(
                "insert into book_core_corpus (title, raw_content, content) values (%s, %s, %s)",
                (name, page.model_dump_json(), page.model_dump_json()),
            )


if __name__ == "__main__":
    # ocr()
    for item in [
        "武士的成长与院政：平安时代后期",
        "源赖朝与幕府初创：镰仓时代",
        "《太平记》的时代：南北朝时代-室町时代",
        "织丰政权与江户幕府：战国时代",
        "天下泰平：江户时代前期",
    ]:
        merge(name=item)
