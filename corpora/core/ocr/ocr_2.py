import base64
import json
from pathlib import Path

from dotenv import load_dotenv

from corpora.core.ocr.ocr_1 import Result, parse_header
from corpora.core.wiki.entities import SectionBlock, WikiPage, WikiSection
from corpora.utils.client import get_kimi_client, get_openrouter_client, get_qwen_client
from corpora.utils.db import get_cursor

prompt = f"""
# Role
你是一个精通多语言书籍排版分析与OCR后处理的专家。你需要处理一张**竖排（Vertical Layout）**的书籍扫描页。

# Goal
将页面内容解析为一个有序的“内容块（Block）”列表。
**核心任务**：
1. 提取正文和标题。
2. **文本标准化**：将繁体中文转换为简体中文，并执行**有条件**的数字标准化。
3. **结构化输出**：自动过滤掉图片注释、页眉页脚。

# Workflow & Rules

1. **阅读顺序 (Vertical Reading Order)**:
   - 该页面为**竖排版**（文字从上到下，列从右到左）。
   - **扫描逻辑**：请先定位最右侧的一列，从上读到下；然后移动到左边一列，依次类推。
   - 必须严格保持原书的段落逻辑，不要随意合并跨度过大的段落。

2. **文本处理规则 (Text Processing) - 优先级最高**:
   **必须严格遵守以下数字处理逻辑，切勿过度转换：**

   **第一步：保护专有名词（绝对不转换）**
   在进行任何数字转换前，先识别以下实体，保持其汉字原样：
   - **历史固有名词/官职/地名**：包含数字的固定词汇**绝不能改**。
     - *正确*：“三管领”、“四职”、“六波罗探题”、“九州探题”。
     - *错误*：“3管领”、“4职”。
   - **日本年号（Era Name）+ 年份**：年号后的年份保持中文汉字。
     - *正确*：“文正二年”、“长享元年”、“应仁二年”。
     - *错误*：“文正2年”、“长享1年”。
   - **章节标题序号**：如“第一章”、“第一节”保持汉字。

   **第二步：转换通用数字（仅转换以下情况）**
   仅在确认不属于上述“保护词”后，才将以下情况转换为阿拉伯数字：
   - **公历年份**：通常出现在括号内。
     - *示例*：“(一四六七)” -> “(1467)”。
   - **时间段/时长**：
     - *示例*：“长达十年” -> “长达10年”。
   - **日期（月/日）**：
     - *示例*：“九月” -> “9月”、“一日” -> “1日”。
   - **普通统计数量**：
     - *示例*：“两个家族” -> “2个家族”。

   **第三步：繁简转换**
   - 将所有提取的文本转换为**简体中文**（保留上述数字保护逻辑）。

3. **排除规则 (Exclusion)**:
   - **忽略图片注释**: 凡是出现在插图附近（通常为左下方或正下方）、字体明显小于正文的说明性文字（如“京畿阴霾区域地图”），请直接丢弃。
   - **忽略页眉页脚**: 忽略页面顶端或底端的页码。但保留正文标题（如“第一章...”）。

4. **类型判断 (`type`)**:
   - **标题 (`title`)**: 
     - 字体显著较大、加粗、或者位于页面最上方/最右侧起始位置的章节名。
   - **段落 (`paragraph`)**: 
     - 页面主体的正文文本。

5. **缩进检测 (`start_with_indent`)**:
   - 观察竖排文本列的**顶端**：
     - **True**: 该列顶端有明显的空白（通常为空两格）。
     - **False**: 该列顶端直接顶格书写。

# Output Format
严格遵守 JSON Schema，返回一个包含 `blocks` 的 JSON 对象。
{json.dumps(Result.model_json_schema(), ensure_ascii=False)}
"""
load_dotenv()


def ocr():
    images_path = Path("preview/images/中")
    files = [file for file in images_path.iterdir() if file.is_file() and file.name != ".DS_Store"]
    files.sort(key=lambda p: int(p.stem.split("_")[-1]))
    for file in files:
        print(file.stem)
        if int(file.stem.split("_")[-1]) != 191:
            continue
        with open(file, "rb") as f:
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
            temperature=0.2,
        )
        result = response.choices[0].message.content
        if result:
            with open(f"preview/jsons/中/{file.stem}.json", mode="w", encoding="utf-8") as f:
                f.write(result)


def merge():
    for item in ["上", "中", "下"]:
        jsons_path = Path(f"preview/jsons/{item}")
        files = [file for file in jsons_path.iterdir() if file.is_file() and file.name != ".DS_Store"]
        files.sort(key=lambda p: int(p.stem.split("_")[-1]))
        pages: list[WikiPage] = []

        for file in files:
            with open(file, mode="r", encoding="utf-8") as f:
                result = Result.model_validate_json(f.read())

            for idx, block in enumerate(result.blocks):
                if block.type == "title":
                    level, title = parse_header(text=block.content)
                    if level == 2:
                        if "—" in title:
                            segment = title.split("—")
                            if " " in segment[-1]:
                                suffix = segment[-1].split(" ")[-1]
                            else:
                                suffix = segment[-1]
                            title = f"{segment[0]}（{suffix}）"
                        pages.append(
                            WikiPage(
                                title=f"日本战国‧织丰时代史（{item}）",
                                category_name="",
                                lang="zh",
                                sections=[WikiSection(title=title, level=level, blocks=[])],
                            )
                        )
                    else:
                        title_level = (
                            3 if idx < len(result.blocks) - 1 and result.blocks[idx + 1].type == "title" else 4
                        )
                        pages[-1].sections.append(WikiSection(title=title, level=title_level, blocks=[]))
                elif block.type == "paragraph":
                    if block.start_with_indent:
                        pages[-1].sections[-1].blocks.append(SectionBlock(type="text", content=block.content))
                    else:
                        if pages[-1].sections[-1].blocks:
                            pages[-1].sections[-1].blocks[-1].content += block.content
                        else:
                            pages[-1].sections[-1].blocks.append(SectionBlock(type="text", content=block.content))

        for idx, page in enumerate(pages):
            with open(f"preview/markdown/{item}/{page.title}_{idx + 1}.md", mode="w", encoding="utf-8") as f:
                f.write(page.merge_sections())
            with get_cursor() as cursor:
                cursor.execute(
                    "insert into book_core_corpus (title, raw_content, content) values ('日本战国‧织丰时代史', %s, %s)",
                    (
                        page.model_dump_json(),
                        page.model_dump_json(),
                    ),
                )


# merge()
def a():
    prefix = "preview/images/上/page_"
    test_images = [
        f"{prefix}6.png",
        f"{prefix}8.png",
        f"{prefix}10.png",
        f"{prefix}12.png",
        f"{prefix}15.png",
        f"{prefix}30.png",
    ]

    for idx, item in enumerate(test_images, start=1):
        with open(item, "rb") as f:
            print(item)
            image_data = f.read()
            base64_bytes = base64.b64encode(image_data)
            base64_str = base64_bytes.decode("utf-8")

            model_name, client = get_openrouter_client()
            response = client.chat.completions.create(
                model="moonshotai/kimi-k2.5",
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


# ocr()
merge()
