import json
from typing import Annotated

from pydantic import BaseModel, Field


class Title(BaseModel):
    text: Annotated[str, Field(description="标题内容")]
    level: Annotated[int, Field(description="标题层级")]


class Paragraph(BaseModel):
    content: Annotated[str, Field(description="段落内容")]
    start_with_indent: Annotated[bool, Field(description="是否有首行缩进")]


class Result(BaseModel):
    title: Annotated[str, Field(description="标题内容")]
    paragraphs: Annotated[list[Paragraph], Field(description="提取到的段落信息")]


prompt = f"""
# Role
你是一个专业的书籍数字化专家，擅长从书籍扫描件中提取结构化数据。你不仅能识别文字（OCR），还能精准分析版面布局（Layout Analysis）。

# Goal
请提取图片中的标题和正文段落，并严格按照 JSON Schema 输出。

# Workflow & Rules
1. **标题提取 (`title`)**:
   - 寻找页面顶部字号最大、加粗或居中的文本作为标题。
   - 如果没有明显的标题，该字段留空字符串。

2. **段落提取 (`paragraphs`)**:
   - 按照阅读顺序提取页面内的正文文本。
   - **视觉缩进检测 (`start_with_indent`)**: 这是一个关键步骤。你需要仔细观察每个文本块的第一行。
     - 如果第一行的起始位置比该段落的其他行明显向右偏移（通常是两个字符的宽度），则标记为 `true`。
     - 如果第一行与左侧边缘对齐（通常出现在引用块、列表或上一页未结束的段落），则标记为 `false`。
     - 对于只有一行的段落，请参考页面上其他段落的缩进习惯来判断。

3. **文本清洗**:
   - 自动合并跨行的断句。
   - 去除原文中的注脚标记（如 [1]）。
   - 保持标点符号的准确性。
   - 如果图片中有页码、页眉页脚，请忽略，不要混入正文。

# Output Format
必须严格遵循提供的 JSON Schema。
{json.dumps(Result.model_json_schema(), ensure_ascii=False)}
"""
