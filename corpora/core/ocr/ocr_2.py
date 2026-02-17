import base64
import json

from dotenv import load_dotenv

from corpora.core.ocr.ocr_1 import Result
from corpora.utils.client import get_kimi_client, get_qwen_client

prompt = f"""
# Role
你是一个精通多语言书籍排版分析与OCR后处理的专家。你需要处理一张**竖排（Vertical Layout）**的书籍扫描页。

# Goal
将页面内容解析为一个有序的“内容块（Block）”列表。
**核心任务**：
1. 提取正文和标题。
2. **文本标准化**：将繁体中文转换为简体中文，并将年份、日期、数量词中的中文数字转换为阿拉伯数字（如“一四六七”转为“1467”）。
3. **结构化输出**：自动过滤掉图片注释、页眉页脚。

# Workflow & Rules

1. **阅读顺序 (Vertical Reading Order)**:
   - 该页面为**竖排版**（文字从上到下，列从右到左）。
   - **扫描逻辑**：请先定位最右侧的一列，从上读到下；然后移动到左边一列，依次类推。
   - 必须严格保持原书的段落逻辑。

2. **文本处理规则 (Text Processing)**:
   - **繁简转换**: 输出时，必须将所有提取的文本转换为**简体中文**。
   - **数字转换逻辑 (关键)**: 
     - **保留项**: 凡是涉及**年号（Era Name）**的年份，保持中文原样。
       - *示例*：“文正二年”、“长享二年”、“应仁元年” -> **保持不变**。
     - **转换项**: 纯公历年份、月份、日期、以及表示数量/时长的词，转换为阿拉伯数字。
       - *示例*：“一四六七” -> “1467”；“九月” -> “9月”。
     - **专有名词**: 如“三管领”、“一向宗”等固定称呼保持原样。

3. **排除规则 (Exclusion)**:
   - **忽略图片注释**: 凡是出现在插图附近（通常为左下方或正下方）、字体明显小于正文的说明性文字（如“京畿阴霾区域地图”），请直接丢弃。
   - **忽略页眉页脚**: 页面最顶端（如章节名）如果与正文明显分离且具有导航性质，或页码，需根据情况判断。但在本任务中，**“第一章...”属于正文标题，需保留**。

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


def a():
    with open("2.png", "rb") as f:
        image_data = f.read()
        base64_bytes = base64.b64encode(image_data)
        base64_str = base64_bytes.decode("utf-8")

    model_name, client = get_kimi_client()
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
    print(result)


a()
