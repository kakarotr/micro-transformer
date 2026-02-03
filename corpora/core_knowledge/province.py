import json
from typing import Annotated

from dotenv import load_dotenv
from psycopg2.extras import Json
from pydantic import BaseModel, Field, TypeAdapter
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from corpora.core_knowledge.wiki.entities import SectionBlock, WikiSection
from utils.client import get_deepseek_client
from utils.db import get_db_conn

load_dotenv()


class Section(BaseModel):
    title: Annotated[str, Field(..., description="根据内容生成的具体、多样化标题（严禁使用通用标题）")]
    paragraph: Annotated[list[str], Field(..., description="该标题下的正文内容，必须切分为多个自然段")]


class ProvinceData(BaseModel):
    # 新增 summary 字段
    summary: Annotated[
        str, Field(..., description="该令制国的战国历史全景摘要（300字左右），高度浓缩地理、经济与核心历史。")
    ]
    sections: list[Section]


prompt = """
# Role
你是一位专注于日本战国史（Sengoku Period）的历史地理学家。你的任务是为大语言模型编写高质量的预训练语料。

# Task
请针对【{{令制国名称}}】这一地理区域，撰写一篇深度分析文章。
文章必须完全由“标题”和“正文段落”组成，不要有任何开场白或结束语。

# Constraints
1.  **语言**：使用流畅、专业的**简体中文**。
2.  **时间范围**：严格限定于日本战国时期（1467年应仁之乱至1615年大阪之阵）。
3.  **标题多样性（关键修改）**：
    * **严禁**使用“地理形势”、“经济基础”、“历史变迁”这种通用标题。
    * **必须**根据该令制国的具体特征起拟标题。
    * *正确示例*：“浓尾平原的交通枢纽地位”、“甲斐群山的天然屏障与封闭性”、“佐渡金山与北国航运的经济命脉”。
4.  **纪年规范**：
    * 文中凡是提及具体年份，必须采用“年号+具体年数（西历年份）”的格式（如：天正10年（1582年））。
5.  **分段逻辑**：
    * 严禁将所有内容塞进一个长段落。
    * `paragraph` 列表中的每个字符串代表一个独立的自然段（约250-350字）。
    * 必须根据时间节点的推移或论述焦点的转换进行强制分段。

# Content Sections (请按以下逻辑撰写，但需自定义标题)

1. **全局摘要 (Summary)**
    * **内容**：高度浓缩的一段话（约300字）。必须包含：该国所属的“道”（如东海道）、核心地理特征（如盆地、平原）、战国时期的最高石高数、以及统治该国最著名的氏族。
    * **作用**：作为整篇文章的开篇索引。

2. **板块一（主题：地理与地缘）**
    * *拟题要求*：标题需包含该国核心地形或战略特征。
    * *内容*：分析地形如何决定行军路线与防御体系。

3. **板块二（主题：经济与资源）**
    * *拟题要求*：标题需体现该国的核心产业、石高特征或著名特产。
    * *内容*：描述农业生产力、特产（矿山、港口）及对军事的支撑。

4. **板块三（主题：历史与战乱）**
    * *拟题要求*：标题需提及统治该国的主要家族或决定性战役。
    * *内容*：叙述统治权的更迭（守护大名 -> 战国大名）及关键战役。

# Output Schema
输出必须是一个 JSON 列表，列表中的每个元素必须符合以下 Schema：
```json
{{
  "title": "结合了具体地名、人名或事件的描述性标题 (严禁使用只有2-4个字的通用词)",
  "paragraph": [
    "段落1：专注于该主题的某一个侧面或时间段。",
    "段落2：承接上文，但在逻辑或时间上有所推进，避免与上一段粘连。",
    "段落3：进一步深入，确保每个段落长度适中。"
  ]
}}

# JSON SCHEMA
{json_schema}
"""

provinces = [
    "安艺国",
    "淡路国",
    "安房国",
    "阿波国",
    "伊贺国",
    "壹岐国",
    "石狩国",
    "伊豆国",
    "和泉国",
    "出云国",
    "伊势国",
    "因幡国",
    "胆振国",
    "伊予国",
    "磐城国",
    "岩代国",
    "石见国",
    "羽后国",
    "羽前国",
    "越后国",
    "越前国",
    "越中国",
    "大隅国",
    "隐岐国",
    "渡岛国",
    "甲斐国",
    "加贺国",
    "上总国",
    "河内国",
    "纪伊国",
    "北见国",
    "钏路国",
    "上野国",
    "相模国",
    "萨摩国",
    "佐渡国",
    "赞岐国",
    "信浓国",
    "志摩国",
    "下总国",
    "下野国",
    "后志国",
    "周防国",
    "骏河国",
    "摄津国",
    "但马国",
    "丹后国",
    "丹波国",
    "筑后国",
    "筑前国",
    "千岛国",
    "对马国",
    "天盐国",
    "远江国",
    "十胜国",
    "土佐国",
    "长门国",
    "根室国",
    "能登国",
    "播磨国",
    "肥后国",
    "备前国",
    "肥前国",
    "日高国",
    "常陆国",
    "飞驒国",
    "备中国",
    "日向国",
    "备后国",
    "丰前国",
    "丰后国",
    "伯耆国",
    "三河国",
    "美作国",
    "武蔵国",
    "山城国",
    "大和国",
    "陆奥国",
    "陆前国",
    "陆中国",
    "琉球国",
    "若狭国",
]

model_name, client = get_deepseek_client()

progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
)
with Live(refresh_per_second=10) as live:
    task = progress.add_task("生成令制国内容...", total=len(provinces))
    conn = get_db_conn()

    for idx, province in enumerate(provinces, start=1):
        status_msg = f"[bold yellow]当前令制国:[/bold yellow] {province} {idx} / {len(provinces)}"
        live.update(Group(status_msg, progress))
        progress.advance(task)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": prompt.format(
                        json_schema=json.dumps(ProvinceData.model_json_schema(), ensure_ascii=False)
                    ),
                },
                {"role": "user", "content": f"目标令制国：{province}"},
            ],
            temperature=1.2,
            response_format={"type": "json_object"},
        )

        result = response.choices[0].message.content
        assert result is not None
        result = ProvinceData.model_validate_json(result)
        sections = [
            WikiSection(
                title="summary",
                level=2,
                blocks=[SectionBlock(type="text", content=result.summary)],
            ),
            *[
                WikiSection(
                    title=item.title,
                    level=2,
                    blocks=[SectionBlock(type="text", content=block) for block in item.paragraph],
                )
                for item in result.sections
            ],
        ]
        data: str = TypeAdapter(list[WikiSection]).dump_json(sections).decode()
        cursor = conn.cursor()
        cursor.execute(
            "insert into wiki_pages (title, raw_sections, sections, category_name, lang) values (%s, %s, %s, %s, %s)",
            (province, data, data, "日本の旧国名", "zh"),
        )
        conn.commit()
    live.update("[bold green]✅ 所有令制国已成功同步并入库！")
    conn.close()
    conn.close()
