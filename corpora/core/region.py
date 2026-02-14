import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field, TypeAdapter

from corpora.core.wiki.entities import SectionBlock, WikiSection
from corpora.utils.client import get_deepseek_client
from corpora.utils.db import get_cursor


class RegionResult(BaseModel):
    summary: Annotated[str, Field(description="该区域的简短宏观概述 (300字以内)")]
    geography_geopolitics: Annotated[str, Field(description="地理特征、气候影响及地缘战略价值 (500字左右)")]
    economy_kokudaka: Annotated[str, Field(description="石高概况、特产资源、商业中心 (500字左右)")]
    political_transition: Annotated[str, Field(description="守护体制的崩溃与战国大名的权力更迭 (500字左右)")]
    military_strongholds: Annotated[str, Field(description="著名城池、防御体系及兵种战术 (500字左右)")]
    key_events: Annotated[str, Field(description="发生在此区域的决定性历史事件 (500字左右)")]


prompt = f"""
# Content Requirements (必须涵盖的维度)
请严格按照以下五个维度进行撰写：

1.  **地理与地缘战略 (Geography & Geopolitics):**
    * 地形特征（山地、平原、盆地）。
    * 气候对军事的影响（如北陆的大雪、关东的旱季）。
    * 交通要道（如东海道、中山道、水运港口）。
    * 战略价值（是上洛必经之路？是天然要塞？还是产粮大区？）。

2.  **经济与石高 (Economy & Kokudaka):**
    * 该地区的概算石高（生产力）。
    * 特产与战略资源（如：甲斐的金矿、堺的火枪、备前的刀剑、种程岛的铁炮）。
    * 商业中心或自由都市（如有）。

3.  **政治权力演变 (Political Transition):**
    * **守护大名时代：** 室町幕府时期原本是谁在统治（如管领家族）。
    * **战国大名崛起：** 谁通过“下克上”夺取了政权？（如北条早云夺取伊豆/相模）。
    * **主要势力：** 该区域长期割据的豪族或大名家（如中国地区的毛利氏、九州的岛津氏）。

4.  **军事与筑城 (Military & Strongholds):**
    * 该区域著名的坚城（如小田原城、春日山城、月山富田城）。
    * 该区域特有的兵种或战术风格（如武田骑兵、杂贺铁炮众、濑户内海水军）。

5.  **核心历史事件 (Key Events):**
    * 发生在该区域的决定性战役（如严岛之战、耳川之战、川中岛之战）。

# Style Constraints (风格约束)
* **语气：** 严谨、客观、学术化。禁止使用“小说体”、“演义体”或情感色彩浓重的词汇（如“遗憾的是”、“壮哉”）。
* **格式：** 输出为纯文本，段落分明。每一部分要有小标题。
* **术语：** 必须准确使用历史专有名词（如“守护代”、“国人众”、“一向一揆”、“惣村”）。

# Content Constraints (内容约束)
1.  **语言：** 必须使用**简体中文**。
2.  **风格：** 客观、严谨、学术化。禁止使用演义体或小说笔法。
3.  **时间锚点：** 以战国时代（1467-1600）为核心，必要时回溯镰仓/室町渊源。
4.  **实体准确性：** 人名、地名、官职名必须准确无误。
5.  **格式：** 适当使用Markdown的加粗符号（**）来标记重要内容。

# Output Schema
根据下面的Json Schema进行输出
{json.dumps(RegionResult.model_json_schema(), ensure_ascii=False)}
"""
load_dotenv()
regions = ["关东", "北陆", "近畿", "畿内", "中国", "山阳", "山阴", "甲信", "奥州", "出羽", "东海道", "四国", "九州"]
model_name, client = get_deepseek_client()

with get_cursor(autocommit=True) as cursor:
    for region in regions:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": region},
            ],
            temperature=1.2,
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        assert result is not None
        result = RegionResult.model_validate_json(result)
        sections = [
            WikiSection(title="summary", level=2, blocks=[SectionBlock(type="text", content=result.summary)]),
            WikiSection(
                title="地理与地缘战略",
                level=2,
                blocks=[SectionBlock(type="text", content=result.geography_geopolitics)],
            ),
            WikiSection(
                title="经济与石高", level=2, blocks=[SectionBlock(type="text", content=result.economy_kokudaka)]
            ),
            WikiSection(
                title="政治权力演变", level=2, blocks=[SectionBlock(type="text", content=result.political_transition)]
            ),
            WikiSection(
                title="军事与筑城", level=2, blocks=[SectionBlock(type="text", content=result.military_strongholds)]
            ),
            WikiSection(title="核心历史事件", level=2, blocks=[SectionBlock(type="text", content=result.key_events)]),
        ]
        data = TypeAdapter(list[WikiSection]).dump_json(sections).decode()
        cursor.execute(
            "insert into pedia_core_corpus (raw_title, title, raw_sections, sections, source, lang) values (%s, %s, %s, %s, 'wiki_ja', 'zh')",
            (region, region, data, data),
        )
