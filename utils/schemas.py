from typing import Annotated, Literal

from pydantic import BaseModel, Field


class WikiTableSchema(BaseModel):
    is_ordered: Annotated[bool, Field(description="表格内容是否有序")] = False
    mode: Annotated[
        Literal["Grouped", "Flat"], Field(description="表格的模型, 如果内容存在分组为Grouped, 否则为Flat")
    ] = "Flat"
    data: Annotated[
        dict[str, list[str]] | list[str],
        Field(
            description="清洗后的数据, 当mode=Grouped时类型是字典, Key为分组的标题, Value为清洗之后的内容列表；当mode=Flat时类型是列表, 内容是清洗之后的结果"
        ),
    ]


class WikiListSchema(BaseModel):
    # list_title: Annotated[str | None, Field(description="翻译之后的列表标题")] = None
    items: Annotated[list[str], Field(..., description="列表语义化转换之后的元素")]
