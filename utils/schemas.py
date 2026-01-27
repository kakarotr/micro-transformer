from typing import Annotated

from pydantic import BaseModel, Field


class TableSchema(BaseModel):
    lines: Annotated[list[str], Field(..., description="表格转化之后每行的内容")]


class WikiListItemSchema(BaseModel):
    items: Annotated[list[str], Field(..., description="列表语义化转换之后的元素")]
