from typing import Annotated

from pydantic import BaseModel, Field


class WikiTableSchema(BaseModel):
    lines: Annotated[list[str], Field(..., description="表格转化之后每行的内容")]


class WikiListSchema(BaseModel):
    items: Annotated[list[str], Field(..., description="列表语义化转换之后的元素")]
