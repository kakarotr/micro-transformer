from typing import Annotated

from pydantic import BaseModel, Field


class TableSchema(BaseModel):
    lines: Annotated[list[str], Field(..., description="表格转化之后每行的内容")]
