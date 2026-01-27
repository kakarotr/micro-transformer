from typing import Literal

from pydantic import BaseModel

type BlockType = Literal["text", "table", "list-title", "list-item"]


class SectionBlock(BaseModel):
    type: BlockType
    content: str | None


class WikiSection(BaseModel):
    title: str
    level: int
    blocks: list[SectionBlock]


class WikiPage(BaseModel):
    title: str
    category_name: str
    lang: str
    sections: list[WikiSection]
    full_content: str | None

    def merge_sections(self):
        contents = []
        contents.append(f"# {self.title}\n\n")
        for section in self.sections:
            title = section.title
            if title == "summary":
                contents.append(f"{self._merge_blocks(blocks=section.blocks)}")
            else:
                content = self._merge_blocks(blocks=section.blocks)
                contents.append(f"{'#' * section.level} {title}")
                if content:
                    contents.append("\n")
                    contents.append(f"{content}")
                else:
                    contents.append("\n\n")
        return "".join(contents).rstrip("\n")

    def _merge_blocks(self, blocks: list[SectionBlock]):
        contents = []
        for block in blocks:
            if block.content:
                contents.append(f"{block.content}\n\n")
        return "".join(contents)
