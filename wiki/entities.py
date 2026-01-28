from typing import Literal

from pydantic import BaseModel

type BlockType = Literal["text", "table", "olist", "ulist"]


class SectionBlock(BaseModel):
    type: BlockType
    list_title: str | None = None
    content: str | list


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
        contents.append(f"# {self.title}")
        for section in self.sections:
            title = section.title
            if title == "summary":
                contents.append(self._merge_blocks(blocks=section.blocks))
            else:
                content = self._merge_blocks(blocks=section.blocks)
                if content:
                    contents.append(f"{'#' * section.level} {title}\n{content}")
                else:
                    contents.append(f"{'#' * section.level} {title}")

        return "\n\n".join(contents)

    def _merge_blocks(self, blocks: list[SectionBlock]):
        contents = []
        for block in blocks:
            if block.content:
                if block.type == "text":
                    contents.append(block.content)
                elif block.type == "table":
                    lines = [f"- {line}" for line in block.content]
                    contents.append("\n".join(lines))
                elif block.type == "ulist" or block.type == "olist":
                    if isinstance(block.content, list):
                        list_title = block.list_title
                        if block.type == "ulist":
                            list_items = [f"- {item}" for item in block.content]
                        else:
                            list_items = [f"{idx}. {item}" for idx, item in enumerate(block.content, start=1)]
                        content = ""
                        if list_title:
                            content = f"{list_title}\n"
                        content += f"{'\n'.join(list_items)}"

        return "\n\n".join(contents)
