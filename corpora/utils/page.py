import bs4

from corpora.core.wiki.entities import BlockType, SectionBlock, WikiPage


def add_block(
    doc: bs4.Tag,
    page: WikiPage,
    current_title: str,
    block_type: BlockType,
    content: str | list[str],
    find_title,
    list_title: str | None = None,
):
    """
    添加Block
    """
    if current_title == find_title(tag=doc):
        # 当前添加的Block归属的条目已经添加过
        # 判断当前Block和最后一个Block的类型是否一致
        # 一致则在上一个Block的内容后面最后进行追加当前Block的内容
        last_section = page.sections[-1]
        if last_section.blocks and last_section.blocks[-1].type == block_type:
            block_content: str | list[str] = last_section.blocks[-1].content
            if block_content:
                if isinstance(content, str):
                    if block_type == "text":
                        last_section.blocks[-1].content = f"{block_content}\n\n{content}"
                    else:
                        last_section.blocks.append(
                            SectionBlock(type=block_type, content=content, list_title=list_title)
                        )

                else:
                    last_section.blocks.append(SectionBlock(type=block_type, content=content, list_title=list_title))
            else:
                last_section.blocks[-1] = SectionBlock(type=block_type, content=content, list_title=list_title)
        else:
            last_section.blocks.append(SectionBlock(type=block_type, content=content, list_title=list_title))
    else:
        page.sections[-1].blocks.append(SectionBlock(type=block_type, content=content, list_title=list_title))
