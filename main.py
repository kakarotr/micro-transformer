from dotenv import load_dotenv
from pydantic import RootModel

load_dotenv()


if __name__ == "__main__":
    from wiki.category import insert_page_by_category
    from wiki.page import WikiPageParser

    # 織田信長
    parser = WikiPageParser()
    wiki_page = parser.parse(title="今川義元", lang="ja")
    with open("preview.md", mode="w", encoding="utf-8") as f:
        f.write(wiki_page.merge_sections())
    with open("preview.json", mode="w", encoding="utf-8") as f:
        f.write(RootModel(wiki_page).model_dump_json())
