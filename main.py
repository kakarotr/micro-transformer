from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    from wiki.page import WikiPageParser

    test_titles = [
        "織田信長",
        "豊臣秀吉",
        "徳川家康",
        "武田信玄",
        "上杉謙信",
        "桶狭間の戦い",
        "本能寺の変",
        "関ヶ原の戦い",
        "大坂の陣",
        "方広寺鐘銘事件",
    ]
    # # 織田信長
    parser = WikiPageParser()
    wiki_page = parser.parse(title="大坂の陣", lang="ja")
    if wiki_page:
        with open("preview/preview.json", mode="w", encoding="utf-8") as f:
            f.write(wiki_page.model_dump_json(indent=2))
        with open("preview/preview.md", mode="w", encoding="utf-8") as f:
            f.write(wiki_page.merge_sections())
