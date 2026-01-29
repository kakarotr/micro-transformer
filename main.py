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
        "伊達政宗",
        "桶狭間の戦い",
        "本能寺の変",
        "関ヶ原の戦い",
        "大坂の陣",
        "方広寺鐘銘事件",
        "応仁の乱",
    ]
    # # 織田信長
    parser = WikiPageParser()
    title = "応仁の乱"
    wiki_page = parser.parse(title=title, lang="ja")
    if wiki_page:
        with open(f"preview/{title}.json", mode="w", encoding="utf-8") as f:
            f.write(wiki_page.model_dump_json(indent=2))
        with open(f"preview/{title}.md", mode="w", encoding="utf-8") as f:
            f.write(wiki_page.merge_sections())
