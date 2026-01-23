from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    from wiki.category import insert_page_by_category
    from wiki.document import extract

    content = extract(title="織田信長")
    with open("a.md", mode="w", encoding="utf-8") as f:
        f.write(content.full_content)
