from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    from wiki.category import insert_page_by_category
    from wiki.page import extract, fetch_page_content

    # 織田信長
    with open("a.md", mode="w", encoding="utf-8") as f:
        f.write(extract(title="織田信長").full_content)

    # fetch_page_content()
