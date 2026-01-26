from dotenv import load_dotenv
from pydantic import RootModel

load_dotenv()


if __name__ == "__main__":
    import requests
    from bs4 import BeautifulSoup

    from wiki.category import insert_page_by_category
    from wiki.page import WikiPageParser

    # 織田信長
    # parser = WikiPageParser()
    # wiki_page = parser.parse(title="織田信長", lang="ja")
    # with open("preview.md", mode="w", encoding="utf-8") as f:
    #     f.write(wiki_page.merge_sections())
    # with open("preview.json", mode="w", encoding="utf-8") as f:
    #     f.write(RootModel(wiki_page).model_dump_json())
    response = requests.get(
        url=f"https://ja.wikipedia.org/w/api.php",
        params={
            "action": "parse",
            "format": "json",
            "page": "織田信長",  # <--- 这里修改：从 titles 改为 page
            "prop": "text",
            "disableeditsection": 1,
        },
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        },
    )
    data = response.json()

    if "parse" in data:
        html_content = data["parse"]["text"]["*"]
        with open("preivew.html", mode="w", encoding="utf-8") as f:
            soup = BeautifulSoup(html_content, "html.parser")
            for a_tag in soup.find_all("a"):
                a_tag.unwrap()
            for sub_tag in soup.find_all("sup"):
                sub_tag.decompose()
            f.write(str(soup))
