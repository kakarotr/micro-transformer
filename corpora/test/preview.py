from dotenv import load_dotenv
from pydantic import TypeAdapter

from corpora.core_knowledge.wiki.entities import WikiPage, WikiSection
from utils.db import get_db_conn

load_dotenv()

db = get_db_conn()
cursor = db.cursor()
cursor.execute("select title, sections from wiki_pages where sections is not null")
rows = cursor.fetchall()

adapter = TypeAdapter(list[WikiSection])

for row in rows:
    sections = adapter.validate_python(row[1])
    page = WikiPage(title=row[0], category_name="", lang="", sections=sections, full_content="")
    with open(f"preview/{row[0]}.md", mode="w", encoding="utf-8") as f:
        f.write(page.merge_sections())
