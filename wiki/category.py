import wikipediaapi

from wiki.data import categories
from wiki.utils import get_db_conn, to_simplified

wiki = wikipediaapi.Wikipedia(
    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    language="zh",
)

lst = set()
conn = get_db_conn()

for category in categories:
    print(category)
    cat = wiki.page(f"Category:{category}")
    for member in cat.categorymembers.values():
        if member.namespace == wikipediaapi.Namespace.MAIN:
            if member.title not in lst:
                conn.execute(
                    "insert into wiki_page (name, name_tc, category, status) values (?, ?, ?, ?)",
                    (to_simplified(member.title), member.title, to_simplified(category), 0),
                )
                lst.add(member.title)

conn.commit()
conn.close()
