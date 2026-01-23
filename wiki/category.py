import wikipediaapi

from wiki.utils import get_db_conn, to_simplified


def insert_page_by_category():
    lst = set()
    conn = get_db_conn()
    wiki = wikipediaapi.Wikipedia(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        language="zh",
    )

    cursor = conn.execute("select name from wiki_category where status = 0")
    rows = cursor.fetchall()
    for row in rows:
        category_name = row[0]

        cat = wiki.page(f"Category:{category_name}")
        for member in cat.categorymembers.values():
            if member.namespace == wikipediaapi.Namespace.MAIN:
                if member.title not in lst:
                    conn.execute(
                        "insert into wiki_page (name, name_tc, category, status) values (?, ?, ?, ?)",
                        (to_simplified(member.title), member.title, to_simplified(category_name), 0),
                    )
                    lst.add(member.title)

        conn.execute("update wiki_category set status = 1 where name = ?", (category_name,))

    conn.commit()
    conn.close()
