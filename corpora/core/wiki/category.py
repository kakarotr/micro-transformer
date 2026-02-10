import wikipediaapi
from psycopg2.extras import execute_values
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from utils.db import get_db_conn
from wiki.utils import get_wiki


def insert_page_by_category():
    conn = get_db_conn()
    cursor = conn.cursor()
    inserted_pages = get_inserted_pages(cursor=cursor)

    cursor.execute("SELECT name, lang FROM wiki_categories WHERE status = 0")
    rows = cursor.fetchall()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    )

    with Live(refresh_per_second=10) as live:
        task = progress.add_task("同步 Wikipedia 页面标题...", total=len(rows))

        for row in rows:
            category_name, lang = row
            status_msg = f"[bold yellow]当前分类:[/bold yellow] {category_name} ({lang})"
            live.update(Group(status_msg, progress))

            wiki = get_wiki(lang)
            category = wiki.page(f"Category:{category_name}")
            pages_to_insert = []

            for member in category.categorymembers.values():
                if member.namespace == wikipediaapi.Namespace.MAIN:
                    if member.title not in inserted_pages:
                        pages_to_insert.append((member.title, category_name, lang))
                        inserted_pages.add(member.title)

            if pages_to_insert:
                execute_values(cursor, "INSERT INTO wiki_pages (title, category_name, lang) VALUES %s", pages_to_insert)

            cursor.execute("UPDATE wiki_categories SET status = 1 WHERE name = %s", (category_name,))
            conn.commit()
            progress.advance(task)

        live.update("[bold green]✅ 所有分类页面已成功同步并入库！")

    cursor.close()
    conn.close()


def get_inserted_pages(cursor) -> set[str]:
    cursor.execute("select title from wiki_pages")
    rows = cursor.fetchall()
    return set([row[0] for row in rows])
