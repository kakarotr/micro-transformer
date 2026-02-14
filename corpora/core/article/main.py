from pathlib import Path

from dotenv import load_dotenv

from corpora.utils.db import get_cursor


def save():
    load_dotenv()
    with get_cursor() as cursor:
        target_dir = Path("preview/article/core/zhanguos/zhanyi")
        for item in target_dir.iterdir():
            if item.is_file():
                cursor.execute(
                    "insert into article_core_corpus (title, domain, source, content) values (%s, 'core', 'zhanguos', %s)",
                    (item.name.rstrip(".md"), item.read_text(encoding="utf-8")),
                )


save()
