import math

import wikipediaapi

from corpora.utils.db import get_db_conn


def get_wiki(lang: str):
    return wikipediaapi.Wikipedia(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        language=lang,
    )


def get_chunks(sql: str, n_threads: int):
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    total = len(rows)
    chunk_size = math.ceil(total / n_threads)

    chunks = [rows[i : i + chunk_size] for i in range(0, total, chunk_size)]

    return chunks
