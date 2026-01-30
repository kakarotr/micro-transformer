import math
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, thread

from pydantic import TypeAdapter

from utils.db import get_db_conn
from wiki.entities import WikiSection
from wiki.page import WikiPageParser


def fetch_page(n_threads: int = 10):
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("select id, title, lang from wiki_pages where raw_sections is null")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    total = len(rows)
    chunk_size = math.ceil(total / n_threads)

    chunks = [rows[i : i + chunk_size] for i in range(0, total, chunk_size)]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(process, chunks)


def process(chunk):
    conn = get_db_conn()
    conn.autocommit = True
    cursor = conn.cursor()
    parser = WikiPageParser()
    for item in chunk:
        try:
            page = parser.parse(page_title=item[1], lang=item[2])
            if page:
                cursor.execute(
                    "update wiki_pages set raw_sections = %s where id = %s",
                    (TypeAdapter(list[WikiSection]).dump_json(page.sections).decode(), item[0]),
                )
                print(f"{item[1]}处理完成")
            time.sleep(random.uniform(0.5, 2))
        except:
            error_stack = traceback.format_exc()
            print(f"{item[1]}错误: err: {error_stack}")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    fetch_page(n_threads=5)
