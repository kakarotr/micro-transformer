from concurrent.futures import ThreadPoolExecutor

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from corpora.core.wiki.utils import get_chunks
from corpora.utils.client import get_deepseek_client
from corpora.utils.db import get_db_conn

from .translate import Result, title_prompt


def translate_title(n_threads: int = 10):
    chunks = get_chunks(sql="select id, raw_title from pedia_core_corpus where title is null", n_threads=n_threads)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(process, chunks)


def process(chunk):
    conn = get_db_conn()
    cursor = conn.cursor()
    model_name, client = get_deepseek_client()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    )
    with Live(refresh_per_second=10) as live:
        task = progress.add_task("翻译页面标题...", total=len(chunk))
        for row in chunk:
            status_msg = f"[bold yellow]当前标题:[/bold yellow] {row[1]})"
            live.update(Group(status_msg, progress))
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": title_prompt}, {"role": "user", "content": row[1]}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content
            assert result is not None
            result = Result.model_validate_json(result)
            cursor.execute("update pedia_core_corpus set title = %s where id = %s", (result.text, row[0]))
            conn.commit()
            progress.advance(task)

    cursor.close()
    conn.close()


translate_title(n_threads=43)
