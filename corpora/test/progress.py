import time

from rich.progress import (
    BarColumn,
    FileSizeColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

with Progress(
    SpinnerColumn(),
    TextColumn("[blod blue]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TransferSpeedColumn(),
    FileSizeColumn(),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("[green]下载固件...", total=1000)
    while not progress.finished:
        progress.update(task, advance=5)
        time.sleep(0.02)
