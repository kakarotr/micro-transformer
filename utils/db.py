import os
from contextlib import contextmanager

import psycopg2


def get_db_conn():
    host = os.environ["DATABASE_HOST"]
    port = os.environ["DATABASE_PORT"]
    username = os.environ["DATABASE_USERNAME"]
    password = os.environ["DATABASE_PASSWORD"]
    db_name = os.environ["DATABASE_NAME"]

    return psycopg2.connect(host=host, port=port, user=username, password=password, database=db_name)


@contextmanager
def get_cursor(autocommit: bool = False):
    host = os.environ["DATABASE_HOST"]
    port = os.environ["DATABASE_PORT"]
    username = os.environ["DATABASE_USERNAME"]
    password = os.environ["DATABASE_PASSWORD"]
    db_name = os.environ["DATABASE_NAME"]

    conn = psycopg2.connect(host=host, port=port, user=username, password=password, database=db_name)
    if autocommit:
        conn.autocommit = True
    cursor = conn.cursor()

    try:
        yield cursor
    finally:
        if not autocommit:
            conn.commit()
        cursor.close
        conn.close()
