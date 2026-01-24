import os

import psycopg2

host = os.environ["DATABASE_HOST"]
port = os.environ["DATABASE_PORT"]
username = os.environ["DATABASE_USERNAME"]
password = os.environ["DATABASE_PASSWORD"]
db_name = os.environ["DATABASE_NAME"]


def get_db_conn():
    return psycopg2.connect(host=host, port=port, user=username, password=password, database=db_name)
