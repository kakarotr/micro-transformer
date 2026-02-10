import csv

from utils.db import get_db_conn


def save_pedia_info():
    conn = get_db_conn()
    cursor = conn.cursor()

    with open("pedia.csv", mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            title = row[2]
            douyin_url = row[3]
            baidu_url = row[4]
            if douyin_url:
                cursor.execute(
                    "insert into pedia_core_corpus (title, title_sp, source, url) values (%s, %s, 'douyin', %s)",
                    (title, title, f"https://www.baike.com{douyin_url}"),
                )
            if baidu_url:
                cursor.execute(
                    "insert into pedia_core_corpus (title, title_sp, source, url) values (%s, %s, 'baidu', %s)",
                    (title, title, baidu_url),
                )

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    with open("pedia.csv", mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            title = row[2]
            douyin_url = row[3]
            baidu_url = row[4]

            if douyin_url and not baidu_url:
                print(title)
