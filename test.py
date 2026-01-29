import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from utils.prompt import wiki_list_prompt, wiki_table_prompt
from utils.schemas import WikiListSchema, WikiTableSchema

load_dotenv()

client = OpenAI(base_url=os.environ["LLM_URL"], api_key=os.environ["LLM_KEY"])


def test_list():
    response = client.chat.completions.create(
        model=os.environ["LLM_NAME"],
        messages=[
            {
                "role": "system",
                "content": wiki_list_prompt.format(
                    json_schema=json.dumps(WikiListSchema.model_json_schema()),
                    page_title="本能寺の変",
                    section_title="本能寺",
                    list_title="",
                ),
            },
            {
                "role": "user",
                "content": """
    <ul>
    <li>
    織田信長
    </li>
    <li>
    森成利
    （蘭丸） -
    可成
    三男
    </li>
    <li>
    森長隆
    （坊丸） - 可成四男
    </li>
    <li>
    森長氏
    （力丸） - 可成五男
    </li>
    <li>
    小河愛平
    </li>
    <li>
    高橋虎松
    </li>
    <li>
    金森義入
    </li>
    <li>
    菅屋角蔵
    - 長頼の子
    </li>
    <li>
    魚住勝七
    </li>
    <li>
    武田喜太郎
    </li>
    <li>
    大塚又一郎
    </li>
    <li>
    狩野又九郎
    </li>
    <li>
    薄田輿五郎
    </li>
    <li>
    今川孫二郎
    </li>
    <li>
    落合小八郎
    </li>
    <li>
    伊藤彦作
    </li>
    <li>
    久々利亀
    </li>
    <li>
    種田亀
    </li>
    <li>
    山田弥太郎
    </li>
    <li>
    飯河宮松
    </li>
    <li>
    祖父江孫
    </li>
    <li>
    柏原鍋兄弟
    </li>
    <li>
    一雲斎針阿弥
    -
    同朋衆
    </li>
    <li>
    平尾久助
    </li>
    <li>
    大塚孫三
    </li>
    <li>
    湯浅直宗
    </li>
    <li>
    小倉松寿
    </li>
    <li>
    矢代勝介
    </li>
    <li>
    伴太郎左衛門
    </li>
    <li>
    伴正林
    </li>
    <li>
    村田吉五
    </li>
    <li>
    小澤六左衛門
    </li>
    <li>
    赤座直則
    </li>
    </ul>
    """,
            },
        ],
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content
    assert result is not None
    # print(WikiListItemSchema.model_validate_json(result))
    print(result)


def test_table():
    response = client.chat.completions.create(
        model=os.environ["LLM_NAME"],
        messages=[
            {
                "role": "system",
                "content": wiki_table_prompt.format(
                    json_schema=json.dumps(WikiTableSchema.model_json_schema()),
                    page_title="本能寺の変",
                    section_title="本能寺",
                ),
            },
            {
                "role": "user",
                "content": """
 <table class="wikitable">
  <tbody>
   <tr>
    <th style="background-color:#deb0c4;">
    </th>
    <th style="background-color:#deb0c4;">
     武将
    </th>
    <th style="background-color:#deb0c4;">
     石高（万石）
    </th>
    <th style="background-color:#deb0c4;">
     兵力
    </th>
    <th style="background-color:#b0c4de;">
    </th>
    <th style="background-color:#b0c4de;">
     武将
    </th>
    <th style="background-color:#b0c4de;">
     石高（万石）
    </th>
    <th style="background-color:#b0c4de;">
     兵力
    </th>
   </tr>
   <tr>
    <th rowspan="36" style="background-color:#deb0c4;">
     西軍
    </th>
    <td>
     毛利輝元
    </td>
    <td style="text-align:right">
     120.5
    </td>
    <td style="text-align:right">
     -
    </td>
    <th rowspan="36" style="background-color:#b0c4de;">
     東軍
    </th>
    <td>
     徳川家康
     ○
    </td>
    <td style="text-align:right">
     256.0
    </td>
    <td style="text-align:right">
     約30,000
    </td>
   </tr>
   <tr>
    <td>
     毛利秀元
     ▲
    </td>
    <td style="text-align:right">
     （20.0）
    </td>
    <td style="text-align:right">
     約16,000
    </td>
    <td>
     松平忠吉
     ○
    </td>
    <td style="text-align:right">
     （10.0）
    </td>
    <td style="text-align:right">
     3,000
    </td>
   </tr>
   <tr>
    <td>
     吉川広家
     ●
    </td>
    <td style="text-align:right">
     （14.2）
    </td>
    <td style="text-align:right">
     －
    </td>
    <td>
     井伊直政
     ○
    </td>
    <td style="text-align:right">
     （12.0）
    </td>
    <td style="text-align:right">
     3,600
    </td>
   </tr>
   <tr>
    <td>
     大友義統
    </td>
    <td style="text-align:right">
     文禄の役
     で改易
     <br/>
     毛利輝元の支援
    </td>
    <td style="text-align:right">
     約2,000
    </td>
    <td>
     本多忠勝
     ○
    </td>
    <td style="text-align:right">
     （10.0）
    </td>
    <td style="text-align:right">
     500
    </td>
   </tr>
   <tr>
    <td>
     上杉景勝
    </td>
    <td style="text-align:right">
     120.0
    </td>
    <td style="text-align:right">
     約25,000
    </td>
    <td>
     前田利長
    </td>
    <td style="text-align:right">
     84.0
    </td>
    <td style="text-align:right">
     約25,000
    </td>
   </tr>
   <tr>
    <td>
     島津義弘
     ○
    </td>
    <td style="text-align:right">
     73.0
    </td>
    <td style="text-align:right">
     約1,700
    </td>
    <td>
     伊達政宗
    </td>
    <td style="text-align:right">
     58.0
    </td>
    <td style="text-align:right">
     23,000 - 28,000
    </td>
   </tr>
   <tr>
    <td>
     宇喜多秀家
     ○
    </td>
    <td style="text-align:right">
     57.0
    </td>
    <td style="text-align:right">
     17,220
    </td>
    <td>
     堀秀治
    </td>
    <td style="text-align:right">
     45.0
    </td>
    <td style="text-align:right">
     －
    </td>
   </tr>
   <tr>
    <td>
     佐竹義宣
    </td>
    <td style="text-align:right">
     54.0
    </td>
    <td style="text-align:right">
     －
    </td>
    <td>
     最上義光
    </td>
    <td style="text-align:right">
     24.0
    </td>
    <td style="text-align:right">
     約7,000
    </td>
   </tr>
   <tr>
    <td>
     小早川秀秋
     ●
    </td>
    <td style="text-align:right">
     37.0
    </td>
    <td style="text-align:right">
     15,675
    </td>
    <td>
     福島正則
     ○
    </td>
    <td style="text-align:right">
     24.0
    </td>
    <td style="text-align:right">
     6,000
    </td>
   </tr>
   <tr>
    <td>
     長宗我部盛親
     ▲
    </td>
    <td style="text-align:right">
     22.0
    </td>
    <td style="text-align:right">
     6,660
    </td>
    <td>
     加藤清正
    </td>
    <td style="text-align:right">
     20.0
    </td>
    <td style="text-align:right">
     －
    </td>
   </tr>
   <tr>
    <td>
     小西行長
     ○
    </td>
    <td style="text-align:right">
     20.0
    </td>
    <td style="text-align:right">
     6,000
    </td>
    <td>
     筒井定次
     ○
    </td>
    <td style="text-align:right">
     20.0
    </td>
    <td style="text-align:right">
     2,850
    </td>
   </tr>
   <tr>
    <td>
     増田長盛
    </td>
    <td style="text-align:right">
     20.0
    </td>
    <td style="text-align:right">
     －
    </td>
    <td>
     細川忠興
     ○
    </td>
    <td style="text-align:right">
     18.0
    </td>
    <td style="text-align:right">
     5,100
    </td>
   </tr>
   <tr>
    <td>
     石田三成
     ○
    </td>
    <td style="text-align:right">
     19.4
    </td>
    <td style="text-align:right">
     5,820
    </td>
    <td>
     黒田長政
     ○
    </td>
    <td style="text-align:right">
     18.0
    </td>
    <td style="text-align:right">
     5,400
    </td>
   </tr>
   <tr>
    <td>
     織田秀信
    </td>
    <td style="text-align:right">
     13.5
    </td>
    <td style="text-align:right">
     約9,000
    </td>
    <td>
     蜂須賀至鎮
     ○
    </td>
    <td style="text-align:right">
     17.7
    </td>
    <td style="text-align:right">
     不明
    </td>
   </tr>
   <tr>
    <td>
     小川祐忠
     ●
    </td>
    <td style="text-align:right">
     7.0
    </td>
    <td style="text-align:right">
     2,100
    </td>
    <td>
     浅野幸長
     ○
    </td>
    <td style="text-align:right">
     16.0
    </td>
    <td style="text-align:right">
     6,510
    </td>
   </tr>
   <tr>
    <td>
     安国寺恵瓊
     ▲
    </td>
    <td style="text-align:right">
     6.0
    </td>
    <td style="text-align:right">
     1,800
    </td>
    <td>
     池田輝政
     ○
    </td>
    <td style="text-align:right">
     15.2
    </td>
    <td style="text-align:right">
     4,500
    </td>
   </tr>
   <tr>
    <td>
     毛利勝信
     ○
    </td>
    <td style="text-align:right">
     6.0
    </td>
    <td style="text-align:right">
     不明
    </td>
    <td>
     生駒一正
     ○
    </td>
    <td style="text-align:right">
     15.0
    </td>
    <td style="text-align:right">
     1,830
    </td>
   </tr>
   <tr>
    <td>
     長束正家
     ▲
    </td>
    <td style="text-align:right">
     5.0
    </td>
    <td style="text-align:right">
     1,500
    </td>
    <td>
     中村一栄
    </td>
    <td style="text-align:right">
     14.5
    </td>
    <td style="text-align:right">
     4,350
    </td>
   </tr>
   <tr>
    <td>
     大谷吉継
     ○
    </td>
    <td style="text-align:right">
     5.0
    </td>
    <td style="text-align:right">
     1,500
    </td>
    <td>
     藤堂高虎
     ○
    </td>
    <td style="text-align:right">
     8.0
    </td>
    <td style="text-align:right">
     2,490
    </td>
   </tr>
   <tr>
    <td>
     大谷吉治
     ○
    </td>
    <td style="text-align:right">
     －
    </td>
    <td style="text-align:right">
     <p>
      3,500
     </p>
    </td>
    <td>
     堀尾吉晴
    </td>
    <td style="text-align:right">
     10.0
    </td>
    <td style="text-align:right">
     －
    </td>
   </tr>
   <tr>
    <td>
     木下頼継
     ○
    </td>
    <td style="text-align:right">
     2.5
    </td>
    <td style="text-align:right">
     750
    </td>
    <td>
     加藤嘉明
     ○
    </td>
    <td style="text-align:right">
     10.0
    </td>
    <td style="text-align:right">
     3,000
    </td>
   </tr>
   <tr>
    <td>
     田丸直昌
     ○
    </td>
    <td style="text-align:right">
     4.0
    </td>
    <td style="text-align:right">
     不明
    </td>
    <td>
     田中吉政
     ○
    </td>
    <td style="text-align:right">
     10.0
    </td>
    <td style="text-align:right">
     3,000
    </td>
   </tr>
   <tr>
    <td>
     真田昌幸
    </td>
    <td style="text-align:right">
     3.8
    </td>
    <td style="text-align:right">
     2,500 - 3,000
    </td>
    <td>
     京極高知
     ○
    </td>
    <td style="text-align:right">
     10.0
    </td>
    <td style="text-align:right">
     不明
    </td>
   </tr>
   <tr>
    <td>
     脇坂安治
     ●
    </td>
    <td style="text-align:right">
     3.3
    </td>
    <td style="text-align:right">
     990
    </td>
    <td>
     京極高次
    </td>
    <td style="text-align:right">
     6.0
    </td>
    <td style="text-align:right">
     約3,000
    </td>
   </tr>
   <tr>
    <td>
     赤座直保
     ●
    </td>
    <td style="text-align:right">
     2.0
    </td>
    <td style="text-align:right">
     600
    </td>
    <td>
     寺沢広高
     ○
    </td>
    <td style="text-align:right">
     8.0
    </td>
    <td style="text-align:right">
     2,400
    </td>
   </tr>
   <tr>
    <td>
     平塚為広
     ○
    </td>
    <td style="text-align:right">
     1.2
    </td>
    <td style="text-align:right">
     360
    </td>
    <td>
     山内一豊
     ○
    </td>
    <td style="text-align:right">
     5.9
    </td>
    <td style="text-align:right">
     2,058
    </td>
   </tr>
   <tr>
    <td>
     朽木元綱
     ●
    </td>
    <td style="text-align:right">
     1.0
    </td>
    <td style="text-align:right">
     600
    </td>
    <td>
     金森長近
     ○
    </td>
    <td style="text-align:right">
     3.9
    </td>
    <td style="text-align:right">
     1,140
    </td>
   </tr>
   <tr>
    <td>
     戸田勝成
     ○
    </td>
    <td style="text-align:right">
     1.0
    </td>
    <td style="text-align:right">
     300
    </td>
    <td>
     有馬豊氏
     ○
    </td>
    <td style="text-align:right">
     3.0
    </td>
    <td style="text-align:right">
     900
    </td>
   </tr>
   <tr>
    <td>
     河尻秀長
     ○
    </td>
    <td style="text-align:right">
     1.0
    </td>
    <td style="text-align:right">
     300
    </td>
    <td>
     織田長益
     ○
    </td>
    <td style="text-align:right">
     0.2
    </td>
    <td style="text-align:right">
     450
    </td>
   </tr>
   <tr>
    <td>
     石川貞清
     ○
    </td>
    <td style="text-align:right">
     －
    </td>
    <td style="text-align:right">
     360
    </td>
    <td>
     古田重然
     ○
    </td>
    <td style="text-align:right">
     －
    </td>
    <td style="text-align:right">
     1,020
    </td>
   </tr>
   <tr>
    <td>
     織田信高
     ○
    </td>
    <td style="text-align:right">
     －
    </td>
    <td style="text-align:right">
     不明
    </td>
    <td>
     徳川秀忠
    </td>
    <td style="text-align:right">
     （－）
    </td>
    <td style="text-align:right">
     約15,000
    </td>
   </tr>
   <tr>
    <td>
     毛利元康
    </td>
    <td style="text-align:right">
     （－）
    </td>
    <td style="text-align:right">
     －
    </td>
    <td>
     榊原康政
    </td>
    <td style="text-align:right">
     （10.0）
    </td>
    <td style="text-align:right">
     3,000
    </td>
   </tr>
   <tr>
    <td>
     小早川秀包
    </td>
    <td style="text-align:right">
     13.0
    </td>
    <td style="text-align:right">
     －
    </td>
    <td>
     大久保忠隣
    </td>
    <td style="text-align:right">
     （6.5）
    </td>
    <td style="text-align:right">
     －
    </td>
   </tr>
   <tr>
    <td>
     立花宗茂
    </td>
    <td style="text-align:right">
     13.2
    </td>
    <td style="text-align:right">
     －
    </td>
    <td>
     酒井家次
    </td>
    <td style="text-align:right">
     （3.7）
    </td>
    <td style="text-align:right">
     900
    </td>
   </tr>
   <tr>
    <td>
     筑紫広門
    </td>
    <td style="text-align:right">
     1.8
    </td>
    <td style="text-align:right">
     －
    </td>
    <td>
     黒田孝高
    </td>
    <td style="text-align:right">
     （18.0）
    </td>
    <td style="text-align:right">
     9,000（一時的に13,000）
    </td>
   </tr>
   <tr>
    <td>
     鍋島直茂
     ・
     勝茂
     ●
    </td>
    <td style="text-align:right">
    </td>
    <td style="text-align:right">
     32,000
    </td>
    <td>
    </td>
    <td>
    </td>
    <td>
    </td>
   </tr>
  </tbody>
 </table>
""",
            },
        ],
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content
    print(result)


if __name__ == "__main__":
    test_table()
