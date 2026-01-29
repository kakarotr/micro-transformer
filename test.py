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
                    page_title="豊臣秀吉",
                    section_title="制度",
                    list_title="信長旧臣",
                ),
            },
            {
                "role": "user",
                "content": """
 <dt>
   信長旧臣
  </dt>
  <dd>
   丹羽長秀
   、
   蜂須賀正勝
   、
   前野長康
   、
   蒲生氏郷
   、
   堀秀政
   、
   細川藤孝
   、
   細川忠興
   、
   蜂屋頼隆
   、
   京極高次
   、
   長谷川秀一
   、
   長谷川与次
   、
   日根野弘就
   、
   日根野盛就
   、
   長谷川宗仁
   、
   矢部家定
   、
   建部寿徳
   、
   稲葉一鉄
   、
   市橋長利
   、
   伊東長久
   、
   九鬼嘉隆
   、
   古田重然
   、
   堀内氏善
   、
   丸毛兼利
   、
   毛利秀頼
   、
   猪子一時
  </dd>
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
                    page_title="応仁の乱",
                    section_title="各勢力の動向",
                ),
            },
            {
                "role": "user",
                "content": """
 <table class="wikitable" style="text-align:center; font-size:x-small; margin-right:0px;">
  <caption style="font-size:x-small;">
   応仁の乱における守護大名・国人の東軍（細川方）・西軍（山名方）去就一覧表
   <br/>
   ×印は応仁の乱終戦までに死去した武将、勢力欄の桃色表記は足利氏一門、黄緑色表記は足利氏一門以外の
   七頭
   を示す。
  </caption>
  <tbody>
   <tr style="white-space:nowrap;">
    <th style="width:6em">
     勢力
    </th>
    <th style="width:20em">
     領国
    </th>
    <th style="width:23em">
     西軍
     <br/>
     ★印は東軍から西軍へ寝返った武将
    </th>
    <th style="width:23em">
     東軍
     <br/>
     ◆印は西軍から東軍へ寝返った武将
    </th>
    <th style="width:15em">
     備考
    </th>
   </tr>
   <tr>
    <th style="background-color:#fcf;">
     足利氏
    </th>
    <td style="text-align:left">
     奉公衆
     領
     <br/>
     奉行衆
     領
    </td>
    <td style="text-align:left">
     足利義視
     （義政の弟）★
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     足利義政
     （第8代将軍）
     <br/>
     足利義尚
     （第9代将軍）
    </td>
    <td style="text-align:left">
     将軍家
    </td>
   </tr>
   <tr>
    <th style="background-color:#fcf;">
     吉良氏
    </th>
    <td style="text-align:left">
     三河
    </td>
    <td style="text-align:left">
     吉良義藤
     （東条家）
    </td>
    <td style="text-align:left">
     吉良義真
     （西条家）
    </td>
    <td style="text-align:left">
     御一家
    </td>
   </tr>
   <tr>
    <th style="background-color:#fcf;">
     渋川氏
    </th>
    <td style="text-align:left">
     肥前
    </td>
    <td style="text-align:left">
     渋川教直
     （肥前守護）
    </td>
    <td>
    </td>
    <td style="text-align:left">
     九州探題
     <br/>
     御一家
    </td>
   </tr>
   <tr>
    <th style="background-color:#fcf;">
     今川氏
    </th>
    <td style="text-align:left">
     駿河
    </td>
    <td>
    </td>
    <td style="text-align:left">
     今川義忠
     （駿河守護）×
    </td>
    <td style="text-align:left">
    </td>
   </tr>
   <tr>
    <th style="background-color:#fcf;">
     仁木氏
    </th>
    <td style="text-align:left">
     伊勢
     <br/>
     伊賀
     <br/>
     丹波
    </td>
    <td style="text-align:left">
     仁木教将
     （伊勢仁木氏）
     <br/>
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     <br/>
     仁木政長
     （伊賀守護）
     <br/>
     仁木成長
     （丹波仁木氏）
    </td>
    <td style="text-align:left">
    </td>
   </tr>
   <tr>
    <th style="background-color:#fcf;">
     斯波氏
    </th>
    <td style="text-align:left">
     越前・遠江
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     尾張
    </td>
    <td style="text-align:left">
     斯波義廉
     （管領・渋川系当主）
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     織田敏広
     （尾張守護代・岩倉家）
    </td>
    <td style="text-align:left">
     斯波義敏
     （大野系当主）
     <br/>
     斯波義寛
     （義敏の子）
     <br/>
     斯波持種
     （大野家）×
     <br/>
     甲斐敏光
     （越前遠江守護代）◆
     <br/>
     朝倉孝景
     ◆
     <br/>
     織田敏定
     （清洲家）
    </td>
    <td style="text-align:left">
     管領
     （1467-77年）
     <br/>
     三管領家
     <br/>
     ※
     武衛騒動
     も参照
    </td>
   </tr>
   <tr>
    <th style="background-color:#fcf;">
     畠山氏
    </th>
    <td style="text-align:left">
     河内・紀伊・越中
     <br/>
     <br/>
     <br/>
     能登
    </td>
    <td style="text-align:left">
     畠山義就
     （総州家）
     <br/>
     畠山政国
     （義就の猶子）×
     <br/>
     遊佐就家
     <br/>
     畠山義統
     （能登守護）
    </td>
    <td style="text-align:left">
     畠山政長
     （管領・尾州家）
     <br/>
     神保長誠
     （越中紀伊郡守護代）
     <br/>
     遊佐長直
     （河内守護代）
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     管領（-1467,73,77年-）
     <br/>
     三管領家
    </td>
   </tr>
   <tr>
    <th style="background-color:#fcf;">
     細川氏
    </th>
    <td style="text-align:left">
     摂津・丹波・讃岐・土佐
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     和泉
     <br/>
     <br/>
     備中
     <br/>
     淡路
     <br/>
     阿波・三河
     <br/>
     <br/>
     <br/>
     <br/>
     伊予
    </td>
    <td>
    </td>
    <td style="text-align:left">
     **細川勝元**
     （管領・宗家）×
     <br/>
     細川政元
     （勝元の子）
     <br/>
     細川政国
     （典厩家・政元後見）
     <br/>
     安富元綱
     （執事）×
     <br/>
     安富元家
     <br/>
     内藤元貞
     （丹波守護代）
     <br/>
     細川四天王
     （讃岐守護代等）
     <br/>
     細川勝益
     （土佐守護代）
     <br/>
     長宗我部文兼
     <br/>
     細川常有
     （和泉上半国守護）
     <br/>
     細川政久
     （和泉下半国守護）
     <br/>
     細川勝久
     （備中守護）
     <br/>
     細川成春
     （淡路守護）
     <br/>
     細川成之
     （阿波三河守護）
     <br/>
     東条国氏
     （三河守護代）×
     <br/>
     三好長之
     （阿波郡守護代）
     <br/>
     三好之長
     （長之の子）
     <br/>
     細川賢氏
     （伊予守護）
    </td>
    <td style="text-align:left">
     管領（1468-73年）
     <br/>
     三管領家
    </td>
   </tr>
   <tr>
    <th style="background-color:#fcf;">
     一色氏
    </th>
    <td style="text-align:left">
     丹後・伊勢
     <br/>
     <br/>
     尾張
     <br/>
     三河
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     一色義直
     （宗家）
     <br/>
     一色義春
     （義直の子）
     <br/>
     一色義遠
     （尾張分郡守護）
     <br/>
     一色政照
     （三河分郡守護）
     <br/>
     一色時家
    </td>
    <td>
    </td>
    <td style="text-align:left">
     四職
     <br/>
     七頭
    </td>
   </tr>
   <tr>
    <th style="background-color:#dfa;">
     京極氏
    </th>
    <td style="text-align:left">
     近江・隠岐
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     出雲
     <br/>
     飛騨
    </td>
    <td style="text-align:left">
     <br/>
     <br/>
     京極乙童子丸
     （勝秀の子）★
     <br/>
     京極政光
     （持清の次男）★×
     <br/>
     多賀清直
     ★
     <br/>
     <br/>
     三木久頼
     ★×
    </td>
    <td style="text-align:left">
     京極持清
     （侍所所司）×
     <br/>
     京極勝秀
     （持清の嫡男）×
     <br/>
     京極孫童子丸
     （勝秀の嫡男）×
     <br/>
     京極政経
     （持清の三男）
     <br/>
     多賀高忠
     （侍所所司代）
     <br/>
     尼子清定
     （出雲守護代）
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     侍所所司
     （-1466年）
     <br/>
     四職
     <br/>
     佐々木氏
     庶流
     <br/>
     ※
     京極騒乱
     も参照
    </td>
   </tr>
   <tr>
    <th style="background-color:#dfa;">
     赤松氏
    </th>
    <td style="text-align:left">
     播磨・備前・美作・加賀
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     有馬元家
     （赤松分家）×
     <br/>
     <br/>
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     赤松政則
     （侍所所司）
     <br/>
     浦上則宗
     （侍所所司代）
     <br/>
     宇野政秀
     （赤松分家）
     <br/>
     小寺則職
     <br/>
     松田元成
    </td>
    <td style="text-align:left">
     侍所所司（1471年-）
     <br/>
     四職
    </td>
   </tr>
   <tr>
    <th style="background-color:#dfa;">
     山名氏
    </th>
    <td style="text-align:left">
     但馬・播磨
     <br/>
     <br/>
     <br/>
     因幡
     <br/>
     伯耆・備前
     <br/>
     <br/>
     石見・美作
     <br/>
     備後・安芸
    </td>
    <td style="text-align:left">
     **山名宗全**
     （宗家）×
     <br/>
     山名教豊
     （宗全の子）×
     <br/>
     山名政豊
     （教豊の子）
     <br/>
     山名豊氏
     （因幡守護）
     <br/>
     山名教之
     （伯耆備前守護）×
     <br/>
     山名豊之
     （教之の子）×
     <br/>
     山名政清
     （石見美作守護）
     <br/>
    </td>
    <td style="text-align:left">
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     山名是豊
     （宗全の子）
    </td>
    <td style="text-align:left">
     四職
     <br/>
     1474年以降は東軍
     <br/>
     播磨・備前・美作は赤松氏に奪還された
    </td>
   </tr>
   <tr>
    <th style="background-color:#dfa;">
     伊勢氏
    </th>
    <td style="text-align:left">
     山城
     <br/>
     <br/>
     <br/>
     <br/>
     三河
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     伊勢貞藤
     （貞親の弟）
     <br/>
     飯尾為数
     （恩賞方）×
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     伊勢貞親
     （政所執事）×
     <br/>
     伊勢貞宗
     （貞親の嫡子）
     <br/>
     伊勢盛定
     （
     後北条氏
     の祖・
     伊勢宗瑞
     の父）
     <br/>
     蜷川親元
     （政所執事代）
     <br/>
     戸田宗光
     <br/>
     松平信光
     （
     徳川氏
     の祖）
    </td>
    <td style="text-align:left">
     政所執事
     <br/>
     将軍養育係
    </td>
   </tr>
   <tr>
    <th style="background-color:#dfa;">
     土岐氏
    </th>
    <td style="text-align:left">
     美濃
     <br/>
     <br/>
     <br/>
     <br/>
     <br/>
     伊勢
    </td>
    <td style="text-align:left">
     土岐成頼
     （美濃守護）
     <br/>
     斎藤利藤
     （美濃守護代）
     <br/>
     斎藤妙椿
     （利藤の後見）
     <br/>
     斎藤妙純
     （利藤の弟）
     <br/>
     石丸利光
     <br/>
    </td>
    <td style="text-align:left">
     <br/>
     <br/>
     富島光仲
     <br/>
     長江景秀
     ×
     <br/>
     長江利景
     （景秀の子）
     <br/>
     土岐政康
     （伊勢半国守護→解任）
    </td>
    <td style="text-align:left">
    </td>
   </tr>
   <tr>
    <th>
     富樫氏
    </th>
    <td style="text-align:left">
     加賀
     <br/>
    </td>
    <td style="text-align:left">
     富樫幸千代
     （政親の弟）
    </td>
    <td style="text-align:left">
     富樫政親
     （加賀半国守護）◆
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th>
     小笠原氏
    </th>
    <td style="text-align:left">
     信濃
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     小笠原清宗
     （府中家）
     <br/>
    </td>
    <td style="text-align:left">
     小笠原政秀
     （信濃守護・鈴岡家）
     <br/>
     小笠原家長
     （松尾家）
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th>
     姉小路家
    </th>
    <td style="text-align:left">
     飛騨
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     姉小路勝言
     （宗家。小島家）
     <br/>
    </td>
    <td style="text-align:left">
     姉小路基綱
     （古川家）
     <br/>
     姉小路之綱
     （向家）
    </td>
    <td style="text-align:left">
     飛騨国司家
    </td>
   </tr>
   <tr>
    <th>
     北畠氏
    </th>
    <td style="text-align:left">
     伊勢
     <br/>
     <br/>
     <br/>
    </td>
    <td>
    </td>
    <td style="text-align:left">
     北畠教具
     （伊勢国司・伊勢半国守護）×
     <br/>
     北畠政郷
     （教具の子）
     <br/>
     木造教親
     （北畠分家）×
    </td>
    <td style="text-align:left">
     伊勢国司家
     <br/>
    </td>
   </tr>
   <tr>
    <th>
     武田氏
    </th>
    <td style="text-align:left">
     若狭・丹後
     <br/>
     <br/>
     <br/>
     <br/>
     安芸
    </td>
    <td style="text-align:left">
     <br/>
     <br/>
     <br/>
     <br/>
     武田元綱
     （安芸分郡守護）★
    </td>
    <td style="text-align:left">
     武田信賢
     （若狭丹後守護）×
     <br/>
     武田国信
     （信賢の弟）
     <br/>
     逸見繁経
     ×
     <br/>
     粟屋賢家
     <br/>
     <br/>
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th>
     六角氏
    </th>
    <td style="text-align:left">
     近江
     <br/>
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     六角高頼
     （宗家）
     <br/>
     山内政綱
     （後見）
     <br/>
     伊庭貞隆
    </td>
    <td style="text-align:left">
     六角政堯
     （→近江守護）×
     <br/>
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     佐々木氏嫡流
     <br/>
    </td>
   </tr>
   <tr>
    <th>
     大内氏
    </th>
    <td style="text-align:left">
     長門・周防・豊前・筑前
     <br/>
     <br/>
     <br/>
     <br/>
     石見
    </td>
    <td style="text-align:left">
     大内政弘
     （宗家。周防長門豊前守護）
     <br/>
     内藤弘矩
     <br/>
     陶弘護
     （周防守護代）
     <br/>
     相良正任
     <br/>
     益田兼堯
    </td>
    <td style="text-align:left">
     大内教幸
     （政弘の伯父）◆
     <br/>
     内藤武盛
     （長門守護代）◆
     <br/>
     仁保盛安
     ◆
     <br/>
     <br/>
     吉見信頼
     ◆
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th>
     河野氏
    </th>
    <td style="text-align:left">
     伊予
    </td>
    <td style="text-align:left">
     河野通春
     （予州家）
    </td>
    <td style="text-align:left">
     河野教通
     （宗家）
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th>
     少弐氏
    </th>
    <td style="text-align:left">
     豊前・筑前
     <br/>
     <br/>
    </td>
    <td>
    </td>
    <td style="text-align:left">
     少弐教頼
     （筑前守護）×
     <br/>
     少弐政資
     （教頼の子）
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th>
     宗氏
    </th>
    <td style="text-align:left">
     対馬
    </td>
    <td>
    </td>
    <td style="text-align:left">
     宗貞国
     （対馬守護）
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th>
     大友氏
    </th>
    <td style="text-align:left">
     豊後・筑後
    </td>
    <td>
    </td>
    <td style="text-align:left">
     大友親繁
     （豊後筑後守護）
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th>
     菊池氏
    </th>
    <td style="text-align:left">
     肥後
    </td>
    <td>
    </td>
    <td style="text-align:left">
     菊池重朝
     （肥後守護）
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th>
     島津氏
    </th>
    <td style="text-align:left">
     薩摩・大隅・日向
     <br/>
     <br/>
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     島津季久
     （豊州家）
     <br/>
     島津久逸
     （伊作家）★
     <br/>
     島津国久
     （薩州家）★
     <br/>
     島津友久
     （相州家）★
    </td>
    <td style="text-align:left">
     島津立久
     （宗家。薩摩大隅日向守護）×
     <br/>
     島津忠昌
     （立久の子）
     <br/>
     <br/>
     <br/>
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <th rowspan="9">
     国人衆
    </th>
   </tr>
   <tr>
    <td style="text-align:left">
     信濃
     <br/>
    </td>
    <td>
    </td>
    <td style="text-align:left">
     木曾家豊
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <td style="text-align:left">
     飛騨
     <br/>
     <br/>
    </td>
    <td>
    </td>
    <td style="text-align:left">
     江馬左馬助
     <br/>
     内ヶ島為氏
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <td style="text-align:left">
     伊勢
    </td>
    <td style="text-align:left">
     長野政高
     ★
    </td>
    <td style="text-align:left">
     関盛元
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <td style="text-align:left">
     近江
     <br/>
     <br/>
    </td>
    <td>
    </td>
    <td style="text-align:left">
     朽木貞綱
     <br/>
     蒲生貞秀
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <td style="text-align:left">
     大和
     <br/>
     <br/>
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     越智家栄
     <br/>
     古市胤栄
     <br/>
     古市澄胤
     （胤栄の弟）
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     筒井順永
     <br/>
     成身院光宣
     （順永の兄）×
     <br/>
     十市遠清
     <br/>
     箸尾為国
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <td style="text-align:left">
     安芸
     <br/>
     <br/>
     <br/>
     <br/>
    </td>
    <td style="text-align:left">
     小早川盛景
     （竹原家）
     <br/>
     小早川弘景
     （盛景の子）
     <br/>
     小早川弘平
     （弘景の子）
     <br/>
     毛利豊元
     ★
    </td>
    <td style="text-align:left">
     小早川煕平
     （沼田家）×
     <br/>
     小早川敬平
     （
     煕平
     の子）
     <br/>
     <br/>
     吉川経基
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <td style="text-align:left">
     肥後
    </td>
    <td style="text-align:left">
     相良為続
     ★
    </td>
    <td style="text-align:left">
     阿蘇惟歳
    </td>
    <td>
    </td>
   </tr>
   <tr>
    <td style="text-align:left">
     日向
    </td>
    <td style="text-align:left">
     伊東祐堯
     ★
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
