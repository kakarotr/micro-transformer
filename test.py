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
 <table class="wikitable" style="font-size:98%;">
  <caption>
   本能寺の変前の織田家諸将（および徳川家康）の動向
  </caption>
  <tbody>
   <tr>
    <th colspan="2" style="white-space:nowrap;">
     大将（与力・一門衆）
    </th>
    <th style="width:3.5em;">
     所在
    </th>
    <th>
     配下の軍勢
    </th>
    <th style="width:2.5em;">
     状況
    </th>
    <th style="width:6.5em;">
     対立武将
    </th>
    <th>
     対立勢力
    </th>
    <th>
     直前の行動・できごと
    </th>
   </tr>
   <tr>
    <td colspan="2">
     **織田信長**
    </td>
    <td rowspan="2">
     山城国
    </td>
    <td rowspan="2">
     20-30
     から150-160
    </td>
    <td rowspan="2">
     在京
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2" style="font-size:90%;">
     5月29日
     、信長は中国出陣の準備をして待機するように命じ、小姓衆をつれて安土より上洛した
     。その際、茶道具の名器38点
     を携えており、
     6月1日
     、
     近衛前久
     を主賓として茶会を開いた
     。京都滞在は5日間の計画で、先に淡路で信孝の閲兵に向かうと伝えられていた
     。
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     小姓衆（
     森成利
     ・
     森長隆
     ・
     森長氏
     等）
    </td>
   </tr>
   <tr>
    <td colspan="2">
     **織田信忠**
    </td>
    <td rowspan="2">
     山城国
    </td>
    <td rowspan="2">
     数百
    </td>
    <td rowspan="2">
     在京
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2" style="font-size:90%;">
     5月14日、信忠は甲州征伐から安土に帰還
     。21日に上洛して
     妙覚寺に滞在。斎藤利治は病気で、信長・信忠に心配されて御供を外されていたが、後日、病気は治ったと
     加治田城
     を出発し
     、兄（
     斎藤利堯
     ）が
     留守居
     する岐阜城を通り過ぎてそのまま、変前日（6月1日）に京に入り、妙覚寺で信忠と合流した
     。同日夜、信忠は
     村井貞勝
     をつれて本能寺を訪れ、父と酒を飲み交わした
     。
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td style="font-size:90%;">
     一門衆
     （
     津田長利
     ・
     勝長
     ）・
     奉行衆
     （
     村井貞勝
     ・
     菅屋長頼
     ）・
     母衣衆
     （
     福富秀勝
     ・
     野々村正成
     ・
     毛利良勝
     ）・御供衆（
     猪子兵助
     ・
     団忠正
     ・
     斎藤利治
     ）等
    </td>
   </tr>
   <tr>
    <td colspan="2">
     **明智光秀**
    </td>
    <td rowspan="2">
     丹波国
     <br/>
     山城国
    </td>
    <td rowspan="2">
     **13,000**
    </td>
    <td rowspan="2">
     出陣
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2" style="font-size:90%;">
     天正9年の
     馬揃え
     では総括責任者を務めた光秀であった
     が、甲州征伐では信長の身辺警固を命じられたのみで、活躍の場はなく安土に帰還。その後、徳川家康の饗応役に任命されて準備をしたが、
     備中高松城
     包囲中の羽柴秀吉から急使があり、援軍に赴くように信長から急遽命じられて、饗応役も長秀と交代。光秀はすぐに軍勢の支度のために5月17日に坂本城に戻り、さらには26日には領地の丹波・
     亀山城
     に向かった。
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     明智秀満
     ・
     明智光忠
     ・
     斎藤利三
     ・
     溝尾茂朝
     ・
     藤田行政
     ・
     伊勢貞興
     ・
     山崎長徳
     ・
     並河易家
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     明智十五郎
     ・
     阿閉貞征
     ・
     妻木広忠
     ・
     京極高次
     ・
     山崎堅家
    </td>
    <td>
     近江国
    </td>
    <td>
     不明
    </td>
    <td>
     在番
    </td>
    <td>
     -
    </td>
    <td>
     -
    </td>
    <td style="font-size:90%;">
     嫡男十五郎は
     坂本城
     に留守居。阿閉は
     山本山城
     、京極は
     上平寺城
     と、それぞれ居城にいた。（山城国の）山崎堅家は安土の館に詰めていた。
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     筒井順慶
    </td>
    <td>
     山城国
    </td>
    <td>
     -
    </td>
    <td>
     在京
    </td>
    <td>
     -
    </td>
    <td>
     -
    </td>
    <td style="font-size:90%;">
     筒井順慶は甲州征伐に明智配下として出征して
     大和
     郡山城
     に帰還
     。直前は順慶本人は京に滞在していた
     。
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     長岡忠興
     （
     長岡藤孝
     ）・
     池田恒興
     ・
     高山右近
     ・
     中川清秀
     ・
     塩川長満
    </td>
    <td>
     丹後国
     <br/>
     摂津国
    </td>
    <td>
     8,500以上
    </td>
    <td>
     準備
    </td>
    <td>
     -
    </td>
    <td>
     -
    </td>
    <td style="font-size:90%;">
     長岡忠興・池田（
     元助
     ・
     照政
     ）・中川は、甲州征伐に明智配下として出征したが、5月17日、秀吉に援軍として向かう光秀の与力として、他2氏と共に先鋒を命じられたので、領国に戻って再び出陣準備をしていた
     。
    </td>
   </tr>
   <tr>
    <td colspan="2">
     **羽柴秀吉**
    </td>
    <td rowspan="2">
     備中国
    </td>
    <td rowspan="2">
     30,000
     <br/>
     から
     <br/>
     60,000
    </td>
    <td rowspan="2">
     対陣
    </td>
    <td rowspan="2">
     清水宗治
     ・
     末近信賀
     ・
     毛利輝元
     ・
     吉川元春
     ・
     小早川隆景
    </td>
    <td rowspan="2">
     35,000以上
     <br/>
     ~50,000
    </td>
    <td rowspan="2" style="font-size:90%;">
     天正10年3月5日、秀吉は山陽道に出陣し、4月4日、
     宇喜多秀家
     の
     岡山城
     に入城。14日、秀吉は宇喜多勢と龍王山と八幡山に陣した。25日に
     冠山城
     を攻略して
     林重真
     が切腹。5月2日に
     乃美元信
     が開城して
     宮路山城
     を退去し、
     加茂城
     では
     生石治家
     が寝返ったが
     桂広繁
     が（宇喜多勢の）
     戸川秀安
     を撃退して本丸は守った。7日、秀吉は蛙ヶ鼻に陣を移し、足守川を堰き止めて高松城を水没させた
     。15日、秀吉は信長に状況を知らせ、毛利勢の総大将・毛利輝元が間もなく出陣すると報告した。2日後、これを聞いた信長は、明智光秀らに出陣を命じた。小早川隆景が
     幸山城
     に入り、21日、毛利輝元・吉川元春も合流して総勢5万の援軍が到着した
     。
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     羽柴秀長
     ・
     羽柴秀勝
     ・
     杉原家次
     ・
     蜂須賀正勝
     ・
     堀尾吉晴
     ・
     神子田正治
     ・
     宇喜多忠家
     ・
     黒田孝高
     ・
     仙石秀久
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     宮部継潤
     ・
     亀井茲矩
    </td>
    <td>
     因幡国
    </td>
    <td>
     不明
    </td>
    <td>
     城番
    </td>
    <td>
     -
    </td>
    <td>
     -
    </td>
    <td style="font-size:90%;">
     宮部は鳥取城。亀井は
     鹿野城
     。
    </td>
   </tr>
   <tr>
    <td colspan="2">
     **神戸信孝**
     ・
     丹羽長秀
    </td>
    <td rowspan="2">
     和泉国
     <br/>
     摂津国
    </td>
    <td rowspan="3">
     14,000
     <br/>
     （三好勢6,000
     ）
    </td>
    <td rowspan="2">
     準備
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2" style="font-size:90%;">
     5月11日
     、信孝は
     住吉
     ヘ出陣し、
     四国征伐
     の渡海準備を始めた
     。予定では6月2日に淡路に渡海して（中国に向かう途中の）信長も4日に来るはずであった
     。長秀は5月14日に家康・梅雪・信忠を
     番場
     で接待し、光秀が出た後は20日以降は4名
     で家康一行を接待した。堀が羽柴の伝令として派遣され、菅屋が奉行の役目で離れ、長秀は信澄と共に引き続き饗応役となるように命じられ、先に大坂に向かった
     。
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     蜂屋頼隆
     ・
     九鬼嘉隆
     ・
     津田信澄
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     三好笑岩
     ・
     十河存保
     ・
     三好康俊
    </td>
    <td>
     阿波国
    </td>
    <td>
     戦闘
    </td>
    <td style="font-size:90%;">
     香宗我部親泰
     ・
     長宗我部信親
     ・
     比江山親興
     ・
     江村親俊
    </td>
    <td>
     3,000
    </td>
    <td style="font-size:90%;">
     先鋒・三好笑岩は5月に
     勝瑞城
     に入り、
     一宮城
     と
     夷山城
     を攻略し、康俊が
     岩倉城
     で織田側に寝返って呼応。阿波半国を奪還して神戸信孝の本隊の到着を待っていた。長宗我部氏は
     畑山城
     からは撤退したが、
     海部城
     ・
     大西城
     では抵抗する構えであった
     。
    </td>
   </tr>
   <tr>
    <td colspan="2">
     **柴田勝家**
    </td>
    <td rowspan="2">
     越中国
     <br/>
     能登国
    </td>
    <td rowspan="2">
     48,000
     <br/>
     （魚津城攻囲15,000
     ）
    </td>
    <td rowspan="2">
     戦闘
    </td>
    <td rowspan="2">
     上杉景勝
     ・
     中条景泰
     ・
     上条政繁
     ・
     吉江宗信
     （
     景資
     ）・
     須賀盛能
    </td>
    <td rowspan="2">
     3,000
     <br/>
     または5,000
     <br/>
     +城兵
    </td>
    <td rowspan="2" style="font-size:90%;">
     <p>
      河田長親
      は既に亡く
      上条政繁が指揮する越中の上杉勢。
      3月11日
      、
      小島職鎮
      ら一揆勢が
      神保長住
      の
      富山城
      を落として長住を監禁したが
      、織田勢が奪還。柴田・前田らは
      松倉城
      と
      魚津城
      を囲み、越境して
      勝山城
      も攻めた。上杉景勝は
      新発田重家
      の反乱
      もあって対応に苦慮。5月16日、景勝は
      天神山城
      に後詰で入るが
      、
      魚津城の戦い
      の最中に
      長景連
      が
      棚木城
      を奪った際にも、長連龍・前田利家による奪還（22日）
      に為すすべなく、勝ち目のない上杉勢は6月を前にして撤退を検討していた。
     </p>
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     柴田勝豊
     ・
     佐々成政
     ・
     前田利家
     ・
     佐久間盛政
     ・
     徳山則秀
     ・
     神保氏張
     ・
     長連龍
     ・椎名孫六入道
    </td>
   </tr>
   <tr>
    <td colspan="2">
     **滝川一益**
    </td>
    <td rowspan="2">
     上野国
    </td>
    <td rowspan="2">
     26,200
    </td>
    <td rowspan="2">
     出陣
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2">
     -
    </td>
    <td rowspan="2" style="font-size:90%;">
     滝川一益は当初より後北条氏との取次役であったが、甲州征伐では信忠の補佐役も務めて、3月11日に天目山で武田勝頼父子を自害させて首を取るという大手柄を挙げた。
     3月23日
     、事実上の一番手柄として、上野国と信濃2郡、名馬を与えられて、関東八州の警固役に任命されて
     、
     上野厩橋城
     に入城した。上州・信州・武州の諸将
     を与力として従え、一益はこの軍勢を糾合して、
     三国峠
     を越えて越後に攻め入る予定であった。
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td style="font-size:90%;">
     滝川益重
     ・
     津田秀政
     ・稲田九蔵・
     小幡信貞
     ・
     真田昌幸
     ・
     内藤昌月
     ・
     由良国繁
     ・
     安中久繁
     ・
     成田氏長
     ・
     木部貞朝
     ・
     依田信蕃
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     河尻秀隆
     ・
     森長可
     ・
     毛利秀頼
     ・
     稲葉貞通
    </td>
    <td>
     甲斐国
     <br/>
     信濃国
    </td>
    <td>
     不明
    </td>
    <td>
     鎮定
    </td>
    <td>
     芋川正元
     ・一揆勢
    </td>
    <td>
     不明
    </td>
    <td style="font-size:90%;">
     河尻は穴山領を除く甲斐国と諏訪郡を領して
     府中城
     に、森長可は北信濃4郡を領して
     海津城
     に、毛利秀頼は伊奈郡を領して
     飯田城
     に入った
     。4月初旬、
     飯山城
     を一揆が攻撃して稲葉貞通を追った。長可が反撃して城を奪回し、一揆勢8千余を鎮圧した
     。その際に女子供を含む数千人を成敗した。信濃は不穏な状況で、長可の越後攻めは遅延していた。
    </td>
   </tr>
   <tr>
    <td style="width:2em; border-top-style:hidden;">
    </td>
    <td>
     木曾義昌
     ・
     小笠原信嶺
    </td>
    <td>
     信濃国
    </td>
    <td>
     不明
    </td>
    <td>
     安堵
    </td>
    <td>
     -
    </td>
    <td>
     -
    </td>
    <td style="font-size:90%;">
     木曾義昌は
     木曽谷
     の2郡の安堵、さらに
     安曇郡
     ・
     筑摩郡
     を加増された
     。小笠原信嶺も旧領安堵された。
    </td>
   </tr>
   <tr>
    <td colspan="2">
     **徳川家康**
     ・
     穴山信君
    </td>
    <td>
     河内国
    </td>
    <td>
     34
    </td>
    <td>
     旅行
    </td>
    <td>
     -
    </td>
    <td>
     -
    </td>
    <td style="font-size:90%;">
     家康は一貫して低姿勢で、天正3年に叔父
     水野信元
     を、天正7年には嫡男
     信康
     を、内通の嫌疑で斬った。天正10年、甲州征伐の折にも信長の帰途を誠心誠意もてなし喜ばれる。駿河を与えられた返礼として家康は穴山梅雪と共に5月中旬に安土は訪れ、信長は光秀や長秀を付けて接待させた。その後、
     堺
     の見物を勧められて
     長谷川秀一
     が案内人として同伴した。
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
