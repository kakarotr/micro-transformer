import sys

import jieba

jieba.initialize()
with open("tokenizer/jieba/jieba_add_words.txt", mode="r", encoding="utf-8") as f:
    words = []
    for line in f.readlines():
        jieba.add_word(line.strip(), freq=20000)

sengoku_suffixes = ["军", "家", "氏", "势", "党", "众", "城", "国", "守", "殿"]
danger_particles = [
    # 方位
    "中",
    "内",
    "外",
    "上",
    "下",
    "里",
    "前",
    "后",
    # 介/连词
    "在",
    "与",
    "和",
    "同",
    "对",
    "向",
    "由",
    "从",
    # 助词
    "的",
    "了",
    "着",
    "过",
    "地",
    "得",
    # 高危副/动词
    "必",
    "将",
    "多",
    "也",
    "都",
    "却",
    "就",
]
for suffix in sengoku_suffixes:
    for particle in danger_particles:
        jieba.suggest_freq((suffix, particle), True)


def test_scenario():
    scenario_1 = [
        "现代人的本能是追求更好的生活，这是中国经济发展的重要原因。",
        "去重庆旅游一定要去看看那里的山城夜景，吃一吃正宗的火锅。",
        "遇到危险时，动物的本能反应是逃跑，这在中国的偏远山区很常见。",
    ]
    scenario_2 = [
        "织田信长的野望是上洛，在桶狭间之战中击败了今川义元，随后与武田军交战。",
        "丰臣秀吉的军队向小田原城进发，对北条家形成了巨大的包围网。",
        "德川家康在关原之战后，确立了江户幕府的统治基础，将领地分封给有功的家臣。",
    ]
    scenario_3 = [
        "织田信长与德川家康结成了同盟，而尼子胜久和长宗我部盛亲则在各地苦战。",
        "武田信玄病逝后，武田胜赖未能维持家族霸权，最终被织田家消灭。",
        "上杉谦信的义理让他在战国独树一帜，但上杉景胜却在关原之战中选错了阵营。",
    ]
    scenario_4 = [
        "室町幕府的权威衰落后，各地守护大名纷纷崛起，为了争夺家督之位开启了下克上的时代。",
        "他被朝廷任命为关白，不仅掌握了天下的政权，还大量没收了国人众的万石领地。",
        "征夷大将军的头衔不仅是荣誉，更是统领全日本武士的合法法理依据。",
    ]
    scenario_5 = [
        "信长从尾张国出兵，放弃了清洲城，将本城迁往岐阜城。他在山城国建立了政权。",
        "丰臣军未能立刻攻克关东的忍城与吉田郡山城，最终只得退回浓州休整。",
        "越前国的朝仓家在稻叶山城陷落后感到了危机，急忙修缮了自家的一乘谷城。",
    ]
    scenario_6 = [
        "武田军的赤备骑兵遭遇了织田家和德川氏的铁炮队，上杉势也在越后按兵不动。",
        "毛利军在水战中击败了尼子氏的残余势力，巩固了毛利家在中国地方的霸权。",
        "岛津势在九州的扩张引起了丰臣秀吉的警觉，最终导致了九州征伐的爆发。",
    ]
    for idx, scenario in enumerate([scenario_1, scenario_2, scenario_3, scenario_4, scenario_5, scenario_6]):
        print(f"{'-' * 5}场景{idx + 1}{'-' * 5}")
        for item in scenario:
            print("|".join(jieba.lcut(item)))
        print("---------------")


arg = sys.argv[1]
print(jieba.lcut(arg))
