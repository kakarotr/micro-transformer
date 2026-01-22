from wiki.utils import to_simplified

categories = [
    "戰國大名",
    "戰國武將",
    "戰國時代_(日本)",
    "戰國時代戰役_(日本)",
    "戰國時代事件_(日本)",
    "安土桃山時代",
    "安土桃山時代戰役",
    "安土桃山時代事件",
    "武家政權",
    "室町幕府",
    "室町幕府官制",
    "關東公方",
    "大坂之役",
]

tc_ignore_sections = [
    "注釋",
    "參見",
    "註釋",
    "徵引",
    "腳註",
    "來源",
    "引用",
    "書目",
    "註腳",
    "參看",
    "登場作品",
    "关联作品",
    "关联项目",
    "來源文獻",
    "參考資料",
    "外部連結",
    "參考文獻",
    "關聯項目",
    "影視改編",
    "基礎資訊",
    "進階資料",
    "參考連結",
    "相關條目",
]
ignore_sections = [to_simplified(item) for item in tc_ignore_sections]
ignore_sections.extend(tc_ignore_sections)

fuzzy_sections = ["參考", "連結", "文獻", "参考", "链接", "文献", "外部"]

replace_links = ("file:", "image:", "category:", "檔案:", "文件:", "分類:", "分类:")
