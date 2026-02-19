import os

import fitz  # PyMuPDF


def convert_pdf_to_images(pdf_path, output_dir="output_images", zoom=2):
    """
    将 PDF 的每一页转换为图片
    :param pdf_path: PDF 文件路径
    :param output_dir: 图片保存目录
    :param zoom: 缩放倍数，2 表示放大 2 倍（增加清晰度）
    """
    # 1. 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误: 找不到文件 {pdf_path}")
        return

    # 2. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 打开 PDF 文件
    pdf_document = fitz.open(pdf_path)

    # 4. 设置缩放矩阵 (控制清晰度)
    # Matrix(x, y) 代表水平和垂直方向的缩放
    mat = fitz.Matrix(zoom, zoom)

    print(f"开始转换: {pdf_path}，总计 {len(pdf_document)} 页")

    # 5. 逐页处理
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # 将页面渲染为像素图 (Pixmap)
        # alpha=False 表示不使用透明通道（即白色背景）
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # 拼接输出文件名
        image_filename = f"page_{page_number + 1}.png"
        image_path = os.path.join(output_dir, image_filename)

        # 保存图片
        pix.save(image_path)
        print(f"进度: [{page_number + 1}/{len(pdf_document)}] 已保存 {image_filename}")

    # 6. 关闭文档
    pdf_document.close()
    print("转换完成！")


# --- 使用示例 ---
if __name__ == "__main__":
    # 在这里输入你的 PDF 文件名
    for name in [
        "最好看的日本战国史1：英雄黎明",
        "最好看的日本战国史2：将星纵横",
        "最好看的日本战国史3：天下棋峙",
        "最好看的日本战国史4：第六天魔王",
        "最好看的日本战国史5：太阁青云",
        "最好看的日本战国史6：八屿混一",
    ]:
        convert_pdf_to_images(f"preview/pdfs/{name}.pdf", output_dir=f"preview/pdf_images/{name}", zoom=3)
