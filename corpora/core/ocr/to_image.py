import os

import fitz  # PyMuPDF
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)


def convert_pdf_to_images(
    pdf_path,
    output_dir="output_images",
    zoom=2,
    top_crop=0,
    bottom_crop=0,
    left_crop=0,
    right_crop=0,
):
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

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(), MofNCompleteColumn()
    ) as progress:
        task_id = progress.add_task(name, total=len(pdf_document))

        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)

            rect = page.rect
            new_x0 = rect.x0 + left_crop
            new_y0 = rect.y0 + top_crop
            new_x1 = rect.x1 - right_crop
            new_y1 = rect.y1 - bottom_crop

            new_x0 = min(new_x0, new_x1)
            new_y0 = min(new_y0, new_y1)

            new_rect = fitz.Rect(new_x0, new_y0, new_x1, new_y1)
            page.set_cropbox(new_rect)

            # 将页面渲染为像素图 (Pixmap)
            # alpha=False 表示不使用透明通道（即白色背景）
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # 拼接输出文件名
            image_filename = f"page_{page_number + 1}.png"
            image_path = os.path.join(output_dir, image_filename)

            # 保存图片
            pix.save(image_path)
            progress.update(task_id, advance=1)

    # 6. 关闭文档
    pdf_document.close()


# --- 使用示例 ---
if __name__ == "__main__":
    # 在这里输入你的 PDF 文件名
    for name in [
        "岩波日本史",
    ]:
        convert_pdf_to_images(f"preview/pdfs/{name}.pdf", output_dir=f"preview/pdf_images/{name}", zoom=3)
