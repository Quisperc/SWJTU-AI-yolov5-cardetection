import os
import pandas as pd
from PIL import Image


def create_labels_csv(image_dir, output_csv):
    """
    遍历图片文件夹，将每张图片的标签信息保存到一个 CSV 文件中。

    参数：
    - image_dir: 图片文件夹路径
    - output_csv: 输出的 CSV 文件路径
    """
    data = []  # 存储所有图片的标签数据

    # 遍历图片文件夹
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)

            # 获取图片尺寸
            with Image.open(img_path) as img:
                width, height = img.size

            # 假设整个图片是一个车的边界框（YOLO 格式）
            x_center = 0.5
            y_center = 0.5
            bbox_width = 1.0
            bbox_height = 1.0

            # 添加到数据列表
            data.append({
                'image_name': img_file,
                'x_center': x_center,
                'y_center': y_center,
                'width': bbox_width,
                'height': bbox_height,
                'image_width': width,
                'image_height': height
            })

    # 转换为 Pandas DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"标签已保存到: {output_csv}")


# 示例用法
if __name__ == '__main__':
    # 生成训练集和验证集的标签
    create_labels_csv('dataset/images/train', 'dataset/labels/train.csv')
    create_labels_csv('dataset/images/val', 'dataset/labels/val.csv')
