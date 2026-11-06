import os
import pandas as pd


def generate_txt_from_csv(csv_file, image_dir, label_dir):
    """
    根据 CSV 文件生成 YOLO 格式的 .txt 标签文件。

    参数:
    - csv_file: 包含标签数据的 CSV 文件路径
    - image_dir: 图片文件夹路径，用于检查图片存在性
    - label_dir: 输出标签文件的目标路径
    """
    # 创建标签输出目录
    os.makedirs(label_dir, exist_ok=True)

    # 读取 CSV 文件
    df = pd.read_csv(csv_file)  # 自动识别逗号分隔符

    for _, row in df.iterrows():
        # 提取图片名称
        image_name = row['image_name']
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))

        # 检查图片是否存在（可选）
        if not os.path.exists(os.path.join(image_dir, image_name)):
            print(f"Warning: Image file {image_name} not found in {image_dir}. Skipping...")
            continue

        # 提取标签信息
        class_id = 0  # 假设只有一个类别 'car'
        x_center = row['x_center']
        y_center = row['y_center']
        width = row['width']
        height = row['height']

        # 写入标签文件
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"完成：所有标签文件已生成到 {label_dir}")

if __name__ == '__main__':
    # 示例调用
    generate_txt_from_csv(
    csv_file='my_dataset/labels/train.csv',
    image_dir='my_dataset/images/train',
    label_dir='my_dataset/labels/train'
)

    generate_txt_from_csv(
    csv_file='my_dataset/labels/val.csv',
    image_dir='my_dataset/images/val',
    label_dir='my_dataset/labels/val'
)
