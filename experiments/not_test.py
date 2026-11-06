import torch
import cv2

def test_model(model_path, image_path, output_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    img = cv2.imread(image_path)

    results = model(img)
    results.render()  # 在图片上绘制检测结果

    cv2.imwrite(output_path, results.imgs[0])
    print(f"检测结果已保存到: {output_path}")

if __name__ == '__main__':
    test_model(
        model_path='models/car_detection/weights/best.pt',
        image_path='my_dataset/images/val/img1.jpg',
        output_path='output/result.jpg'
    )
