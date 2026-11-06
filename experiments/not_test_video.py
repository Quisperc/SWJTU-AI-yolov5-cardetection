import torch
import cv2
def test_video(model_path, video_path, output_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        results.render()
        out.write(results.imgs[0])

    cap.release()
    out.release()
    print(f"视频检测结果已保存到: {output_path}")

if __name__ == '__main__':
    test_video(
        model_path='models/car_detection/weights/best.pt',
        video_path='input/video.mp4',
        output_path='output/result_video.avi'
    )
