from ultralytics import YOLO
if __name__ == "__main__":
    # 加载预训练模型（YOLOv8n）
    model = YOLO("/home/bingxing2/home/scx9878/JiaBSH/cq_instance_post/yolo_code/yolov8n.pt")

    # 训练
    model.train(
        data="/home/bingxing2/home/scx9878/JiaBSH/cq_instance_post/data.yaml",
        epochs=80,
        imgsz=2048,
        batch=4,
        workers=0
    )

    # 训练完成后权重路径在：
    # runs/detect/scale_detector/weights/best.pt
