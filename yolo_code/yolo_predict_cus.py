from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO(r'D:\code\cq_instance_post\yolo_code\runs\detect\train2\weights\best.pt')
results = model(r'D:\code\cq_instance_post\big_data\yolo_dataset\images\val\GGC20251010-2.5x-1-2000.png')[0]  # 取第一个结果

# 拷贝原图
img = results.orig_img.copy()

# 生成固定的类别颜色表（RGB → BGR）
num_classes = len(results.names)
colors = {
    i: tuple(np.random.randint(0, 255, 3).tolist())  # 每类一个随机颜色
    for i in range(num_classes)
}

# 手动绘制预测框与文字

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = box.conf[0].item()
    cls = int(box.cls[0])
    label = f"{results.names[cls]} {conf:.2f}"

    # 获取该类别的颜色（BGR）
    color = colors[cls]

    # 绘制矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 绘制标签文字
    font_scale = 1
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    #cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)  # 背景条
    cv2.putText(img, label, (x1-230, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    

# 保存结果图像
cv2.imwrite("output_colored.jpg", img)
print("✅ 结果已保存为 output_colored.jpg")