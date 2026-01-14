from ultralytics import YOLO
import cv2
import numpy as np
import os

# 模型路径
model = YOLO(r'D:\code\cq_instance_post\yolo_code\runs\detect\train2\weights\best.pt')

# 输入图片文件夹
input_folder = r'D:\石墨烯原始数据\畴区\04-畴区+双层'
# 输出文件夹
output_folder = r'D:\code\cq_instance_post\temp_data\yolo_output_images_04-畴区+双层'
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中所有图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        results = model(img_path)[0]  # 取第一个结果

        # 拷贝原图
        img = results.orig_img.copy()

        # 生成固定的类别颜色表（RGB → BGR）
        num_classes = len(results.names)
        colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(num_classes)}

        # 手动绘制预测框与文字
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = f"{results.names[cls]} {conf:.2f}"

            color = colors[cls]

            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签文字
            font_scale = 1
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # 保存结果图像
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img)
        print(f"✅ {filename} 已处理并保存")
