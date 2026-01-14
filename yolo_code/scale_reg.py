import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re

# ========== 1️⃣ 加载模型 ==========
yolo_model = YOLO(r'E:\cq_instance_post\yolo_code\runs\detect\train5\weights\best.pt')  # 你的YOLOv11模型路径
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # OCR模型（可识别 μm）

# ========== 2️⃣ 推理显微镜图像 ==========
def detect_scale(image_path, conf_thresh=0.5):
    image = cv2.imread(image_path)
    results = yolo_model(image)[0]

    scale_bar_box = None
    scale_text_box = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        label = results.names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        if label == "scale_bar":
            scale_bar_box = xyxy
        elif label == "scale_text":
            scale_text_box = xyxy

    return image, scale_bar_box, scale_text_box

# ========== 3️⃣ OCR 识别比例尺文字 ==========
def recognize_scale_text(image, text_box):
    if text_box is None:
        return None
    x1, y1, x2, y2 = text_box
    crop = image[y1:y2, x1:x2]
    result = ocr.ocr(crop, cls=True)
    if not result or not result[0]:
        return None
    text = result[0][0][1][0]
    return text.strip()

# ========== 4️⃣ 提取数值和单位 ==========
def parse_scale_text(text):
    match = re.search(r"([\d\.]+)\s*([a-zA-Zμμmμμ]*)", text)
    if not match:
        return None, None
    value = float(match.group(1))
    unit = match.group(2).replace("u", "μ")
    return value, unit

# ========== 5️⃣ 计算比例（像素 / 实际长度） ==========
def compute_scale_ratio(scale_bar_box, scale_value, unit):
    if scale_bar_box is None or scale_value is None:
        return None
    x1, y1, x2, y2 = scale_bar_box
    pixel_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return scale_value / pixel_length, unit  # μm per pixel

# ========== 6️⃣ 主流程 ==========
def process_image(image_path):
    image, scale_bar_box, scale_text_box = detect_scale(image_path)
    text = recognize_scale_text(image, scale_text_box)
    if text is None:
        print("未识别到比例尺文字")
        return

    scale_value, unit = parse_scale_text(text)
    if scale_value is None:
        print("无法解析比例尺数值")
        return

    ratio, unit = compute_scale_ratio(scale_bar_box, scale_value, unit)
    if ratio is None:
        print("未检测到比例尺条")
        return

    print(f"识别结果: {scale_value} {unit}")
    print(f"比例尺长度: {ratio:.4f} {unit}/pixel")

    # 可视化检测结果
    if scale_bar_box is not None:
        cv2.rectangle(image, (scale_bar_box[0], scale_bar_box[1]), (scale_bar_box[2], scale_bar_box[3]), (0,255,0), 2)
    if scale_text_box is not None:
        cv2.rectangle(image, (scale_text_box[0], scale_text_box[1]), (scale_text_box[2], scale_text_box[3]), (255,0,0), 2)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========== 示例调用 ==========
if __name__ == "__main__":
    process_image(r'E:\cq_instance_post\big_data\yolo_dataset\images\val\20250107-2.jpg')
