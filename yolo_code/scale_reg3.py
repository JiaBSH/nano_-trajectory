from unittest import result
import cv2
from matplotlib import image
import numpy as np
#from paddle import crop
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re

# ========== 1️⃣ 加载模型 ==========
yolo_model = YOLO('./runs/detect/train/weights/best.pt')  # YOLO模型路径
#ocr = PaddleOCR(use_textline_orientation=True, lang='ch')  # OCR模型（可识别 μm）
# 关闭方向检测，scale bar 一般都是水平的，不需要这个功能
ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False)
def detect_scale(image_path, conf_thresh=0.3):  # 降低置信度阈值方便调试
    image = cv2.imread(image_path)
    if image is None:
        print(f"图像路径错误或无法读取: {image_path}")
        return None, None, None

    results = yolo_model(image)[0]
    scale_bar_box = None
    scale_text_box = None

    print("\n--- YOLO 检测结果 ---")
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        print(f"类别: {label}, 置信度: {conf:.2f}, 坐标: {xyxy}")
        if conf < conf_thresh:
            continue
        if label.lower() == "scale_bar":
            scale_bar_box = xyxy
        elif label.lower() == "scale_text":
            scale_text_box = xyxy

    return image, scale_bar_box, scale_text_box

# ========== 3️⃣ OCR识别比例尺文字 ==========
def recognize_scale_text(image, text_box):
    if text_box is None:
        print("scale_text_box 为 None，无法 OCR")
        return None

    x1, y1, x2, y2 = text_box
    pad = 8
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.shape[1], x2 + pad)
    y2 = min(image.shape[0], y2 + pad)

    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, crop_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    crop_bin = cv2.cvtColor(crop_bin, cv2.COLOR_GRAY2BGR)

    result = ocr.ocr(crop_bin)
    if not result or not isinstance(result, list):
        print("OCR 未识别到文字")
        return None

    # ======== PaddleOCR 输出解析 ========
    text_candidates = []
    if isinstance(result[0], dict) and 'rec_texts' in result[0]:
        rec_texts = result[0]['rec_texts']
        rec_scores = result[0]['rec_scores']
        text_candidates = list(zip(rec_texts, rec_scores))
    else:
        try:
            for line in result[0]:
                if isinstance(line, list) and len(line) > 1:
                    txt, conf = line[1]
                    text_candidates.append((txt, conf))
        except Exception as e:
            print("无法解析 OCR 结果结构:", e)
            print("result =", result)
            return None

    if not text_candidates:
        print("未检测到有效文字")
        return None

    print(f"OCR识别结果: {text_candidates}")
    best_text, best_conf = sorted(text_candidates, key=lambda x: x[1], reverse=True)[0]
    text = best_text.strip()
    print("最终识别文字:", text)

    # ======== 自动修正规则 ========
    # PaddleOCR 常见误识别 μm -> u, rn, wr, w, m
    text = text.replace("u", "μ").replace("µ", "μ").replace("rn", "m").replace("wr", "μm").replace("w", "μ")

    # 如果只有数字或缺单位，自动补 μm
    if re.fullmatch(r"[\d\.]+", text):
        text += "μm"

    print("解析比例尺原始文字:", text)

    # ======== 提取数值和单位 ========
    match = re.search(r"([\d\.]+)\s*([a-zA-Zμ]+)?", text)
    if match:
        value = float(match.group(1))
        unit = match.group(2) if match.group(2) else "μm"  # 默认补 μm
        print(f"解析结果: 数值={value}, 单位={unit}")
        return value, unit
    else:
        print(f"无法解析文字为数值单位: {text}")
        return None


# ========== 4️⃣ 解析数值和单位 ==========
def parse_scale_text(text):
    print(f"解析比例尺原始文字: {text}")
    if not isinstance(text, str):
        text = str(text)
    match = re.search(r"([\d\.]+)\s*([a-zA-Zμum]+)?", text)
    if match:
        value = float(match.group(1))
        unit = match.group(2).replace("u", "μ").replace("µ", "μ") if match.group(2) else ""
        print(f"解析结果: 数值={value}, 单位={unit}")
        return value, unit
    else:
        print(f"无法解析文字为数值单位: {text}")
        return None, None

# ========== 5️⃣ 计算比例（μm/pixel） ==========
def compute_scale_ratio(scale_bar_box, scale_value):
    if scale_bar_box is None or scale_value is None:
        return None
    x1, y1, x2, y2 = scale_bar_box
    pixel_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    ratio = scale_value / pixel_length
    return ratio, pixel_length

# ========== 6️⃣ 主流程 ==========
def process_image(image_path):
    image, scale_bar_box, scale_text_box = detect_scale(image_path)
    if image is None:
        return

    print("检测到的scale_bar_box:", scale_bar_box)
    print("检测到的scale_text_box:", scale_text_box)

    text = recognize_scale_text(image, scale_text_box)
    if text is None:
        print("未识别到比例尺文字")
        return

    scale_value, unit = parse_scale_text(text)
    if scale_value is None:
        return

    ratio, pixel_length = compute_scale_ratio(scale_bar_box, scale_value)
    if ratio is None:
        print("未检测到比例尺条")
        return

    print(f"\n=== 最终结果 ===")
    print(f"识别结果: {scale_value} {unit}")
    print(f"比例尺长度: {pixel_length:.2f} pixels")
    print(f"像素比例: {ratio:.4f} {unit}/pixel")

    # 可视化检测结果
    if scale_bar_box is not None:
        cv2.rectangle(image, (scale_bar_box[0], scale_bar_box[1]),
                             (scale_bar_box[2], scale_bar_box[3]), (0,255,0), 2)
        cv2.putText(image, "Scale Bar", (scale_bar_box[0], scale_bar_box[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    if scale_text_box is not None:
        cv2.rectangle(image, (scale_text_box[0], scale_text_box[1]),
                             (scale_text_box[2], scale_text_box[3]), (255,0,0), 2)
        cv2.putText(image, str(text), (scale_text_box[0], scale_text_box[1]-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    cv2.imwrite("../temp_data/scale.jpg", image)


# ========== 示例调用 ==========
if __name__ == "__main__":
    #process_image(r'E:\G_data\畴区\畴区\02-散图\畴区光镜数据20x-2\20250107-26.jpg')
    process_image('../data/frame/11dd74426e8374ac110c4036c77c09ab_000000000000.jpg')