from unittest import result
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr  # <--- 1. 改用 EasyOCR
import re
import os

# ========== 1️⃣ 加载模型 ==========
# YOLO 继续用，它工作得很完美
yolo_model = YOLO('./runs/detect/train/weights/best.pt') 

# 初始化 EasyOCR (复用 PyTorch 环境)
# gpu=False: 强制用 CPU，稳定且对小图足够快
reader = easyocr.Reader(['en'], gpu=False) 

# ========== 2️⃣ YOLO检测比例尺 ==========
def detect_scale(image_path, conf_thresh=0.3): 
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 图像路径错误或无法读取: {image_path}")
        return None, None, None, []

    results = yolo_model(image)[0]
    scale_bar_box = None
    scale_text_box = None
    detections = []

    print("\n--- YOLO 检测结果 ---")
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        print(f"类别: {label}, 置信度: {conf:.2f}, 坐标: {xyxy}")
        
        if conf < conf_thresh:
            continue

        detections.append({
            "label": str(label),
            "conf": conf,
            "xyxy": xyxy,
        })
        if label.lower() == "scale_bar":
            scale_bar_box = xyxy
        elif label.lower() == "scale_text":
            scale_text_box = xyxy

    return image, scale_bar_box, scale_text_box, detections


def _ensure_output_dir() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "temp")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _annotate_and_save(image, image_path: str, detections, scale_text: str | None = None):
    if image is None:
        return None

    annotated = image.copy()

    for det in detections or []:
        label = det.get("label", "")
        conf = det.get("conf", 0.0)
        xyxy = det.get("xyxy", None)
        if xyxy is None or len(xyxy) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        color = (0, 255, 0) if label.lower() == "scale_bar" else (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}".strip()
        if label.lower() == "scale_text" and scale_text:
            text = f"{text}: {scale_text}" if text else scale_text

        if text:
            y_text = max(0, y1 - 8)
            cv2.putText(
                annotated,
                text,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA,
            )

    out_dir = _ensure_output_dir()
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_annotated.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"✅ 标注结果已保存: {out_path}")
    return out_path

# ========== 3️⃣ EasyOCR 识别比例尺文字 ==========
def recognize_scale_text(image, text_box):
    if text_box is None:
        print("⚠️ scale_text_box 为 None，无法 OCR")
        return None

    x1, y1, x2, y2 = text_box
    # 增加 padding 防止文字贴边
    pad = 5
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = image[y1:y2, x1:x2]
    
    # 图像增强：放大2倍，二值化 (这对 EasyOCR 也很重要)
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, crop_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # EasyOCR 可以直接吃 numpy 数组
    try:
        # allowlist: 只允许识别数字和常见单位字母，排除干扰
        result = reader.readtext(crop_bin, allowlist='0123456789umNMnm. ')
    except Exception as e:
        print(f"❌ EasyOCR 报错: {e}")
        return None

    if not result:
        print("⚠️ OCR 未识别到文字")
        return None

    print(f"EasyOCR 原始结果: {result}")
    
    # EasyOCR 返回格式: [([[x,y]..], 'text', conf), ...]
    # 我们取置信度最高的一个
    best_result = sorted(result, key=lambda x: x[2], reverse=True)[0]
    text = best_result[1].strip()
    print("最终识别文字:", text)

    return text

# ========== 4️⃣ 解析数值和单位 ==========
def parse_scale_text(text):
    if not text: return None, None
    
    # 自动修正规则
    text = text.replace("u", "μ").replace("µ", "μ").replace("rn", "m")
    
    # 如果只有数字，强制补 μm
    if re.fullmatch(r"[\d\.]+", text):
        text += "μm"
        
    print(f"解析比例尺原始文字: {text}")

    # 提取数值和单位
    match = re.search(r"([\d\.]+)\s*([a-zA-Zμ]+)?", text)
    if match:
        value = float(match.group(1))
        unit = match.group(2) if match.group(2) else "μm"
        # 统一单位写法
        if "nm" in unit.lower(): unit = "nm"
        elif "mm" in unit.lower(): unit = "mm"
        else: unit = "μm"
            
        print(f"✅ 解析结果: 数值={value}, 单位={unit}")
        return value, unit
    else:
        print(f"❌ 无法解析文字为数值单位: {text}")
        return None, None

# ========== 5️⃣ 计算比例（μm/pixel） ==========
def compute_scale_ratio(scale_bar_box, scale_value):
    if scale_bar_box is None or scale_value is None:
        return None, None
    x1, y1, x2, y2 = scale_bar_box
    # 计算像素长度
    pixel_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if pixel_length == 0: return None, None
    
    ratio = scale_value / pixel_length
    return ratio, pixel_length

# ========== 6️⃣ 主流程 ==========
def process_image(image_path):
    image, scale_bar_box, scale_text_box, detections = detect_scale(image_path)
    if image is None: return

    # 如果检测到了文字框，就去识别
    scale_value = None
    unit = "μm"
    
    if scale_text_box is not None:
        text = recognize_scale_text(image, scale_text_box)
        scale_value, unit = parse_scale_text(text)
    
    if scale_value is None:
        print("❌ 无法获取比例尺数值")
        return

    ratio, pixel_length = compute_scale_ratio(scale_bar_box, scale_value)
    if ratio is None:
        print("❌ 未检测到比例尺条")
        return

    print(f"\n=== 🎉 最终结果 🎉 ===")
    print(f"物理数值: {scale_value} {unit}")
    print(f"条长度:   {pixel_length:.2f} pixels")
    print(f"像素比例: {ratio:.6f} {unit}/pixel")

    # 在原图上画出识别出的检测框，并保存到 ./temp
    ocr_text_for_label = None
    if scale_value is not None:
        ocr_text_for_label = f"{scale_value}{unit}"
    _annotate_and_save(image, image_path, detections, scale_text=ocr_text_for_label)

if __name__ == "__main__":
    process_image(r'D:\code\bl0116\big_data\cq_data\20x\image\20x-1.png')