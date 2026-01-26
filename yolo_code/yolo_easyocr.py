import cv2
import numpy as np
from ultralytics import YOLO
import easyocr  # <--- 1. 改用 EasyOCR
import re
import os
import json
import csv
from datetime import datetime
from pathlib import Path
import argparse

# ========== 1️⃣ 加载模型 ==========
# YOLO 继续用，它工作得很完美
yolo_model = YOLO('./runs/detect/train/weights/best.pt') 

# 初始化 EasyOCR (复用 PyTorch 环境)
# gpu=False: 强制用 CPU，稳定且对小图足够快
reader = easyocr.Reader(['en'], gpu=False) 

# ========== 2️⃣ YOLO检测比例尺 ==========
def detect_scale(image_path, conf_thresh=0.3): 
    image = cv2.imread(str(image_path))
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
            # JSON 友好：转成普通 list
            "xyxy": [int(v) for v in xyxy],
        })
        if label.lower() == "scale_bar":
            scale_bar_box = xyxy
        elif label.lower() == "scale_text":
            scale_text_box = xyxy

    # scale_*_box 同样转成 list[int]
    if scale_bar_box is not None:
        scale_bar_box = [int(v) for v in scale_bar_box]
    if scale_text_box is not None:
        scale_text_box = [int(v) for v in scale_text_box]

    return image, scale_bar_box, scale_text_box, detections


def _default_output_dir() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    out_dir = os.path.join(repo_root, "temp")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _annotate_and_save(image, image_path: str, detections, scale_text: str | None = None, out_dir: str | None = None):
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

    out_dir = out_dir or _default_output_dir()
    annotated_dir = _ensure_dir(os.path.join(out_dir, "annotated"))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(annotated_dir, f"{stem}_scale_bar.jpg")
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

    if crop.size == 0:
        print("⚠️ OCR 裁剪区域为空")
        return None

    # 图像增强：放大、灰度、阈值/自适应阈值、反相，多路尝试提高数字识别成功率
    crop_up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_otsu_inv = cv2.bitwise_not(bin_otsu)
    bin_adapt = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    bin_adapt_inv = cv2.bitwise_not(bin_adapt)

    variants = [
        ("bgr", crop_up),
        ("gray", gray),
        ("otsu", bin_otsu),
        ("otsu_inv", bin_otsu_inv),
        ("adapt", bin_adapt),
        ("adapt_inv", bin_adapt_inv),
    ]

    digits_allow = "0123456789.,"
    unit_allow = "μµ㎛uUmMnN"
    full_allow = digits_allow + unit_allow + " "

    def _run_readtext(img, allow: str):
        try:
            return reader.readtext(img, allowlist=allow)
        except Exception as e:
            print(f"❌ EasyOCR 报错({allow=}): {e}")
            return []

    def _pick_best(items, prefer_digits: bool):
        # items: list of (bbox, text, conf)
        best = None
        best_score = -1.0
        for (_bbox, txt, conf) in items or []:
            t = (txt or "").strip()
            if not t:
                continue
            digit_count = len(re.findall(r"\d", t))
            has_unit = bool(re.search(r"(?i)(nm|mm|μm|um|μ|µ)", t))
            # prefer_digits 时，数字数量权重更高
            score = (digit_count * (50 if prefer_digits else 10)) + (5 if has_unit else 0) + float(conf)
            if score > best_score:
                best_score = score
                best = (t, float(conf), digit_count, has_unit)
        return best

    best_number = None  # (text, conf, digit_count, has_unit)
    best_unit = None
    best_full = None

    for name, img_var in variants:
        # 1) 数字优先：尽量抓到 1000 / 500 / 200 等
        r_num = _run_readtext(img_var, allow=digits_allow)
        if r_num:
            pick = _pick_best(r_num, prefer_digits=True)
            if pick and (best_number is None or pick[2] > best_number[2] or (pick[2] == best_number[2] and pick[1] > best_number[1])):
                best_number = pick

        # 2) 单位优先：抓到 μm / nm / mm
        r_unit = _run_readtext(img_var, allow=unit_allow)
        if r_unit:
            pick = _pick_best(r_unit, prefer_digits=False)
            if pick and (best_unit is None or pick[1] > best_unit[1]):
                best_unit = pick

        # 3) 全量：备用（数字+单位一起）
        r_full = _run_readtext(img_var, allow=full_allow)
        if r_full:
            pick = _pick_best(r_full, prefer_digits=True)
            if pick and (best_full is None or pick[2] > best_full[2] or (pick[2] == best_full[2] and pick[1] > best_full[1])):
                best_full = pick

        if (best_full and best_full[2] >= 2) or (best_number and best_number[2] >= 2 and best_unit):
            # 已经有较可靠的数字了，提前结束
            break

    # 组合策略：有数字 + 有单位则拼起来；否则退回 best_full 或 best_number
    chosen = None
    if best_number and best_unit:
        num_txt = best_number[0]
        unit_txt = best_unit[0]
        chosen = f"{num_txt}{unit_txt}"
        print(f"最终识别文字(数字+单位组合): {chosen}")
        return chosen

    if best_full:
        chosen = best_full[0]
        print(f"最终识别文字(全量最优): {chosen}")
        return chosen

    if best_number:
        chosen = best_number[0]
        print(f"最终识别文字(仅数字): {chosen}")
        return chosen

    print("⚠️ OCR 未识别到有效文字")
    return None

# ========== 4️⃣ 解析数值和单位 ==========
def parse_scale_text(text):
    if not text:
        return None, None

    raw = str(text).strip()

    # 归一化：处理多种微米写法、空格、逗号、常见误识别
    s = raw
    s = s.replace("µ", "μ")
    s = s.replace("㎛", "μm")
    s = s.replace("rn", "m")
    # 去掉常见分隔符（OCR 可能输出 1,000 或 1，000）
    s = s.replace(",", "").replace("，", "")
    # 把 'um' / 'u m' 统一成 'μm'
    s = re.sub(r"(?i)u\s*m", "μm", s)
    # 去掉多余空白
    s = re.sub(r"\s+", "", s)

    # 如果只有数字（含小数），默认单位 μm
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        s = f"{s}μm"

    print(f"解析比例尺原始文字: {raw} -> {s}")

    # 提取：数值 + 单位（支持 nm / μm / mm / um）
    m = re.search(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>nm|mm|μm|um)?", s, flags=re.IGNORECASE)
    if not m:
        print(f"❌ 无法解析文字为数值单位: {raw}")
        return None, None

    value_str = m.group("value")
    unit_str = m.group("unit") or "μm"

    try:
        value = float(value_str)
    except Exception:
        print(f"❌ 数值转换失败: {value_str} (raw={raw})")
        return None, None

    unit_lower = unit_str.lower()
    if unit_lower == "nm":
        unit = "nm"
    elif unit_lower == "mm":
        unit = "mm"
    else:
        # 包括 μm / um
        unit = "μm"

    print(f"✅ 解析结果: 数值={value}, 单位={unit}")
    return value, unit

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


def _is_image_file(path: Path, exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def process_folder(
    input_dir: str,
    out_dir: str | None = None,
    conf_thresh: float = 0.3,
    recursive: bool = False,
    write_csv: bool = True,
):
    in_dir = Path(input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    out_dir = out_dir or _default_output_dir()
    _ensure_dir(out_dir)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    iterator = in_dir.rglob("*") if recursive else in_dir.glob("*")
    image_paths = [p for p in iterator if _is_image_file(p, exts)]
    image_paths.sort()

    results_json_path = os.path.join(out_dir, "results.json")
    results_csv_path = os.path.join(out_dir, "results.csv")

    all_rows: list[dict] = []
    print(f"\n=== 批量处理开始 ===")
    print(f"输入文件夹: {in_dir}")
    print(f"图片数量:   {len(image_paths)}")
    print(f"输出目录:   {out_dir}")

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] 处理: {img_path}")
        row: dict = {
            "image": str(img_path),
            "annotated_image": None,
            "scale_value": None,
            "unit": None,
            "pixel_length": None,
            "ratio": None,
            "detections": [],
            "error": None,
        }

        try:
            image, scale_bar_box, scale_text_box, detections = detect_scale(str(img_path), conf_thresh=conf_thresh)
            row["detections"] = detections

            if image is None:
                row["error"] = "image_read_failed"
                all_rows.append(row)
                continue

            scale_value = None
            unit = "μm"
            ocr_text_for_label = None

            if scale_text_box is not None:
                text = recognize_scale_text(image, scale_text_box)
                scale_value, unit = parse_scale_text(text)

            if scale_value is None:
                row["error"] = "scale_text_parse_failed"
                # 依然把检测框画出来，方便排查
                row["annotated_image"] = _annotate_and_save(image, str(img_path), detections, scale_text=None, out_dir=out_dir)
                all_rows.append(row)
                continue

            ratio, pixel_length = compute_scale_ratio(scale_bar_box, scale_value)
            if ratio is None:
                row["error"] = "scale_bar_not_found"
                row["annotated_image"] = _annotate_and_save(image, str(img_path), detections, scale_text=f"{scale_value}{unit}", out_dir=out_dir)
                all_rows.append(row)
                continue

            row["scale_value"] = float(scale_value)
            row["unit"] = unit
            row["pixel_length"] = float(pixel_length) if pixel_length is not None else None
            row["ratio"] = float(ratio)

            ocr_text_for_label = f"{scale_value}{unit}"
            row["annotated_image"] = _annotate_and_save(image, str(img_path), detections, scale_text=ocr_text_for_label, out_dir=out_dir)

        except Exception as e:
            row["error"] = f"exception: {type(e).__name__}: {e}"

        all_rows.append(row)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(in_dir),
        "out_dir": str(out_dir),
        "count": len(all_rows),
        "results": all_rows,
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if write_csv:
        with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "annotated_image",
                    "scale_value",
                    "unit",
                    "pixel_length",
                    "ratio",
                    "error",
                ],
            )
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k) for k in writer.fieldnames})

    print("\n=== 批量处理完成 ===")
    print(f"结果JSON: {results_json_path}")
    if write_csv:
        print(f"结果CSV:  {results_csv_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO + EasyOCR 批量识别比例尺并保存标注结果")
    parser.add_argument("--input_dir", type=str, required=True, help="待处理图片文件夹")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认: yolo_code/temp）")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO 置信度阈值")
    parser.add_argument("--recursive", action="store_true", help="递归遍历子目录")
    parser.add_argument("--no_csv", action="store_true", help="不输出 results.csv，仅输出 results.json")
    return parser

if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    process_folder(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        conf_thresh=args.conf,
        recursive=args.recursive,
        write_csv=not args.no_csv,
    )