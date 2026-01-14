import glob
import os

# 标签文件目录（修改为你自己的路径）
label_dir = r"C:\Users\Keepself\OneDrive\桌面\grahpene\cq_instance_post\big_data\yolo_dataset\labels\train"



if not os.path.exists(label_dir):
    raise FileNotFoundError(f"❌ 路径不存在: {label_dir}")

label_files = glob.glob(os.path.join(label_dir, "*.txt"))
print(f"找到 {len(label_files)} 个标签文件")

# === 2️⃣ 处理文件 ===
for f in label_files:
    with open(f, "r", encoding="utf-8") as file:
        lines = file.readlines()

    new_lines = []
    changed = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(float(parts[0]))  # 防止带小数
        except:
            print(f"⚠️ 无法解析类别: {line} in {f}")
            continue

        if cls == 4:
            parts[0] = "0"
            changed = True
        elif cls == 5:
            parts[0] = "1"
            changed = True

        new_lines.append(" ".join(parts))

    if changed:
        print(f"✅ 修改文件: {os.path.basename(f)}")
        with open(f, "w", encoding="utf-8") as wf:
            wf.write("\n".join(new_lines) + "\n")

print("=== 替换完成，开始验证 ===")

# === 3️⃣ 验证最大类别编号 ===
max_class = -1
for f in label_files:
    with open(f, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                parts = line.strip().split()
                try:
                    cls = int(float(parts[0]))
                except:
                    continue
                max_class = max(max_class, cls)
print(f"最大类别编号: {max_class}")

