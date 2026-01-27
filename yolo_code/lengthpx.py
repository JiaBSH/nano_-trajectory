import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
df = pd.read_csv(r"D:\code\nanojccode\data\nanoframes\scalebar_mauel.csv")

# 帧数：1–1058（如果行数正好是 1058）
frames = range(1, len(df) + 1)

plt.figure(figsize=(8, 4))
plt.plot(frames, df["pixel_length"])
plt.xlabel("Frame")
plt.ylabel("Pixel Length (px)")
plt.title("Pixel Length vs Frame")
plt.tight_layout()
plt.show()
