import pandas as pd
import matplotlib.pyplot as plt
import os

# ========== 1. 只填拥挤和稀疏 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paths = {
    "Crowded": os.path.join(BASE_DIR, "runs/lr0.001_e30_iou0.7_crowded/results.csv"),
    "Sparse": os.path.join(BASE_DIR, "runs/lr0.001_e30_iou0.7_sparse/results.csv")
}

# 检查文件
for name, p in paths.items():
    if not os.path.exists(p):
        print("❌ 找不到：", name, p)
        exit()

# ========== 2. 读数据 ==========
data = {}
for name, p in paths.items():
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    data[name] = df

# ========== 3. 画图：2行2列，只两条线 ==========
plt.figure(figsize=(12, 9))

# mAP@0.5
plt.subplot(2, 2, 1)
for name, df in data.items():
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label=name)
plt.title("mAP@0.5")
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5")
plt.legend()
plt.grid(True)

# mAP@0.5:0.95
plt.subplot(2, 2, 2)
for name, df in data.items():
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label=name)
plt.title("mAP@0.5:0.95")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.legend()
plt.grid(True)

# Precision
plt.subplot(2, 2, 3)
for name, df in data.items():
    plt.plot(df["epoch"], df["metrics/precision(B)"], label=name)
plt.title("Precision")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)

# Recall
plt.subplot(2, 2, 4)
for name, df in data.items():
    plt.plot(df["epoch"], df["metrics/recall(B)"], label=name)
plt.title("Recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)

plt.tight_layout()
out_img = os.path.join(BASE_DIR, "crowded_vs_sparse.png")
plt.savefig(out_img, dpi=300)
plt.close()

# ========== 4. 输出最终指标表格 ==========
rows = []
for name, df in data.items():
    last = df.iloc[-1]
    rows.append({
        "场景": name,
        "mAP@0.5": round(last["metrics/mAP50(B)"], 4),
        "mAP@0.5:0.95": round(last["metrics/mAP50-95(B)"], 4),
        "Precision": round(last["metrics/precision(B)"], 4),
        "Recall": round(last["metrics/recall(B)"], 4)
    })
tbl = pd.DataFrame(rows)
print("===== 拥挤 vs 稀疏 最终指标 =====")
print(tbl.to_string(index=False))
tbl.to_csv(os.path.join(BASE_DIR, "crowded_vs_sparse.csv"), index=False)

print("\n✅ 完成：")
print("图：", out_img)