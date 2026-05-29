# -*- coding: gbk -*-
import os
import cv2
import numpy as np
import pandas as pd

# ==================== 路径配置 ====================
IMAGE_DIR = r"C:/Users/33407/OneDrive/Desktop/wenjianjia/Python_task/pedestrian_dataset/pedestrian/right/JPEGImages"
CSV_PATH = r"C:/Users/33407/OneDrive/Desktop/wenjianjia/Python_task/pedestrian_dataset/pedestrian/right/annotations_final.csv"

# ==================== 输出设置 ====================
OUTPUT_IMAGE_DIR = r"C:/Users/33407/OneDrive/Desktop/low_light_images"
OUTPUT_CSV_PATH = r"C:/Users/33407/OneDrive/Desktop/low_light_annotations.csv"

# ==================== 低光判定阈值 ====================
AVG_BRIGHT_THRESH = 100        # 亮度 < 
CONTRAST_THRESH = 40          # 对比度 < 
DARK_RATIO_THRESH = 0.55      # 暗像素占比 > 
DARK_PIXEL_LEVEL = 100         # 小于算暗像素

def judge_low_light(image_path):
    """多维度综合判定：至少满足2个条件才算低光"""
    img = cv2.imread(image_path)
    if img is None:
        return False, 0, 0, 0

    # 1. 计算亮度（HSV V通道，最准确）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[..., 2]
    avg_bright = np.mean(v_channel)

    # 2. 计算对比度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)

    # 3. 计算暗像素占比
    total_pixels = gray.shape[0] * gray.shape[1]
    dark_pixels = np.sum(gray < DARK_PIXEL_LEVEL)
    dark_ratio = dark_pixels / total_pixels

    # 满足 >=2 个条件才算低光
    cond1 = avg_bright < AVG_BRIGHT_THRESH
    cond2 = contrast < CONTRAST_THRESH
    cond3 = dark_ratio > DARK_RATIO_THRESH

    satisfied_conditions = sum([cond1, cond2, cond3])
    is_low = satisfied_conditions >= 2

    return is_low, avg_bright, contrast, dark_ratio

# ==================== 主程序 ====================
print("正在读取标注文件...")
df = pd.read_csv(CSV_PATH, header=None, names=['filename', 'x1', 'y1', 'x2', 'y2'])
print(f"已加载 {len(df)} 条标注，开始筛选...")

low_light_rows = []

try:
    for idx, row in df.iterrows():
        filename = row['filename']
        img_path = os.path.join(IMAGE_DIR, filename)

        if not os.path.exists(img_path):
            print(f"图片不存在: {img_path}，跳过")
            continue

        is_low, bright, cont, dark_r = judge_low_light(img_path)

        if is_low:
            low_light_rows.append(row)
            os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
            dst_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
            if not os.path.exists(dst_path):
                img = cv2.imread(img_path)
                cv2.imwrite(dst_path, img)

except KeyboardInterrupt:
    print("\n手动中断，已保存当前筛选结果...")

if low_light_rows:
    low_light_df = pd.DataFrame(low_light_rows)
    low_light_df.to_csv(OUTPUT_CSV_PATH, index=False, header=False)
    print("-" * 50)
    print(f"? 筛选完成！共找到 {len(low_light_df)} 张低光图片")
    print(f"? 图片保存到：{OUTPUT_IMAGE_DIR}")
    print(f"? 标注保存到：{OUTPUT_CSV_PATH}")
else:
    print("未找到符合条件的低光图片")