# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import shutil

print("="*60)
print("拥挤/稀疏场景分类脚本")
print("="*60)

# ==================== 路径配置（改成你的路径） ====================
BASE_DIR = r"D:\StudyinG\PythonStudy\merged_dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CSV_PATH = os.path.join(BASE_DIR, "merged_annotations.csv")

# ==================== 输出路径 ====================
OUTPUT_DIR = r"D:\StudyinG\PythonStudy\test_scenes"
OUTPUT_CROWDED_DIR = os.path.join(OUTPUT_DIR, "crowded")
OUTPUT_SPARSE_DIR = os.path.join(OUTPUT_DIR, "sparse")

# ==================== 拥挤判断阈值 ====================
CROWDED_THRESHOLD = 4        # 行人数量 ≥ 4 为拥挤场景

# 创建输出目录
os.makedirs(OUTPUT_CROWDED_DIR, exist_ok=True)
os.makedirs(OUTPUT_SPARSE_DIR, exist_ok=True)

print(f"图片目录: {IMAGE_DIR}")
print(f"CSV文件: {CSV_PATH}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"拥挤阈值: ≥ {CROWDED_THRESHOLD} 人")

# ==================== 读取CSV并统计每张图片的人数 ====================
print("\n正在读取标注文件...")
df = pd.read_csv(CSV_PATH, header=None, names=['filename', 'x1', 'y1', 'x2', 'y2'])
print(f"已加载 {len(df)} 条标注记录")

# 统计每张图片的行人数量
person_count = df.groupby('filename').size()
print(f"总图片数: {len(person_count)}")

# ==================== 分类 ====================
print("\n开始筛选拥挤/稀疏场景...")

crowded_images = set()
sparse_images = set()
crowded_rows = []
sparse_rows = []

for filename, count in person_count.items():
    img_path = os.path.join(IMAGE_DIR, filename)
    
    if not os.path.exists(img_path):
        print(f"图片不存在: {filename}，跳过")
        continue
    
    if count >= CROWDED_THRESHOLD:
        crowded_images.add(filename)
        # 复制图片
        dst_path = os.path.join(OUTPUT_CROWDED_DIR, filename)
        if not os.path.exists(dst_path):
            shutil.copy2(img_path, dst_path)
    else:
        sparse_images.add(filename)
        # 复制图片
        dst_path = os.path.join(OUTPUT_SPARSE_DIR, filename)
        if not os.path.exists(dst_path):
            shutil.copy2(img_path, dst_path)

# ==================== 根据分类结果整理 CSV 标注 ====================
print("\n正在整理标注文件...")

for idx, row in df.iterrows():
    filename = row['filename']
    if filename in crowded_images:
        crowded_rows.append(row)
    elif filename in sparse_images:
        sparse_rows.append(row)

# ==================== 保存 CSV 标注文件 ====================
crowded_csv_path = os.path.join(OUTPUT_DIR, "crowded_annotations.csv")
sparse_csv_path = os.path.join(OUTPUT_DIR, "sparse_annotations.csv")

if crowded_rows:
    crowded_df = pd.DataFrame(crowded_rows)
    crowded_df.to_csv(crowded_csv_path, index=False, header=False)
    
if sparse_rows:
    sparse_df = pd.DataFrame(sparse_rows)
    sparse_df.to_csv(sparse_csv_path, index=False, header=False)

# ==================== 输出结果 ====================
print("-" * 50)
print(f"筛选完成！")
print(f"\n拥挤场景（≥{CROWDED_THRESHOLD}人）:")
print(f"  图片数: {len(crowded_images)} 张")
print(f"  标注记录数: {len(crowded_rows)} 条")
print(f"  图片保存到: {OUTPUT_CROWDED_DIR}")
print(f"  标注保存到: {crowded_csv_path}")
print(f"\n稀疏场景（<{CROWDED_THRESHOLD}人）:")
print(f"  图片数: {len(sparse_images)} 张")
print(f"  标注记录数: {len(sparse_rows)} 条")
print(f"  图片保存到: {OUTPUT_SPARSE_DIR}")
print(f"  标注保存到: {sparse_csv_path}")

# ==================== 生成统计报告 ====================
report = f"""
========================================
        拥挤/稀疏场景分类报告
========================================

一、分类标准
----------------------------------------
拥挤场景: 行人数量 ≥ {CROWDED_THRESHOLD} 人
稀疏场景: 行人数量 < {CROWDED_THRESHOLD} 人

二、统计结果
----------------------------------------
总图片数: {len(person_count)} 张
总标注框: {len(df)} 个

拥挤场景: {len(crowded_images)} 张, {len(crowded_rows)} 个标注框
稀疏场景: {len(sparse_images)} 张, {len(sparse_rows)} 个标注框

三、输出文件
----------------------------------------
拥挤场景图片: {OUTPUT_CROWDED_DIR}
稀疏场景图片: {OUTPUT_SPARSE_DIR}
拥挤场景标注: {crowded_csv_path}
稀疏场景标注: {sparse_csv_path}

========================================
"""

report_path = os.path.join(OUTPUT_DIR, "crowded_sparse_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ 报告已保存: {report_path}")
print("="*60)
print("✅ 拥挤/稀疏场景分类完成！")