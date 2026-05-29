# merge_datasets_no_pandas.py
# 纯Python实现，不需要pandas

import os
import shutil
import csv

print("="*60)
print("合并三人标注数据集 (无pandas版)")
print("="*60)

# ========== 配置路径 ==========
BASE_DIR = r"D:\StudyinG\PythonStudy"

# 三个人的文件夹路径
PERSON_A_DIR = os.path.join(BASE_DIR, 'person_A')
PERSON_B_DIR = os.path.join(BASE_DIR, 'person_B')
PERSON_C_DIR = os.path.join(BASE_DIR, 'person_C')

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'merged_dataset')
OUTPUT_IMAGES = os.path.join(OUTPUT_DIR, 'images')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'merged_annotations.csv')

os.makedirs(OUTPUT_IMAGES, exist_ok=True)

print(f"\n输出目录: {OUTPUT_DIR}")
print(f"图片输出: {OUTPUT_IMAGES}")
print(f"CSV输出: {OUTPUT_CSV}")

# ========== 查找CSV和图片的函数 ==========
def find_csv_and_images(person_dir):
    """在person目录中查找CSV文件和图片文件夹"""
    
    # 查找CSV文件
    csv_path = None
    for file in os.listdir(person_dir):
        if file.endswith('.csv'):
            csv_path = os.path.join(person_dir, file)
            break
    
    if not csv_path:
        return None, None
    
    # 查找图片文件夹
    img_dir = None
    # 优先查找 images 文件夹
    test_dirs = ['images', 'img', 'pictures', 'JPEGImages', '']
    
    for folder in test_dirs:
        if folder:
            test_dir = os.path.join(person_dir, folder)
        else:
            test_dir = person_dir
        
        if os.path.exists(test_dir):
            # 检查是否有图片
            for f in os.listdir(test_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_dir = test_dir
                    break
            if img_dir:
                break
    
    return csv_path, img_dir

# ========== 读取CSV文件 ==========
def read_csv(csv_path):
    """读取CSV，返回列表"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 5:
                # 处理可能存在的BOM头
                img_name = row[0].strip()
                if img_name.startswith('\ufeff'):
                    img_name = img_name[1:]
                try:
                    x_min = int(row[1].strip())
                    y_min = int(row[2].strip())
                    x_max = int(row[3].strip())
                    y_max = int(row[4].strip())
                    data.append([img_name, x_min, y_min, x_max, y_max])
                except:
                    print(f"    警告: 跳过无效行 {row}")
    return data

# ========== 处理每个人 ==========
persons = [
    ('A', PERSON_A_DIR),
    ('B', PERSON_B_DIR),
    ('C', PERSON_C_DIR),
]

all_annotations = []
total_images = 0

for person_name, person_dir in persons:
    print(f"\n处理人员 {person_name}...")
    
    if not os.path.exists(person_dir):
        print(f"  ❌ 目录不存在: {person_dir}")
        continue
    
    # 查找CSV和图片
    csv_path, img_dir = find_csv_and_images(person_dir)
    
    if not csv_path:
        print(f"  ❌ 找不到CSV文件")
        print(f"     请确保 {person_dir} 下有 .csv 文件")
        continue
    
    if not img_dir:
        print(f"  ❌ 找不到图片文件夹")
        print(f"     请确保 {person_dir} 下有 images 文件夹或图片文件")
        continue
    
    print(f"  📄 CSV: {os.path.basename(csv_path)}")
    print(f"  📁 图片: {os.path.basename(img_dir)}")
    
    # 读取CSV
    annotations = read_csv(csv_path)
    print(f"  📊 标注记录: {len(annotations)} 条")
    
    # 获取所有图片文件名（用于快速查找）
    img_files = set()
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_files.add(f)
    print(f"  🖼️ 图片数量: {len(img_files)} 张")
    
    # 复制并重命名
    person_image_count = 0
    for ann in annotations:
        old_name = ann[0]
        new_name = f"{person_name}_{old_name}"
        
        # 复制图片
        old_img_path = os.path.join(img_dir, old_name)
        new_img_path = os.path.join(OUTPUT_IMAGES, new_name)
        
        if os.path.exists(old_img_path):
            shutil.copy2(old_img_path, new_img_path)
            person_image_count += 1
        else:
            # 尝试添加扩展名
            found = False
            for ext in ['.jpg', '.png', '.jpeg']:
                if old_name.lower().endswith(ext):
                    continue
                test_path = os.path.join(img_dir, old_name + ext)
                if os.path.exists(test_path):
                    shutil.copy2(test_path, new_img_path)
                    found = True
                    person_image_count += 1
                    break
            if not found:
                print(f"  ⚠️ 图片不存在: {old_name}")
        
        # 保存标注
        all_annotations.append([
            new_name,
            ann[1], ann[2], ann[3], ann[4]
        ])
    
    total_images += person_image_count
    print(f"  ✅ 完成，复制了 {person_image_count} 张图片")

# ========== 保存合并后的CSV ==========
if all_annotations:
    with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for ann in all_annotations:
            writer.writerow(ann)
    
    print("\n" + "="*60)
    print("合并完成！")
    print("="*60)
    print(f"总图片数: {total_images}")
    print(f"总标注框: {len(all_annotations)}")
    if total_images > 0:
        print(f"平均每图: {len(all_annotations)/total_images:.2f} 个框")
    print(f"\n输出文件:")
    print(f"  📁 图片: {OUTPUT_IMAGES}")
    print(f"  📄 CSV: {OUTPUT_CSV}")
else:
    print("\n❌ 没有找到任何数据！")
    print("\n请确认目录结构为:")
    print("""
D:\\StudyinG\\PythonStudy\\
├── person_A\\
│   ├── annotations.csv    (CSV文件)
│   └── images\\            (图片文件夹)
│       ├── 001.jpg
│       └── ...
├── person_B\\
│   ├── annotations.csv
│   └── images\\
└── person_C\\
    ├── annotations.csv
    └── images\\
""")