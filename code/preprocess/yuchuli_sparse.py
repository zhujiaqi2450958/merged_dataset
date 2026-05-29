import os
import cv2
import numpy as np
import random

print("="*60)
print("稀疏数据集 图像预处理 + 数据集划分")
print("="*60)

# ========== 配置参数 ==========
IMAGE_SIZE = 640
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42

# ========== 自动获取项目根目录 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====================== 稀疏数据集 ======================
IMAGE_DIR = os.path.join(BASE_DIR, "sparse")
CSV_PATH = os.path.join(BASE_DIR, "sparse_annotations.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_dataset_sparse")
CONFIG_NAME = "dataset_sparse.yaml"
# ==========================================================

print(f"\n图片目录: {IMAGE_DIR}")
print(f"CSV文件: {CSV_PATH}")
print(f"输出目录: {OUTPUT_DIR}")

# ========== 1. 检查路径 ==========
print("\n【1. 检查路径】")

if not os.path.exists(IMAGE_DIR):
    print(f"❌ 图片目录不存在: {IMAGE_DIR}")
    print("请确认 'sparse' 文件夹已复制到项目根目录")
    exit()
if not os.path.exists(CSV_PATH):
    print(f"❌ CSV文件不存在: {CSV_PATH}")
    print("请确认 'sparse_annotations.csv' 文件已复制到项目根目录")
    exit()
print("✅ 路径检查通过")

# ========== 2. 读取CSV ==========
print("\n【2. 读取标注信息】")

annotations = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(',')
            if len(parts) == 5:
                img_name = parts[0].strip()
                try:
                    x_min = int(parts[1].strip())
                    y_min = int(parts[2].strip())
                    x_max = int(parts[3].strip())
                    y_max = int(parts[4].strip())
                    annotations.append([img_name, x_min, y_min, x_max, y_max])
                except:
                    pass

print(f"总标注框数: {len(annotations)}")

csv_image_names = list(set([ann[0] for ann in annotations]))
print(f"CSV中的图片数: {len(csv_image_names)}")

actual_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"文件夹中的图片数: {len(actual_images)}")

valid_images = [img for img in csv_image_names if img in actual_images]
print(f"有效图片数: {len(valid_images)}")

if len(valid_images) == 0:
    print("\n❌ 没有找到匹配的图片！")
    exit()

image_to_boxes = {}
for ann in annotations:
    img_name = ann[0]
    if img_name in valid_images:
        box = ann[1:5]
        if img_name not in image_to_boxes:
            image_to_boxes[img_name] = []
        image_to_boxes[img_name].append(box)

# ========== 3. 划分数据集 ==========
print("\n【3. 划分数据集】")

random.seed(RANDOM_SEED)
all_images = list(image_to_boxes.keys())
random.shuffle(all_images)

n_total = len(all_images)
n_test = int(n_total * TEST_SIZE)
n_val = int(n_total * VAL_SIZE)

test_images = all_images[:n_test]
val_images = all_images[n_test:n_test+n_val]
train_images = all_images[n_test+n_val:]

print(f"训练集: {len(train_images)} 张")
print(f"验证集: {len(val_images)} 张")
print(f"测试集: {len(test_images)} 张")

# ========== 4. 创建输出目录 ==========
print("\n【4. 创建输出目录】")

for split in ['train', 'val', 'test']:
    os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)

print("✅ 输出目录创建成功")

# ========== 5. 预处理函数 ==========
def letterbox_resize(image, target_size=640):
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def convert_to_yolo_format(box, img_width, img_height):
    x_min, y_min, x_max, y_max = box
    x_min = max(0, min(img_width, x_min))
    x_max = max(0, min(img_width, x_max))
    y_min = max(0, min(img_height, y_min))
    y_max = max(0, min(img_height, y_max))
    
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    x_center = max(0.0001, min(0.9999, x_center))
    y_center = max(0.0001, min(0.9999, y_center))
    width = max(0.0001, min(0.9999, width))
    height = max(0.0001, min(0.9999, height))
    
    return x_center, y_center, width, height

# ========== 6. 处理函数 ==========
def process_split(image_list, split_name):
    print(f"\n处理 {split_name} 集...")
    success_count = 0
    total_boxes = 0
    
    for i, img_name in enumerate(image_list):
        if (i + 1) % 50 == 0 or (i + 1) == len(image_list):
            print(f"  进度: {i+1}/{len(image_list)}")
        
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        orig_h, orig_w = img.shape[:2]
        resized_img = letterbox_resize(img, IMAGE_SIZE)
        
        save_img_path = f'{OUTPUT_DIR}/images/{split_name}/{img_name}'
        cv2.imencode('.jpg', resized_img)[1].tofile(save_img_path)
        
        boxes = image_to_boxes.get(img_name, [])
        if not boxes:
            continue
        
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        save_label_path = f'{OUTPUT_DIR}/labels/{split_name}/{label_name}'
        
        with open(save_label_path, 'w') as f:
            for box in boxes:
                x_center, y_center, width, height = convert_to_yolo_format(box, orig_w, orig_h)
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_boxes += 1
        
        success_count += 1
    
    print(f"  ✅ {split_name}集完成: {success_count}/{len(image_list)} 张, {total_boxes} 个框")
    return success_count, total_boxes

# ========== 7. 执行预处理 ==========
print("\n【5. 执行预处理】")
train_stats = process_split(train_images, 'train')
val_stats = process_split(val_images, 'val')
test_stats = process_split(test_images, 'test')

# ========== 8. 生成配置文件 ==========
print("\n【6. 生成YOLO配置文件】")
config_content = f"""path: {OUTPUT_DIR.replace(os.sep, '/')}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['person']
"""

config_path = os.path.join(BASE_DIR, CONFIG_NAME)
with open(config_path, 'w', encoding='utf-8') as f:
    f.write(config_content)

print(f"✅ 配置文件已保存: {config_path}")
print("\n✅ 稀疏数据集预处理完成！")