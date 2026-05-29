import os
# 解决OMP报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ======== 项目根目录（自动找，不用改） ========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======== 1. 拥挤图片目录（你只要把图片放进 merged_dataset/crowded） ========
CROWDED_DIR = os.path.join(BASE_DIR, "crowded")

# ======== 2. 生成一个临时的数据集结构和yaml ========
import shutil
import random

# 输出目录
OUT_DATA = os.path.join(BASE_DIR, "processed_dataset_crowded")
os.makedirs(os.path.join(OUT_DATA, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUT_DATA, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUT_DATA, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUT_DATA, "labels", "val"), exist_ok=True)

# 拿到所有图片
imgs = [f for f in os.listdir(CROWDED_DIR) if f.endswith(('jpg','jpeg','png'))]
random.shuffle(imgs)
split = int(len(imgs)*0.8)
train_imgs = imgs[:split]
val_imgs = imgs[split:]

# 复制图片（标签你后面再补也行，先跑图片过一遍流程）
def copyimg(imglist, splitname):
    for img in imglist:
        shutil.copy(os.path.join(CROWDED_DIR, img),
                    os.path.join(OUT_DATA, "images", splitname, img))

copyimg(train_imgs, "train")
copyimg(val_imgs, "val")

# 生成 dataset_crowded.yaml
yaml_content = f"""
path: {OUT_DATA.replace(os.sep, '/')}
train: images/train
val: images/val
nc: 1
names: ['person']
"""
yaml_path = os.path.join(BASE_DIR, "dataset_crowded.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

# ======== 3. 只跑这一组：lr=0.001, epochs=30, iou=0.7 ========
model = YOLO("yolov8n.pt")

results = model.train(
    data=yaml_path,
    epochs=30,
    batch=8,
    imgsz=640,
    lr0=0.001,     # 学习率 0.001
    iou=0.7,       # iou=0.7
    name="lr0.001_e30_iou0.7_crowded",
    project=os.path.join(BASE_DIR, "runs"),
    device="cpu"
)

print("✅ 拥挤数据集训练完成：lr=0.001, e30, iou=0.7")