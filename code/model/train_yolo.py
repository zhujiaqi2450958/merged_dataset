import os
# 解决OMP报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ========== 1. 路径配置（不要动，保持和你一致） ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(BASE_DIR, "dataset.yaml")

# ========== 2. 每次只改这 4 行！（按上面表格改） ==========
lr0 = 0.001
epochs = 30
iou = 0.9
name = "lr0.001_e30_iou0.9"

# ========== 3. 加载模型并训练 ==========
model = YOLO("yolov8n.pt")

results = model.train(
    data=data_path,
    epochs=epochs,
    batch=8,
    imgsz=640,
    lr0=lr0,
    iou=iou,
    name=name,
    project=os.path.join(BASE_DIR, "runs"),
    device="cpu"   # 你现在是CPU，不要改成cuda
)

print(f"✅ 训练完成：{name}")