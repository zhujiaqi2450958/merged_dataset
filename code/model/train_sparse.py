import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# 自动路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(BASE_DIR, "dataset_sparse.yaml")

# 固定参数：0.001  30  0.7
model = YOLO("yolov8n.pt")

results = model.train(
    data=data_path,
    epochs=30,
    batch=8,
    imgsz=640,
    lr0=0.001,
    iou=0.7,
    name="lr0.001_e30_iou0.7_sparse",
    project=os.path.join(BASE_DIR, "runs"),
    device="cpu"
)

print("✅ 稀疏数据集训练完成！")