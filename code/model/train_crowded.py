import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ====================== 自动获取项目路径（不用改） ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(BASE_DIR, "dataset_crowded.yaml")  # 拥挤数据集配置

# ====================== 固定你要的参数：0.001  30  0.7 ======================
model = YOLO("yolov8n.pt")

results = model.train(
    data=data_path,
    epochs=30,        # 固定30轮
    batch=8,
    imgsz=640,
    lr0=0.001,        # 固定学习率0.001
    iou=0.7,          # 固定IOU 0.7
    name="lr0.001_e30_iou0.7_crowded",
    project=os.path.join(BASE_DIR, "runs"),
    device="cpu"
)

print("✅ 拥挤数据集训练完成！")