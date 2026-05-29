from ultralytics import YOLO
import os

# 自动获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 加载训练好的最佳模型权重
model_path = os.path.join(BASE_DIR, "runs", "pedestrian_detection", "weights", "best.pt")
model = YOLO(model_path)

# 加载数据配置文件
data_path = os.path.join(BASE_DIR, "dataset.yaml")

# 在验证集上评估模型
metrics = model.val(data=data_path, split="val")

# 打印关键指标
print("📊 模型评估结果：")
print(f"mAP@50: {metrics.box.map50:.4f}")       # IoU=0.5时的平均精度
print(f"mAP@50-95: {metrics.box.map:.4f}")    # IoU从0.5到0.95的平均精度
print(f"精确率(Precision): {metrics.box.p:.4f}")
print(f"召回率(Recall): {metrics.box.r:.4f}")