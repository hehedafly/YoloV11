from ultralytics import YOLO
import torch
import os
from torch.utils.data import DataLoader
 
model = YOLO(r'models\yolo11n.pt')  # 加载预训练模型（建议用于训练）
_epochs = 30
train = True

if not train:
    model.export(format = 'engine', dynamic = False, half = False, nms = False)


def checkWeight(path:str) -> bool:
    return any(file.endswith(".pt") for _, _, files in os.walk(path) for file in files)

# 使用模型
if __name__ == "__main__":
    if train:
        # model.train(data="yoloTrain.yaml", epochs= _epochs, multi_scale=True, profile=True, workers=0)  # 训练模型
        model.train(data="yoloTrain.yaml", epochs= _epochs, multi_scale=True, profile=True, workers=0)  # 训练模型
        metrics = model.val()  # 在验证集上评估模型性能
        train_dirs = [d for d in os.listdir("./runs/detect") if os.path.isdir(os.path.join("./runs/detect", d)) and d.startswith("train") and checkWeight(os.path.join("./runs/detect", d))]
        train_dirs.sort(key=lambda x: int(x[len("train") :] if len(x) > 5 else 0))

        model = YOLO(f"./runs/detect/{train_dirs[-1]}/weights/best.pt")
        # model.export(format="onnx", imgsz = [480, 640],  nms = True, device = "cpu") 
        # model.export(format="openvino", imgsz = [480, 640], device = "cpu", batch = 32)
        model.export(format = 'engine', dynamic = False, half = False)

# results = model.predict(source="YoloTrainData\images\test",save=True,save_conf=True,save_txt=True,name='output')