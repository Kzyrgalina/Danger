import torch
from ultralytics import YOLO
from src.config import Config

class DetectionModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(Config.MODEL_NAME)
        self.model.to(self.device)
    
    def predict(self, frame):
        """
        Выполняет предсказание на одном кадре
        """
        return self.model(
            frame,
            verbose=False,
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD
        )[0]
