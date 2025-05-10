from pathlib import Path

class Config:
    # Пути к папкам
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'output'
    
    # Параметры модели
    MODEL_NAME = 'yolov8l.pt'
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    # Параметры обработки видео
    SMOOTH_WINDOW = 5
    MIN_DETECTIONS = 2
    FRAME_WINDOW = 3
    
    # Поддерживаемые форматы видео
    VIDEO_FORMATS = ('.mp4', '.avi', '.mov')
