import cv2
from pathlib import Path
import datetime
from src.config import Config
from src.models.detector import DetectionModel
from src.preprocessing.smoother import DetectionSmoother
from src.output.reporter import DetectionReporter

class VideoProcessor:
    def __init__(self):
        self.model = DetectionModel()
        self.smoother = DetectionSmoother()
        self.reporter = DetectionReporter()
    
    def process_frame(self, frame, video_file, frame_count, fps):
        """
        Обработка одного кадра видео
        """
        time_in_video = frame_count / fps
        
        # Получаем предсказания
        results = self.model.predict(frame)
        
        # Собираем детекции
        frame_detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            class_name = results.names[int(cls)]
            
            detection = {
                'video_file': video_file,
                'time': time_in_video,
                'class': class_name,
                'confidence': conf,
                'frame': frame_count,
                'bbox': (x1, y1, x2, y2)
            }
            frame_detections.append(detection)
        
        # Применяем сглаживание
        return self.smoother.smooth_detections(frame_detections, frame_count)
    
    def draw_detections(self, frame, detections):
        """
        Отрисовка детекций на кадре
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    
    def process_videos(self, video_folder=None):
        """
        Обработка всех видео в папке
        """
        video_folder = Path(video_folder) if video_folder else Config.DATA_DIR
        
        # Создаем папку для результатов
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = Config.OUTPUT_DIR / timestamp
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Получаем список видео
        video_files = [f for f in video_folder.glob('*') if f.suffix.lower() in Config.VIDEO_FORMATS]
        
        all_detections = []
        
        for video_file in video_files:
            print(f"Processing {video_file.name}...")
            
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Создаем writer для выходного видео
            output_video_path = output_folder / f"processed_{video_file.name}"
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Обрабатываем кадр
                detections = self.process_frame(frame, video_file.name, frame_count, fps)
                
                # Отрисовываем детекции
                frame = self.draw_detections(frame, detections)
                
                # Сохраняем детекции
                all_detections.extend(detections)
                
                writer.write(frame)
                frame_count += 1
            
            cap.release()
            writer.release()
        
        # Генерируем отчеты
        self.reporter.generate_reports(all_detections, output_folder)
