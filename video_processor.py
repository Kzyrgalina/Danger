import os
import cv2
import torch
import datetime
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

class VideoProcessor:
    def __init__(self):
        # Убедимся что используется GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Загружаем модель YOLOv8m - - баланс между скоростью и точностью
        self.model = YOLO('yolov8l.pt')
        # Установка порогов для повышения точности
        self.conf_threshold = 0.5  # Порог уверенности
        self.iou_threshold = 0.45  # Порог IOU для NMS
        self.model.to(self.device)
        
        # Параметры временного сглаживания
        self.smooth_window = 5  # Размер окна сглаживания
        self.detection_history = []  # История детекций
        
    def smooth_detections(self, current_detections, frame_count):
        # Добавляем текущие детекции в историю
        self.detection_history.append((frame_count, current_detections))
        
        # Удаляем старые детекции за пределами окна сглаживания
        while len(self.detection_history) > self.smooth_window:
            self.detection_history.pop(0)
        
        # Если истории недостаточно, возвращаем текущие детекции
        if len(self.detection_history) < 3:
            return current_detections
        
        smoothed_detections = []
        for detection in current_detections:
            # Проверяем, есть ли похожие детекции в истории
            consistent_detections = 0
            for hist_frame, hist_detections in self.detection_history[-3:]:  # Проверяем последние 3 кадра
                for hist_detection in hist_detections:
                    if (hist_detection['class'] == detection['class'] and 
                        abs(hist_frame - frame_count) <= 3):  # Проверяем только близкие по времени кадры
                        consistent_detections += 1
                        break
            
            # Добавляем детекцию только если она стабильно появляется
            if consistent_detections >= 2:  # Требуем минимум 2 подтверждения
                smoothed_detections.append(detection)
        
        return smoothed_detections

    def process_videos(self, video_folder):
        # Создаем папку для результатов с текущим временем
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = Path('output') / timestamp
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Получаем список всех видео файлов
        video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        all_detections = []
        
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing {video_file}...")
            
            # Открываем видео
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Создаем writer для выходного видео
            output_video_path = output_folder / f"processed_{video_file}"
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
                
                # Получаем время в видео
                time_in_video = frame_count / fps
                
                # Делаем предсказание с установленными порогами
                results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
                
                # Обрабатываем каждое обнаружение
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = r
                    class_name = results.names[int(cls)]
                    
                    # Сохраняем информацию об обнаружении
                    detection = {
                        'video_file': video_file,
                        'time': time_in_video,
                        'class': class_name,
                        'confidence': conf,
                        'frame': frame_count
                    }
                    all_detections.append(detection)
                    
                    # Рисуем рамку и подпись
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # После обработки детекций в кадре
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
                
                # Применяем временное сглаживание
                smoothed_detections = self.smooth_detections(frame_detections, frame_count)
                
                # Отрисовываем только сглаженные детекции
                for detection in smoothed_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{detection['class']}: {detection['confidence']:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Добавляем в общий список только сглаженные детекции
                all_detections.extend(smoothed_detections)
                
                writer.write(frame)
                frame_count += 1
            
            cap.release()
            writer.release()
        
        # Создаем отчет
        df = pd.DataFrame(all_detections)
        report_path = output_folder / 'detection_report.csv'
        df.to_csv(report_path, index=False)
        
        # Создаем текстовый отчет с обобщенной статистикой
        summary_path = output_folder / 'summary_report.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Detection Report - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            for video_file in video_files:
                video_df = df[df['video_file'] == video_file]
                f.write(f"Statistics for {video_file}:\n")
                f.write("-" * 30 + "\n")
                
                # Подсчет объектов по классам
                class_counts = video_df['class'].value_counts()
                f.write("\nDetected objects by class:\n")
                for cls, count in class_counts.items():
                    f.write(f"{cls}: {count} detections\n")
                
                # Средняя уверенность по классам
                class_conf = video_df.groupby('class')['confidence'].mean()
                f.write("\nAverage confidence by class:\n")
                for cls, conf in class_conf.items():
                    f.write(f"{cls}: {conf:.2%}\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"Processing complete. Results saved to {output_folder}")

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_videos('data')
