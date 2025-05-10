from src.config import Config

class DetectionSmoother:
    def __init__(self):
        self.detection_history = []
    
    def smooth_detections(self, current_detections, frame_count):
        """
        Применяет временное сглаживание к детекциям
        """
        # Добавляем текущие детекции в историю
        self.detection_history.append((frame_count, current_detections))
        
        # Удаляем старые детекции
        while len(self.detection_history) > Config.SMOOTH_WINDOW:
            self.detection_history.pop(0)
        
        # Если истории недостаточно, возвращаем текущие детекции
        if len(self.detection_history) < Config.MIN_DETECTIONS:
            return current_detections
        
        smoothed_detections = []
        for detection in current_detections:
            # Проверяем, есть ли похожие детекции в истории
            consistent_detections = 0
            for hist_frame, hist_detections in self.detection_history[-Config.FRAME_WINDOW:]:
                for hist_detection in hist_detections:
                    if (hist_detection['class'] == detection['class'] and 
                        abs(hist_frame - frame_count) <= Config.FRAME_WINDOW):
                        consistent_detections += 1
                        break
            
            # Добавляем детекцию только если она стабильно появляется
            if consistent_detections >= Config.MIN_DETECTIONS:
                smoothed_detections.append(detection)
        
        return smoothed_detections
