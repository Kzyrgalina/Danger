import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from matplotlib.gridspec import GridSpec

class DetectionReporter:
    def __init__(self):
        # Установка цветовой палитры
        self.colors = sns.color_palette('husl', 10)
    
    def create_detection_timeline(self, video_df, output_path):
        """
        Создает визуализацию временной шкалы детекций
        """
        plt.figure(figsize=(15, 8))
        classes = video_df['class'].unique()
        
        for i, cls in enumerate(classes):
            class_data = video_df[video_df['class'] == cls]
            plt.scatter(class_data['time'], [i] * len(class_data),
                       label=cls, alpha=0.6, s=100)
            
            # Добавляем уверенность как размер точек
            sizes = class_data['confidence'] * 200
            plt.scatter(class_data['time'], [i] * len(class_data),
                       s=sizes, alpha=0.3)
        
        plt.yticks(range(len(classes)), classes)
        plt.xlabel('Время (секунды)')
        plt.title('Временная шкала детекций')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'timeline.png')
        plt.close()
    
    def create_confidence_distribution(self, video_df, output_path):
        """
        Создает визуализацию распределения уверенности по классам
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=video_df, x='class', y='confidence')
        plt.xticks(rotation=45)
        plt.title('Распределение уверенности по классам')
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_distribution.png')
        plt.close()
    
    def create_detection_heatmap(self, video_df, output_path):
        """
        Создает тепловую карту детекций во времени
        """
        # Создаем временные бины
        max_time = video_df['time'].max()
        bins = np.linspace(0, max_time, 50)
        classes = video_df['class'].unique()
        
        # Создаем матрицу для тепловой карты
        heatmap_data = np.zeros((len(classes), len(bins)-1))
        
        for i, cls in enumerate(classes):
            class_data = video_df[video_df['class'] == cls]['time']
            hist, _ = np.histogram(class_data, bins=bins)
            heatmap_data[i] = hist
        
        plt.figure(figsize=(15, 5))
        sns.heatmap(heatmap_data, xticklabels=False, yticklabels=classes,
                    cmap='YlOrRd', cbar_kws={'label': 'Количество детекций'})
        plt.xlabel('Время')
        plt.title('Тепловая карта детекций')
        plt.tight_layout()
        plt.savefig(output_path / 'detection_heatmap.png')
        plt.close()
    
    def create_summary_dashboard(self, video_df, output_path):
        """
        Создает сводную информационную панель
        """
        total_detections = len(video_df)
        unique_classes = video_df['class'].nunique()
        avg_confidence = video_df['confidence'].mean()
        video_duration = video_df['time'].max()
        
        class_counts = video_df['class'].value_counts()
        
        # Создаем dashboard
        plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # Основные метрики
        plt.subplot(gs[0, 0])
        metrics = [
            f'Всего детекций: {total_detections}',
            f'Уникальных классов: {unique_classes}',
            f'Средняя уверенность: {avg_confidence:.2%}',
            f'Длительность видео: {video_duration:.1f}s'
        ]
        plt.axis('off')
        for i, metric in enumerate(metrics):
            plt.text(0.1, 0.8 - i*0.2, metric, fontsize=12)
        plt.title('Основные метрики')
        
        # Круговая диаграмма классов
        plt.subplot(gs[0, 1])
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        plt.title('Распределение классов')
        
        # График уверенности во времени
        plt.subplot(gs[1, :])
        sns.scatterplot(data=video_df, x='time', y='confidence', hue='class', alpha=0.6)
        plt.title('Уверенность детекций во времени')
        
        plt.tight_layout()
        plt.savefig(output_path / 'summary_dashboard.png')
        plt.close()
    
    def generate_reports(self, detections, output_folder):
        """
        Генерирует все отчеты
        """
        df = pd.DataFrame(detections)
        
        # Сохраняем CSV для возможного последующего анализа
        df.to_csv(output_folder / 'detection_report.csv', index=False)
        
        # Создаем отчеты для каждого видео
        for video_file in df['video_file'].unique():
            video_df = df[df['video_file'] == video_file]
            video_name = Path(video_file).stem
            
            # Создаем папку для отчетов этого видео
            video_report_dir = output_folder / f'report_{video_name}'
            video_report_dir.mkdir(exist_ok=True)
            
            # Генерируем все визуализации
            self.create_detection_timeline(video_df, video_report_dir)
            self.create_confidence_distribution(video_df, video_report_dir)
            self.create_detection_heatmap(video_df, video_report_dir)
            self.create_summary_dashboard(video_df, video_report_dir)
