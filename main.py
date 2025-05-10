from src.processing.video_processor import VideoProcessor
from src.config import Config
import sys
from pathlib import Path

def main():
    # Создаем необходимые директории
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    Config.DATA_DIR.mkdir(exist_ok=True)
    
    # Создаем процессор
    processor = VideoProcessor()
    
    # Если указана папка с видео в аргументах, используем её
    video_folder = None
    if len(sys.argv) > 1:
        video_folder = Path(sys.argv[1])
        if not video_folder.exists():
            print(f"Error: Folder {video_folder} does not exist")
            return
    
    # Запускаем обработку
    processor.process_videos(video_folder)

if __name__ == "__main__":
    main()
