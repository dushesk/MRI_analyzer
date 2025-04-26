from pathlib import Path

class Settings:
    MODEL_PATH = Path("app/models/best_custom_cnn.h5")
    IMAGE_SIZE = (224, 224)  # Размер изображения для модели

settings = Settings()