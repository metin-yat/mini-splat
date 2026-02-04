import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_PATH = os.getenv("DATA_PATH", "./data")
    SPARSE_PATH = os.getenv("COLMAP_OUTPUT_PATH")
    IMAGES_PATH = os.getenv("IMAGE_PATH")
    
    # Eğitim Parametreleri
    LEARNING_RATE = 1e-3
    ITERATIONS = 30000