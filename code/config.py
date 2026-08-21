# config.py
from dotenv import load_dotenv
import os

load_dotenv()

CODE_PATH = os.getenv('CODE_PATH')
GDC_PATH = os.getenv('GDC_PATH')
RAW_THYROID_PATH = os.getenv('RAW_THYROID_PATH')
THYROID_PATH = os.getenv('THYROID_PATH')
TENSORBOARD_PATH = os.getenv('TENSORBOARD_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')
RESULTS_PATH = os.getenv("RESULTS_PATH")