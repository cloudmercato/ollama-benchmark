import os
from ollama_benchmark import defaults

DATA_DIR = os.getenv('OB_DATA_DIR', defaults.DATA_DIR)
HOST = os.getenv('OB_OLLAMA_HOST', defaults.HOST)
TIMEOUT = os.getenv('OB_OLLAMA_TIMEOUT', defaults.TIMEOUT)
MODEL = os.getenv('OB_MODEL', defaults.MODEL)
JUDGE_MODEL = os.getenv('OB_JUDGE_MODEL', defaults.JUDGE_MODEL)
