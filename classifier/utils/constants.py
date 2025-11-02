import json
from pathlib import Path

from logger import logger

CLASSES_PATH = Path(__file__).parent / "classes.json"
CLASSES = json.load(open(CLASSES_PATH))
NUM_CLASSES = len(CLASSES)

logger.info(f"Loaded {NUM_CLASSES} classes: {CLASSES}")

__all__ = ["CLASSES", "NUM_CLASSES"]
