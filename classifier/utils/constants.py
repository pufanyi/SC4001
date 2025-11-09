import json
from pathlib import Path

from .logger import logger

CLASSES_PATH = Path(__file__).parent.parent.parent / "config" / "classes.json"
CLASSES = json.load(open(CLASSES_PATH))
NUM_CLASSES = len(CLASSES)
ID2LABEL = dict(enumerate(CLASSES))
LABEL2ID = {label: i for i, label in enumerate(CLASSES)}

logger.info(f"Loaded {NUM_CLASSES} classes: {CLASSES}")

__all__ = ["CLASSES", "NUM_CLASSES", "ID2LABEL", "LABEL2ID"]
