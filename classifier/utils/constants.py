import json
from pathlib import Path

CLASSES_PATH = Path(__file__).parent / "classes.json"
CLASSES = json.load(open(CLASSES_PATH))
NUM_CLASSES = len(CLASSES)

__all__ = ["CLASSES", "NUM_CLASSES"]
