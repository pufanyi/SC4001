from abc import ABC, abstractmethod

from PIL import Image


class Model(ABC):
    @abstractmethod
    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, input: Image.Image | list[Image.Image]) -> str:
        raise NotImplementedError("Subclasses must implement this method")
