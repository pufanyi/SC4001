from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor

from .processor import Processor


class ConvNeXtV2Processor(Processor):
    def __init__(
        self,
        processor_name: str = "facebook/convnextv2-huge-22k-384",
    ):
        super().__init__(name="convnextv2_processor")
        self.processor_name = processor_name

        self.hf_processor = AutoImageProcessor.from_pretrained(processor_name)

    def __call__(self, images: list[Image.Image]) -> dict:
        return self.hf_processor(images, return_tensors="pt")


class ConvNeXtTrainProcessor(Processor):
    def __init__(
        self,
        processor_name: str = "facebook/convnextv2-huge-22k-384",
        use_randaugment: bool = True,
        randaugment_n: int = 2,
        randaugment_m: int = 9,
        scale: tuple[float, float] = (0.5, 1.0),
        ratio: tuple[float, float] = (0.75, 1.33),
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.1,
        use_color_jitter: bool = True,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.2,
        grayscale_prob: float = 0.1,
        blur_prob: float = 0.2,
        blur_sigma: tuple[float, float] = (0.1, 2.0),
        rotation_degrees: float = 15.0,
        use_random_erasing: bool = True,
        erasing_prob: float = 0.25,
        erasing_scale: tuple[float, float] = (0.02, 0.33),
        erasing_ratio: tuple[float, float] = (0.3, 3.3),
    ):
        super().__init__(name="convnextv2_train_processor")
        self.processor_name = processor_name

        self.hf_processor = AutoImageProcessor.from_pretrained(processor_name)

        self.image_size = self.hf_processor.size.get("shortest_edge", 384)
        if isinstance(self.image_size, dict):
            self.image_size = self.image_size.get("height", 384)

        self.image_mean = self.hf_processor.image_mean
        self.image_std = self.hf_processor.image_std
        self.resample = getattr(
            transforms.InterpolationMode,
            self.hf_processor.resample.name
            if hasattr(self.hf_processor.resample, "name")
            else "BICUBIC",
        )

        transform_list = []

        transform_list.append(
            transforms.RandomResizedCrop(
                size=self.image_size,
                scale=scale,
                ratio=ratio,
                interpolation=self.resample,
            )
        )

        if hflip_prob > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=hflip_prob))
        if vflip_prob > 0:
            transform_list.append(transforms.RandomVerticalFlip(p=vflip_prob))

        if rotation_degrees > 0:
            transform_list.append(
                transforms.RandomRotation(
                    degrees=rotation_degrees,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )
            )

        if use_randaugment:
            transform_list.append(
                transforms.RandAugment(
                    num_ops=randaugment_n,
                    magnitude=randaugment_m,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )

        if use_color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            )

        if grayscale_prob > 0:
            transform_list.append(transforms.RandomGrayscale(p=grayscale_prob))

        if blur_prob > 0:
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23, sigma=blur_sigma)],
                    p=blur_prob,
                )
            )

        transform_list.append(transforms.ToTensor())

        transform_list.append(
            transforms.Normalize(
                mean=self.image_mean,
                std=self.image_std,
            )
        )

        if use_random_erasing:
            transform_list.append(
                transforms.RandomErasing(
                    p=erasing_prob,
                    scale=erasing_scale,
                    ratio=erasing_ratio,
                    value="random",
                )
            )

        self.transform = transforms.Compose(transform_list)

        self.config = {
            "processor_name": processor_name,
            "image_size": self.image_size,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "use_randaugment": use_randaugment,
            "randaugment_n": randaugment_n,
            "randaugment_m": randaugment_m,
            "scale": scale,
            "ratio": ratio,
            "hflip_prob": hflip_prob,
            "vflip_prob": vflip_prob,
            "use_color_jitter": use_color_jitter,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "grayscale_prob": grayscale_prob,
            "blur_prob": blur_prob,
            "rotation_degrees": rotation_degrees,
            "use_random_erasing": use_random_erasing,
            "erasing_prob": erasing_prob,
        }

    def __call__(self, images: list[Image.Image]) -> dict:
        images = [image.convert("RGB") for image in images if image.mode != "RGB"]

        return {
            "pixel_values": self.transform(images),
        }
