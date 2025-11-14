from omegaconf import DictConfig
from torchvision import transforms
from transformers import AutoImageProcessor, BaseImageProcessor


class ProcessorFactory:
    @staticmethod
    def get_processor(config: DictConfig) -> BaseImageProcessor:
        processor_name = config.processor.name
        return AutoImageProcessor.from_pretrained(processor_name)

    @staticmethod
    def get_train_processor(config: DictConfig) -> transforms.Compose:
        inference_processor = ProcessorFactory.get_processor(config)

        # Handle different processor size formats
        if isinstance(inference_processor.size, dict):
            # Try different keys that might contain the image size
            image_size = inference_processor.size.get("shortest_edge")
            if image_size is None:
                image_size = inference_processor.size.get(
                    "height", inference_processor.size.get("width", 224)
                )
        else:
            # If size is a single integer
            image_size = inference_processor.size

        transform_list = []

        transform_list.append(
            transforms.RandomResizedCrop(
                size=image_size,
                scale=config.processor.train.scale,
                ratio=config.processor.train.ratio,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )

        if config.processor.train.use_different_processors:
            if config.processor.train.hflip_prob > 0:
                transform_list.append(
                    transforms.RandomHorizontalFlip(p=config.processor.train.hflip_prob)
                )
            if config.processor.train.vflip_prob > 0:
                transform_list.append(
                    transforms.RandomVerticalFlip(p=config.processor.train.vflip_prob)
                )

            if config.processor.train.rotation_degrees > 0:
                transform_list.append(
                    transforms.RandomRotation(
                        degrees=config.processor.train.rotation_degrees,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        fill=0,
                    )
                )

            if config.processor.train.use_randaugment:
                transform_list.append(
                    transforms.RandAugment(
                        num_ops=config.processor.train.randaugment_n,
                        magnitude=config.processor.train.randaugment_m,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                )

            if config.processor.train.use_color_jitter:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=config.processor.train.brightness,
                        contrast=config.processor.train.contrast,
                        saturation=config.processor.train.saturation,
                        hue=config.processor.train.hue,
                    )
                )

            if config.processor.train.grayscale_prob > 0:
                transform_list.append(
                    transforms.RandomGrayscale(p=config.processor.train.grayscale_prob)
                )

            if config.processor.train.blur_prob > 0:
                transform_list.append(
                    transforms.RandomApply(
                        [
                            transforms.GaussianBlur(
                                kernel_size=23, sigma=config.processor.train.blur_sigma
                            )
                        ],
                        p=config.processor.train.blur_prob,
                    )
                )

        transform_list.append(transforms.ToTensor())

        transform_list.append(
            transforms.Normalize(
                mean=inference_processor.image_mean,
                std=inference_processor.image_std,
            )
        )

        if (
            config.processor.train.use_different_processors
            and config.processor.train.use_random_erasing
        ):
            transform_list.append(
                transforms.RandomErasing(
                    p=config.processor.train.erasing_prob,
                    scale=config.processor.train.erasing_scale,
                    ratio=config.processor.train.erasing_ratio,
                    value="random",
                )
            )

        transform = transforms.Compose(transform_list)
        return transform
