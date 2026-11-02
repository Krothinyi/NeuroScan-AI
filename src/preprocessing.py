
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizeWithPadding:
    def __call__(self, image):
        image = image.convert("RGB")

        return ImageOps.pad(
            image,
            (IMG_SIZE, IMG_SIZE),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0)
        )


# Training: random augmentation
train_transform = transforms.Compose([
    ResizeWithPadding(),

    transforms.RandomAffine(
        degrees=7,
        translate=(0.02, 0.02),
        scale=(0.97, 1.03),
        interpolation=InterpolationMode.BILINEAR,
        fill=0
    ),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# Validation, calibration, test and deployment:
# No random augmentation
eval_transform = transforms.Compose([
    ResizeWithPadding(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])
