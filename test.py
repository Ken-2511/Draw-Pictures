import torch
from torch import nn
from torchvision import transforms as trans
import numpy as np
import cv2
from PIL import Image


def print_image_info(image):
    print(image.mean())
    print(image.min())
    print(image.max())


def load_image(img_name) -> torch.Tensor:
    transform = trans.Compose([
        trans.ToTensor()
    ])
    image = Image.open(img_name)
    image = transform(image)
    return image


def save_image(image, name):
    transform = trans.ToPILImage()
    image = transform(image)
    image.save(name)


def gaussian_kernel(size, theta):
    def value(x, y):
        return 1 / (2 * np.pi * theta**2) * np.e ** (-(x ** 2 + y ** 2) / (2 * theta ** 2))
    kernel = []
    for x in range(-(size//2), size//2+1):
        row = []
        for y in range(-(size//2), size//2+1):
            row.append(value(x, y))
        kernel.append(row)
    return kernel


def process_image(img_name):
    # load model
    model = nn.Sequential(
        nn.Conv2d(1, 1, 5),
        # nn.Conv2d(1, 1, 5)
    )
    model[0].weight.data = torch.tensor([[[[-1, -1, -1, -1, -1],
                                           [-1, 1, 1, 1, -1],
                                           [-1, 1, 8, 1, -1],
                                           [-1, 1, 1, 1, -1],
                                           [-1, -1, -1, -1, -1]]]], dtype=torch.float)
    model[0].bias.data = torch.tensor([0.])
    # kernel = get_kernel(5, 2)
    # model[1].weight.data = torch.tensor([[kernel]])
    # model[1].bias.data = torch.tensor([0.])


if __name__ == "__main__":
    image = load_image("paimon.jpg")
    print_image_info(image)