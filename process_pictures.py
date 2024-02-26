# This file is used to process pictures. It can be used to apply filters to pictures, such as edge detection, etc.


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


def load_image(f_name):
    image = Image.open(f_name)
    transform = trans.ToTensor()
    image = Image.open(f_name)
    image = transform(image)
    return image


def save_image(image, f_name):
    transform = trans.ToPILImage()
    image = transform(image)
    image.save(f_name)


def get_kernel(size, theta):
    """legacy function, not used in this project."""
    def value(x, y):
        return 1 / (2 * torch.pi * theta**2) * torch.e ** (-(x ** 2 + y ** 2) / (2 * theta ** 2))
    kernel = []
    for x in range(-(size//2), size//2+1):
        row = []
        for y in range(-(size//2), size//2+1):
            row.append(value(x, y))
        kernel.append(row)
    return kernel


def process_image(image: torch.Tensor):
    """get the grey-scale image of the original image."""
    # load model
    model = nn.Sequential(
        nn.Conv2d(3, 1, 5),
        # nn.Conv2d(1, 1, 5)
    )
    matrix = [[-1, -1, -1, -1, -1],
            [-1, 1, 1, 1, -1],
            [-1, 1, 8, 1, -1],
            [-1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1]]
    model[0].weight.data = torch.tensor([[matrix, matrix, matrix]], dtype=torch.float)
    model[0].bias.data = torch.tensor([0.])
    # kernel = get_kernel(5, 2)
    # model[1].weight.data = torch.tensor([[kernel]])
    # model[1].bias.data = torch.tensor([0.])

    image = model(image)
    image = torch.mean(image, 0)
    image = torch.unsqueeze(image, 0)
    image = torch.abs(image)
    image /= image.max()

    return image


def non_maximum_suppression(image, upper_limit, lower_limit):
    """given an image, for each pixel, if it is greater than upper limit, keep it;
    if it is less than lower limit, discard it;
    if it is surrounded with 1, keep it;
    if it is between the two, compare it with its neighbors, if it is the maximum, keep it, otherwise discard it."""
    t = 2  # the number of pixels to be ignored at the edge of the image
    image = image.squeeze()
    temp_img = torch.zeros_like(image)
    for i in range(t, image.shape[0]-t):
        for j in range(t, image.shape[1]-t):
            if image[i, j] > upper_limit:
                temp_img[i, j] = 1
            elif image[i, j] < lower_limit:
                temp_img[i, j] = 0
            else:
                temp_img[i, j] = 0.5
    for i in range(t, image.shape[0]-t):
        for j in range(t, image.shape[1]-t):
            if temp_img[i, j] == 0.5:
                neighbors = image[i-t:i+t+1, j-t:j+t+1]
                if torch.max(neighbors) == 1:
                    temp_img[i, j] = 1
                else:
                    temp_img[i, j] = 0
    return temp_img
    # result = torch.zeros_like(image)
    # t = 1  # the number of pixels to be ignored at the edge of the image
    # if it is surrounded with 1, keep it;
    # num_to_keep = int((t * 2 + 1) ** 2 * 0.5)
    # for i in range(t, image.shape[0]-t):
    #     for j in range(t, image.shape[1]-t):
    #         if temp_img[i, j] == 1:
    #             result[i, j] = 1
    #             continue
    #         neighbors = temp_img[i-t:i+t+1, j-t:j+t+1]
    #         if torch.sum(neighbors) >= num_to_keep:
    #             result[i, j] = 1
    # return result


# for the accurate pixel intensity, we need a blured image
def get_blured_img(image, kernel_size=11, sigma=2):
    # with gaussian blur
    def get_gaussian_kernel(size, theta):
        def value(x, y):
            return 1 / (2 * torch.pi * theta**2) * torch.e ** (-(x ** 2 + y ** 2) / (2 * theta ** 2))
        kernel = []
        for x in range(-(size//2), size//2+1):
            row = []
            for y in range(-(size//2), size//2+1):
                row.append(value(x, y))
            kernel.append(row)
        return kernel
    gausseian_kernel = get_gaussian_kernel(kernel_size, sigma)
    gausseian_kernel = torch.tensor([gausseian_kernel])
    gausseian_kernel = gausseian_kernel.expand(1, 1, kernel_size, kernel_size)
    image = torch.nn.functional.conv2d(image, gausseian_kernel, padding=kernel_size//2)
    image /= image.max()
    return image


if __name__ == '__main__':
    img_name = "test.jpg"
    image = load_image(img_name)[:3]
    result = process_image(image)
    result = get_blured_img(result, 11)
    threshold = torch.sort(result.reshape(-1), descending=True).values[int(0.1*result.shape[1]*result.shape[2])]
    result = torch.tensor(result > threshold, dtype=torch.float32)
    result = get_blured_img(result, 3)
    result = torch.tensor(result > threshold, dtype=torch.float32)
    result = get_blured_img(result, 5)
    result = torch.tensor(result > threshold, dtype=torch.float32)
    result = 1 - result
    print_image_info(result)
    save_image(result, "result.jpg")
