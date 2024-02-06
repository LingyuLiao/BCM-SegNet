import cv2
import numpy as np
import pandas as pd
import os
import sys
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPoint
from typing import List

ROOT_DIR = os.path.abspath("../../")
print(ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from nucleus import NucleusInferenceConfig
import tensorboard
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class InferenceConfig(NucleusInferenceConfig):
    NAME = "my_dataset"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 类别数
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


def convert_to_micrometers(pixel_value):
    SCALE = 0.5528
    return pixel_value * SCALE


def calculate_sphericity(area, perimeter):
    # 计算相同面积的最小周长圆形的半径和周长
    radius = np.sqrt(area / np.pi)
    p_circle = 2 * np.pi * radius

    # 计算二维形状磨圆度
    sphericity = p_circle / perimeter
    return sphericity


def convex_hull(points):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(set(points))
    if len(points) <= 1:
        return points

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def calculate_solidity(mask):
    # Find contours of the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0

    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour)

    # Create a blank image filled with zeros
    hull_image = np.zeros_like(mask, dtype=np.uint8)

    # Draw the hull contour onto the blank image
    cv2.drawContours(hull_image, [hull], 0, (1), -1)

    a_particle = np.sum(mask)
    a_hull = np.sum(hull_image)

    if a_hull == 0:
        return 0

    solidity = a_particle / a_hull
    return solidity


def generate_random_colors(n):
    """
    生成 n 个不同的随机颜色
    :param n: 需要生成的颜色数量
    :return: 随机颜色列表，每个颜色由 RGB 三个通道的值组成
    """
    colors = []
    for i in range(n):
        color = [random.randint(0, 255) for _ in range(3)]
        colors.append(color)
    return colors


def apply_mask(image, mask, color, alpha=0.5, low_gray=False):
    """
    将掩码应用到图像上，并返回带有掩码的新图像
    :param image: 原始图像
    :param mask: 需要应用的掩码
    :param color: 掩码颜色
    :param alpha: 控制掩码透明度的参数
    :param low_gray: 是否为灰度值小于20的掩码
    :return: 应用掩码后的新图像
    """
    new_image = image.copy()
    for c in range(3):
        if low_gray:
            new_image[:, :, c] = np.where(mask == 1,
                                          color[c] * 255,
                                          new_image[:, :, c])
        else:
            new_image[:, :, c] = np.where(mask == 1,
                                          new_image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                          new_image[:, :, c])
    return new_image


def get_low_gray_area(image, threshold=50):
    """
    获取图像中灰度值小于阈值的部分，并返回相应的掩码和面积
    :param image: 原始图像
    :param threshold: 灰度值阈值
    :return: 掩码和面积
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray_image < threshold, 1, 0).astype(np.uint8)
    area = np.sum(mask)
    return mask, area


def measure_instances(image_path, model_path, output_path, show_image=True):
    # 加载模型和权重
    model_dir = os.path.dirname(model_path)
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    model.load_weights(model_path, by_name=True)

    # 加载图片并进行实例分割
    image = cv2.imread(image_path)
    results = model.detect([image], verbose=1)
    r = results[0]

    # Apply masks to the original image to create an image with only the instances
    sand_particle = np.zeros_like(image)
    for i in range(r['masks'].shape[-1]):
        mask = r['masks'][:, :, i]
        sand_particle[mask] = image[mask]

    # Crop the information bar from the image and sand particle image
    clipping_image = image[:-60, :]  # Assuming the info bar is 120 pixels in height
    sand_particle_clipping = sand_particle[:-60, :]

    # Extract and calculate the area of voids
    gray_image_clipping = cv2.cvtColor(clipping_image, cv2.COLOR_BGR2GRAY)
    voids_mask = gray_image_clipping <= 35
    voids_image = np.zeros_like(clipping_image)
    voids_image[voids_mask] = clipping_image[voids_mask]
    voids_area = np.sum(voids_mask)

    # Calculate the matrix by subtracting sand particles and voids from the clipping image
    matrix_mask = ~(voids_mask | (sand_particle_clipping.any(axis=-1)))
    matrix = np.zeros_like(clipping_image)
    matrix[matrix_mask] = clipping_image[matrix_mask]
    matrix_area = np.sum(matrix_mask)

    # Save the cropped image and the separate parts
    cv2.imwrite(os.path.join(output_path, 'clipping_image.png'), clipping_image)
    cv2.imwrite(os.path.join(output_path, 'sand_particle.png'), sand_particle_clipping)
    cv2.imwrite(os.path.join(output_path, 'voids.png'), voids_image)
    cv2.imwrite(os.path.join(output_path, 'matrix.png'), matrix)

    # Show the image if required
    if show_image:
        plt.figure(figsize=(10, 10))
        plt.imshow(matrix)
        plt.axis('off')
        plt.show()

    # Output the area information
    sand_particle_area = np.sum(sand_particle_clipping.any(axis=-1))
    print(f"Sand particle area: {sand_particle_area}")
    print(f"Voids area: {voids_area}")
    print(f"Matrix area: {matrix_area}")


if __name__ == "__main__":
    image_path = "F:/TaoFan_Beikeda/1/D_mould.png"
    model_path = "F:/TaoFan_Beikeda/USTB_mask_r-cnn/samples/bit/model.h5"
    output_path = "F:/TaoFan_Beikeda/USTB_mask_r-cnn/samples/bit/three_parts/D_mould"
    measure_instances(image_path, model_path, output_path, show_image=False)
