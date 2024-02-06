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
    SCALE = 0.5528   # 每像素对应的微米数
    return pixel_value * SCALE

def calculate_feret_diameter(contour):
    max_distance = 0
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            dist = np.linalg.norm(contour[i] - contour[j])
            max_distance = max(max_distance, dist)
    return max_distance


def calculate_sphericity(area, perimeter):
    # 计算相同面积的最小周长圆形的半径和周长
    radius = np.sqrt(area / np.pi)
    p_circle = 2 * np.pi * radius

    # 计算二维形状磨圆度
    sphericity = p_circle / perimeter
    return sphericity


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


def get_low_gray_area(image, threshold=35):
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

    # 获取并打印图像分辨率
    height, width = image.shape[:2]
    print(f"图像分辨率: 宽度={width} 像素, 高度={height} 像素")

    # 裁剪图像信息条
    clipping_image = image[:-60, :]  # 假设信息条在图像底部且高度为120像素

    # 计算度量并存储结果
    total_instance_area = 0
    measurements = []
    for i in range(r['rois'].shape[0]):
        mask = r['masks'][:, :, i]
        label_img = label(mask)
        props = regionprops(label_img)[0]

        # 先以像素为单位计算
        area_pixels = props.area
        perimeter_pixels = props.perimeter
        diameter_pixels = props.major_axis_length

        # 转换为微米
        area_um = convert_to_micrometers(area_pixels)
        perimeter_um = convert_to_micrometers(perimeter_pixels)    #周长转换

        # 磨圆度
        radius_um = np.sqrt(area_um / np.pi)
        P_circle_um = 2 * np.pi * radius_um
        sphericity = perimeter_um / P_circle_um if perimeter_um > 0 else 0

        # 密实度
        solidity = props.area / props.convex_area

        # 最大粒径
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 假设每个mask只有一个轮廓
        if contours:
            feret_diameter = calculate_feret_diameter(contours[0])
            feret_diameter_um = convert_to_micrometers(feret_diameter)  # 转换为微米
        else:
            feret_diameter_um = 0

        measurements.append([i + 1, area_um, feret_diameter_um, sphericity, solidity])
        total_instance_area += area_pixels  #累加每个实例的面积

    df = pd.DataFrame(measurements, columns=['实例序号', '面积', '最大直径', '磨圆度', '密实度'])
    df.to_excel(output_path, index=False)

    if show_image:
        colors = generate_random_colors(len(r['class_ids']) + 1)
        masked_image = image.astype(np.uint8).copy()
        for i in range(len(r['class_ids'])):
            color = colors[i]
            mask = r['masks'][:, :, i]
            # 找到每个掩码的轮廓
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 在原始图像上绘制轮廓
            cv2.drawContours(masked_image, contours, -1, color, 2)  # 2是轮廓的线宽

        # 显示灰度值小于50的区域
        low_gray_mask, low_gray_area = get_low_gray_area(image)
        # masked_image = apply_mask(masked_image, low_gray_mask, colors[-1], low_gray=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(masked_image.astype(np.uint8))
        plt.axis('off')

        plt.show()

    image_area = clipping_image.shape[0] * clipping_image.shape[1]
    remaining_area = image_area - total_instance_area - low_gray_area

    instance_area_percentage = (total_instance_area / image_area) * 100
    low_gray_area_percentage = (low_gray_area / image_area) * 100
    remaining_area_percentage = (remaining_area / image_area) * 100

    print("大颗粒面积百分比:", instance_area_percentage)
    print("孔洞面积百分比:", low_gray_area_percentage)
    print("粘土面积百分比:", remaining_area_percentage)

if __name__ == "__main__":
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    tensorboard_dir = sys.argv[4]   # 添加 TensorBoard 日志目录为参数
    measure_instances(image_path, model_path, output_path, show_image=True)
