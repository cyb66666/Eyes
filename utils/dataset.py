import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random

# 数据增强处理：CLAHE、亮度均衡以及随机几何变换
import cv2
import numpy as np
import random



def adjust_brightness(image, target_brightness=70, center_ratio=0.7):
    """
    将图像中心区域（center_ratio区域）的平均亮度调整到 target_brightness，
    然后用该调整因子对整个图像的亮度进行调整。

    参数:
        target_brightness: 目标亮度值（0~255）。
        center_ratio: 用于计算平均亮度的中心区域比例（例如0.7表示70%的中心区域）。
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    # 计算中心区域的尺寸（宽、高）
    ew, eh = int(w * center_ratio), int(h * center_ratio)
    # 定义中心区域的左上角和右下角坐标
    x1, y1 = max(cx - ew // 2, 0), max(cy - eh // 2, 0)
    x2, y2 = min(cx + ew // 2, w), min(cy + eh // 2, h)

    # 转换到HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # 计算中心区域的平均亮度（V通道均值）
    center_region = v_channel[y1:y2, x1:x2]
    avg_brightness = np.mean(center_region)

    # 计算调整因子
    factor = target_brightness / (avg_brightness + 1e-5)
    # 限制因子范围（防止过度增强或减弱），你可以根据需求调整上下限
    factor = max(min(factor, 2.0), 0.5)

    # 对整个图像的 V 通道进行调整
    v_new = np.clip(v_channel * factor, 0, 255).astype(np.uint8)

    # 合并通道并转换回BGR空间
    hsv_adjusted = cv2.merge([h_channel, s_channel, v_new])
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    return adjusted_image


def apply_augmentation(image):
    # 1. 对比度增强：CLAHE处理
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahe_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. 亮度均衡：转换到HSV后对v通道均衡化
    hsv = cv2.cvtColor(clahe_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v])
    balanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. 混合原图与增强图，保留原始细节
    balanced_image = cv2.addWeighted(balanced_image, 0.1, image, 0.90, 0)

    # 4. 对图像中心区域亮度检测与调整（智能增亮）
    balanced_image = adjust_brightness(balanced_image, target_brightness=100, center_ratio=0.7)

    # 5. 随机旋转 -10 到 10 度
    angle = random.uniform(-10, 10)
    h_img, w_img = balanced_image.shape[:2]
    M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), angle, 1)
    balanced_image = cv2.warpAffine(balanced_image, M, (w_img, h_img))

    # 6. 随机平移（左右上下各最多移动2%的宽高）
    tx = random.uniform(-0.02, 0.02) * w_img
    ty = random.uniform(-0.02, 0.02) * h_img
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    balanced_image = cv2.warpAffine(balanced_image, M_trans, (w_img, h_img))

    return balanced_image


# 预处理函数：统一调整图片尺寸为640x640
def preprocess_image(image, target_size=(640, 640)):
    return cv2.resize(image, target_size)


# 读取CSV并加载图像对及分类标签，同时保存原始文件名
def load_images_from_csv(csv_path, img_folder, test_size=0.2):
    df = pd.read_csv(csv_path)
    image_pairs = []

    for index, row in df.iterrows():
        left_filename = row['左眼图片名称']
        right_filename = row['右眼图片名称']
        label = row['分类']  # 分类标签

        left_path = os.path.join(img_folder, left_filename)
        right_path = os.path.join(img_folder, right_filename)
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is not None and right_img is not None:
            image_pairs.append((left_img, right_img, label, left_filename, right_filename))
        else:
            print(f"警告：无法读取 {left_path} 或 {right_path}")

    train_pairs, val_pairs = train_test_split(image_pairs, test_size=test_size, random_state=42)
    return train_pairs, val_pairs


# 数据生成器：返回左右眼图像、标签和原始文件名
def data_generator(image_pairs, batch_size=32, target_size=(640, 640)):
    while True:
        batch_left = []
        batch_right = []
        batch_labels = []
        batch_left_names = []
        batch_right_names = []
        image_pairs = shuffle(image_pairs)

        for i in range(batch_size):
            left_img, right_img, label, left_filename, right_filename = image_pairs[i]

            # 统一尺寸
            left_img = preprocess_image(left_img, target_size)
            right_img = preprocess_image(right_img, target_size)

            # 数据增强
            left_img = apply_augmentation(left_img)
            right_img = apply_augmentation(right_img)

            batch_left.append(left_img)
            batch_right.append(right_img)
            batch_labels.append(label)
            batch_left_names.append(left_filename)
            batch_right_names.append(right_filename)

        yield [np.array(batch_left), np.array(batch_right)], np.array(batch_labels), batch_left_names, batch_right_names


# 构建数据集生成器
def build_dataset_and_augment(csv_path, img_folder, test_size=0.2, batch_size=32, target_size=(640, 640)):
    train_pairs, val_pairs = load_images_from_csv(csv_path, img_folder, test_size)
    train_gen = data_generator(train_pairs, batch_size, target_size)
    val_gen = data_generator(val_pairs, batch_size, target_size)
    return train_gen, val_gen


if __name__ == '__main__':
    csv_path = './data/merged_image_info.csv'  # 替换为你的CSV文件路径
    img_folder = './data/Cropped_Dataset'  # 替换为你的图片文件夹路径
    save_folder = './data/Enhanced_Dataset'  # 输出文件夹
    os.makedirs(save_folder, exist_ok=True)

    # 设置batch_size为20
    train_gen, _ = build_dataset_and_augment(csv_path, img_folder, test_size=0.2, batch_size=16, target_size=(640, 640))

    # 从生成器中获取一个batch数据
    [batch_left, batch_right], batch_labels, batch_left_names, batch_right_names = next(train_gen)

    # 对每一对左右图像添加文件名称，并保存显示
    for i in range(16):
        left_img = batch_left[i].copy()  # 防止修改原数组
        right_img = batch_right[i].copy()
        left_name = batch_left_names[i]
        right_name = batch_right_names[i]

        # 在图像上添加文字
        cv2.putText(left_img, left_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(right_img, right_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 拼接左右图像（水平拼接）
        combined = np.hstack([left_img, right_img])

        # 保存图片，例如：sample_0.jpg, sample_1.jpg, ...
        save_path = os.path.join(save_folder, f"sample_{i}.jpg")
        cv2.imwrite(save_path, combined)
        print(f"保存: {save_path}")