import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import conf
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
target_brightness = 120
center_ratio = 0.7
# df = pd.read_csv(conf.data_path.csv)

def Resize_image(image):
    """
    统一调整图片尺寸为224x224
    """
    return cv2.resize(image, conf.image_size)

def apply_augmentation(image):
    """
    应用数据增强方法
    """
    # 1. 对比度增强：CLAHE处理
    # if len(image.shape) == 3 and image.shape[2] == 3:  # 彩色图像
    #     if image.shape[2] == 3 and image[0, 0, 0] <= 255:  # 检查是否为RGB格式
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转为BGR格式
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
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
    balanced_image = adjust_brightness(balanced_image)

    # 5. 去噪、效果不好
    # balanced_image = cv2.fastNlMeansDenoisingColored(balanced_image, None, 10, 10, 1, 3)

    return balanced_image

def adjust_brightness(image):
    """
    将图像中心区域（center_ratio区域）的平均亮度调整到 target_brightness，
    然后用该调整因子对整个图像的亮度进行调整。
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    ew, eh = int(w * center_ratio), int(h * center_ratio)
    x1, y1 = max(cx - ew // 2, 0), max(cy - eh // 2, 0)
    x2, y2 = min(cx + ew // 2, w), min(cy + eh // 2, h)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    center_region = v_channel[y1:y2, x1:x2]
    avg_brightness = np.mean(center_region)

    factor = target_brightness / (avg_brightness + 1e-5)
    factor = max(min(factor, 2.0), 0.5)

    v_new = np.clip(v_channel * factor, 0, 255).astype(np.uint8)

    hsv_adjusted = cv2.merge([h_channel, s_channel, v_new])
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    return adjusted_image
        
def load_images_and_enhance(csv_path):
    img_folder = conf.data_path.Cropped_Dataset
    df = pd.read_csv(csv_path)
    for _, row in tqdm(df.iterrows(),total=len(df),desc="Enhancing Images:"):
        left_filename = row['左眼图片名称']
        right_filename = row['右眼图片名称']
        ID = row["ID"]
        left_path = os.path.join(img_folder, left_filename)
        right_path = os.path.join(img_folder, right_filename)
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        left_img = apply_augmentation(Resize_image(left_img))
        right_img = apply_augmentation(Resize_image(right_img))
        if left_img is not None and right_img is not None:
            img = cv2.hconcat([left_img, right_img])
            save_path = os.path.join(conf.data_path.Enhanced_Dataset, f"{ID}.jpg")
            cv2.imwrite(save_path, img)
        else:
            print(f"警告：无法读取 {left_path} 或 {right_path}")
    print("增强结束")

if __name__ == "__main__":
    csv_path = conf.data_path.csv
    load_images_and_enhance(csv_path)