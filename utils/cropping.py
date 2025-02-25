import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import torch
import cv2
import os
import numpy as np
from config import conf

# 设定图片来源和保存路径
source_path = conf.data_path.Training_Dataset  # 原始图片目录
save_path = conf.data_path.Cropped_Dataset  # 处理后图片目录

# 确保保存目录存在
os.makedirs(save_path, exist_ok=True)

def crop_black_borders_torch(image):
    """ 使用 PyTorch 在 GPU 上去除黑边 """
    if not isinstance(image, np.ndarray):
        raise ValueError("输入的 image 必须是 NumPy 数组")

    # 转换为 Tensor
    img_tensor = torch.from_numpy(image).float().cuda()

    # 转换为灰度图
    img_gray = 0.299 * img_tensor[:, :, 0] + 0.587 * img_tensor[:, :, 1] + 0.114 * img_tensor[:, :, 2]

    # 进行二值化（设定阈值 > 5 作为非黑区域）
    binary_image = (img_gray > 5).float()

    # 计算行列的黑色区域
    rows = torch.nonzero(binary_image.sum(dim=1) > 0).squeeze()
    cols = torch.nonzero(binary_image.sum(dim=0) > 0).squeeze()

    if rows.numel() == 0 or cols.numel() == 0:
        return image  # 如果没有非黑区域，返回原图

    # 找到裁剪边界
    top, bottom = rows.min().item(), rows.max().item()
    left, right = cols.min().item(), cols.max().item()

    # 裁剪
    cropped = image[top:bottom+1, left:right+1]

    return cropped

if __name__ == "__main__":
    for filename in os.listdir(source_path):
        # 只处理图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_path, filename)
            save_img_path = os.path.join(save_path, filename)

            # 读取图片
            image = cv2.imread(img_path)
            if image is None:
                print(f"跳过无法读取的文件: {filename}")
                continue
            
            # 进行裁剪
            cropped_image = crop_black_borders_torch(image)

            # 保存裁剪后的图片
            cv2.imwrite(save_img_path, cropped_image)
            print(f"处理完成: {filename} -> {save_img_path}")

    print("所有图片处理完成！")
