import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
# 定义标签及其映射
labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
label_map = {label: index for index, label in enumerate(labels)}
# 解决 CUDA 进程问题
mp.set_start_method('spawn', force=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Using device: {device}")

class EyeDataset(Dataset):
    def __init__(self, csv_path, img_folder, target_size=(640, 640), train=True):
        self.df = pd.read_csv(csv_path)
        self.img_folder = img_folder
        self.target_size = target_size
        self.train = train

        # 只用前 10 条数据测试
        self.df = self.df[:10]  

        # 数据集划分
        train_data, val_data = train_test_split(self.df, test_size=0.2, random_state=42)
        self.data = train_data if train else val_data

        # 数据增强
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  # 平移2%
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def load_image(self, filename):
        img_path = os.path.join(self.img_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 无法加载图像: {img_path}")
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 读入 BGR，转换 RGB
        return img

    

    def __getitem__(self, index):
        row = self.data.iloc[index]
        left_img = self.load_image(row["左眼图片名称"])
        right_img = self.load_image(row["右眼图片名称"])

        # 处理分类字段（可能是 "A,B" 这种格式）
        label_str = row["分类"]
        label_list = label_str.split(",")  # 按 "," 拆分
        label_main = label_list[0]  # 取第一个类别（可以改为其他策略）

        # 创建独热编码
        one_hot_label = np.zeros(len(labels), dtype=np.float32)  # 创建全零数组
        if label_main in label_map:  # 如果标签存在于映射中
            one_hot_label[label_map[label_main]] = 1  # 将对应位置设置为 1

        # 复制原始图像用于可视化
        original_left_img = left_img.copy()
        original_right_img = right_img.copy()

        # 数据增强
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        return (original_left_img, original_right_img), (left_img, right_img), torch.tensor(one_hot_label, dtype=torch.float32)


# 数据可视化
def show_images(original_imgs, transformed_imgs, labels):
    fig, axes = plt.subplots(len(original_imgs), 4, figsize=(10, len(original_imgs) * 2.5))
    for i in range(len(original_imgs)):
        original_left, original_right = original_imgs[i]
        transformed_left, transformed_right = transformed_imgs[i]

        # 左眼 - 原始 & 变换后
        axes[i, 0].imshow(original_left)
        axes[i, 0].set_title(f"原始左眼\nLabel: {labels[i]}")
        axes[i, 1].imshow(transformed_left.permute(1, 2, 0))  # (C,H,W) -> (H,W,C)
        axes[i, 1].set_title("增强后左眼")

        # 右眼 - 原始 & 变换后
        axes[i, 2].imshow(original_right)
        axes[i, 2].set_title("原始右眼")
        axes[i, 3].imshow(transformed_right.permute(1, 2, 0))  # (C,H,W) -> (H,W,C)
        axes[i, 3].set_title("增强后右眼")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

# 运行测试
if __name__ == '__main__':
    csv_path = "./data/merged_image_info.csv"
    img_folder = "./data/Cropped_Dataset"

    # 获取小批量数据集
    dataset = EyeDataset(csv_path, img_folder, target_size=(640, 640), train=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # 取一个 batch 进行测试
    for (original_imgs, transformed_imgs, labels) in dataloader:
        labels = labels.numpy()
        show_images(original_imgs, transformed_imgs, labels)
        break  # 只显示一个 batch
