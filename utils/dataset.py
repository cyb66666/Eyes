import random
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm
from config import conf
import glob

class EyeDataset(Dataset):
    def __init__(self,train=True):
        if train:
            self.image_path = conf.data_path.train
        else:
            self.image_path = conf.data_path.test
        self.csv = pd.read_csv(conf.data_path.csv)
        self.data = self.get_path_label()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.ToTensor()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id_value = self.data[idx]["ID"]
        img_path = self.image_path + f"/{id_value}.jpg"
        label = self.data[idx]["label"]
        img = self.transform(Image.open(img_path))
        return img, label

    def get_path_label(self):
        """
        解析CSV，匹配self.image_path目录下的jpg文件，并返回ID到label的映射
        """
        jpg_files = set(glob.glob(os.path.join(self.image_path, "*.jpg")))  # 获取目录下所有 .jpg 文件
        jpg_filenames = [int(os.path.basename(f)[0:-4]) for f in jpg_files]  # 仅保留文件名

        # 选择需要的标签列
        label_columns = ["N", "D", "G", "C", "A", "H", "M", "O"]

        # 过滤CSV数据，使其包含当前目录下的图片
        filtered_data = self.csv[self.csv["ID"].isin(jpg_filenames)]

        # 创建 {ID: label} 映射
        id_label_map = []
        for _, row in filtered_data.iterrows():
            img_id = row["ID"]
            # 将标签转换为 torch.float32 类型
            label = torch.tensor(row[label_columns].tolist(), dtype=torch.float32)
            id_label_map.append({
                "ID": img_id,
                "label": label
            })

        return id_label_map
    
def loader(train=True):
    return DataLoader(EyeDataset(train), batch_size=conf.batch_size, shuffle=True)

if __name__ == '__main__':
    train_dataloader = DataLoader(EyeDataset(True), batch_size=conf.batch_size, shuffle=True)
    test_dataloader = DataLoader(EyeDataset(False), batch_size=conf.batch_size, shuffle=True)
    # dataset = EyeDataset()
    # print(dataset[0][0].size())
    for imgs, labels in train_dataloader:
        print(imgs.size(), labels.size())
        break
