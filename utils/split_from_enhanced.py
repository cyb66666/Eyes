import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import shutil
import random
from sklearn.model_selection import train_test_split
from config import conf

# 读取增强后的数据集路径
dataset_dir = conf.data_path.Enhanced_Dataset
train_dir = conf.data_path.train
test_dir = conf.data_path.test

# **确保目录存在，并清空训练集和测试集目录**
def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # 删除整个文件夹
    os.makedirs(directory)  # 重新创建文件夹

clear_directory(train_dir)
clear_directory(test_dir)

# 获取所有图片文件名
image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# **打乱数据集**
random.shuffle(image_files)

# 按 4:1 分割数据集
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

# 复制文件到训练集和测试集
def copy_files(file_list, src_dir, dest_dir):
    for file in file_list:
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy(src_path, dest_path)

copy_files(train_files, dataset_dir, train_dir)
copy_files(test_files, dataset_dir, test_dir)

print(f"数据集划分完成: 训练集 {len(train_files)} 张, 测试集 {len(test_files)} 张")
