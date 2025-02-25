import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=128):
        super(ImageEncoder, self).__init__()
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 定义全连接层
        self.fc1 = nn.Linear(256 * 16 * 16, 512)  # 假设输入图像大小为 256x256
        self.fc2 = nn.Linear(512, output_dim)
        
        # 定义激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 卷积层 + 池化层
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        
        # 展平操作
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
