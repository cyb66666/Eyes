import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dataset import loader

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        """
        :param gamma: Focal Loss 的 γ 值，控制难样本的权重，默认为 2
        :param alpha: 类别权重，可以是标量 (对所有类别一样) 或 tensor (不同类别不同权重)
        :param reduction: "mean" / "sum" / "none" 控制损失的返回方式
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.Tensor([0.0228, 0.0487, 0.1290, 0.1181, 0.1645, 0.3370, 0.1525, 0.0274]).to(device=torch.device('cuda'))
        else:
            self.alpha = None
            self.class_weigh = torch.Tensor([1.0, 2.141256, 5.666667, 5.186441, 7.228346, 14.806452, 6.70511, 1.203145]).to(device=torch.device('cuda'))
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算 Focal Loss
        :param inputs: 形状为 (batch_size, num_classes) 的 logits (未经过 sigmoid)
        :param targets: 形状为 (batch_size, num_classes) 的 0/1 二值标签
        :return: Focal Loss 值
        """
        # 计算 BCE Loss，但不做均值化
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # 计算 sigmoid 概率
        probs = torch.sigmoid(inputs)
        
        # 计算 Focal Loss 权重
        p_t = probs * targets + (1 - probs) * (1 - targets)  # 取正确类别的概率
        focal_weight = (1 - p_t) ** self.gamma

        # 计算 Focal Loss
        focal_loss = focal_weight * bce_loss
        
        # 如果使用 alpha 权重
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        else:
            focal_loss = self.class_weigh * focal_loss
        # 选择 reduction 方式
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss  # (batch_size, num_classes)


