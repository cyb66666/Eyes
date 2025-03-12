import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import torch
from tqdm import tqdm
from utils.dataset import loader
from model.coattnet import coattnet_v2_withWeighted_tiny
from config import conf
from tools.focal_loss import FocalLoss

def eval(model, test_loader, criterion, threshold=0.5, device="cuda"):
    """
    在整个测试集上评估多标签分类模型，计算准确率、精确率、召回率和损失（Loss）。
    
    参数：
    - model: 训练好的 PyTorch 模型
    - test_loader: DataLoader，包含测试数据
    - criterion: 损失函数
    - threshold: 预测阈值
    - device: 运行设备（默认 "cuda"）

    返回：
    - metrics: 包含 accuracy, precision, recall, macro_precision, macro_recall, loss
    """
    model.eval()  # 进入评估模式
    all_y_true = []
    all_y_pred = []
    total_loss = 0.0  # 记录总损失
    num_batches = 0   # 记录 batch 数量

    with torch.no_grad():  # 关闭梯度计算，提高推理效率
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)  # 迁移到设备
            outputs = model(inputs)
            # print(outputs.sigmoid())
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()
            num_batches += 1

            all_y_true.append(labels.cpu())  # 存储真实标签
            all_y_pred.append(outputs.sigmoid().cpu())  # 存储预测得分

            # 在 tqdm 进度条中显示当前 batch 损失
            tqdm.write(f"Batch Loss: {loss.item():.4f}")

    # 计算平均损失
    avg_loss = total_loss / num_batches
    print(f"Avg Loss: {avg_loss:.4f}")

    # 合并所有 batch 结果
    y_true = torch.cat(all_y_true, dim=0)
    y_pred = torch.cat(all_y_pred, dim=0)
    # 计算评估指标
    metrics = multi_label_metrics(y_true, y_pred, threshold)
    metrics["loss"] = avg_loss  # 添加 loss 结果

    return metrics

def multi_label_metrics(y_true, y_pred, threshold=0.5):
    """
    计算多标签分类的准确率（Accuracy）、宏平均精确率（Macro Precision）、宏平均召回率（Macro Recall）。
    
    参数：
    - y_true: 真实标签 (Tensor) [N, C]，0或1
    - y_pred: 预测得分 (Tensor) [N, C]，值域[0,1]
    - threshold: 预测分数的阈值，超过该值则认为该类别被预测为1
    
    返回：
    - accuracy: 多标签Jaccard相似度
    - precision_per_class: 每个类别的精确率
    - recall_per_class: 每个类别的召回率
    - macro_precision: 宏平均精确率
    - macro_recall: 宏平均召回率
    """
    y_pred = (y_pred >= threshold).float()  # 二值化
    intersection = (y_true * y_pred).sum(dim=1).float()
    union = (y_true + y_pred).clamp(0, 1).sum(dim=1).float()
    accuracy = (intersection / (union + 1e-8)).mean().item()  # 避免除零
    # accuracy = (y_pred == y_true).sum(dim)
    TP = (y_true * y_pred).sum(dim=0).float()
    FP = ((1 - y_true) * y_pred).sum(dim=0).float()
    FN = (y_true * (1 - y_pred)).sum(dim=0).float()

    precision_per_class = TP / (TP + FP + 1e-8)  # 避免除零
    recall_per_class = TP / (TP + FN + 1e-8)  # 避免除零

    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()

    return {
        "accuracy": accuracy,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall
    }
 
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = coattnet_v2_withWeighted_tiny(num_classes=conf.task_layer.num_classes).to(device)
    test_loader = loader(train=False) 
    checkpoint = torch.load('results/512_100_8-0.0008-focal_loss/512_100_8-0.0008-focal_loss.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    criterion = FocalLoss()  # 使用 Focal Loss
    # criterion = torch.nn.BCEWithLogitsLoss()
    eval_results = eval(model, test_loader, criterion)

    print(eval_results)
