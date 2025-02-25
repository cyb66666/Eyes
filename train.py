import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================
# 1. 定义数据集类
# ==========================
class EyeDataset(Dataset):
    def __init__(self, json_path):
        # 读取 JSON 文件（假设文件内容是一个列表，每个元素形如：
        # {
        #   "ID": 1,
        #   "Left-Keywords": [0.12, 0.34, ..., 0.56],
        #   "Right-Keywords": [0.78, 0.90, ..., 0.12],
        #   "label": [0, 0, 1, 0, 0, 0, 0, 0]  # one-hot，只有一个位置为1
        # }
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 将左右眼关键字编码转换为 tensor，假设每个编码为一维列表，长度为 emb_dim（例如1024）
        left_embedding = torch.tensor(sample['Left-Keywords'], dtype=torch.float)   # shape: (emb_dim,)
        right_embedding = torch.tensor(sample['Right-Keywords'], dtype=torch.float) # shape: (emb_dim,)
        # 标签为 8 维 one-hot 向量
        label = torch.tensor(sample['label'], dtype=torch.float)  # shape: (num_classes,)
        return left_embedding, right_embedding, label

# ==========================
# 2. 定义模型（多类别分类模型）
# ==========================
class EyeClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_classes=8):
        """
        emb_dim: 每只眼睛编码的维度（例如1024）
        hidden_dim: 隐藏层维度
        num_classes: 类别数（本例为8）
        """
        super(EyeClassifier, self).__init__()
        # 拼接左右眼编码后，输入维度为 2 * emb_dim
        self.fc1 = nn.Linear(2 * emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # 输出层输出类别得分（logits），未经过激活
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, left, right):
        # left, right shape: (batch_size, emb_dim)
        # 注意：这里不要使用 squeeze，否则当 batch_size=1 时会丢失 batch 维度
        x = torch.cat([left, right], dim=-1)  # 拼接后 shape: (batch_size, 2*emb_dim)\
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # shape: (batch_size, num_classes)
        return torch.squeeze(logits)

# ==========================
# 3. 辅助函数：计算准确率、召回率及混淆矩阵绘制
# ==========================
def onehot_to_index(onehot_labels):
    """
    将 one-hot 标签转换为类别索引
    输入 shape: (batch_size, num_classes)
    """
    # print(onehot_labels)
    # print(torch.argmax(onehot_labels, dim=1))
    return torch.argmax(onehot_labels, dim=1)

def calculate_accuracy(logits, onehot_labels):
    """
    计算多类别分类的准确率
    logits: 模型输出 (batch_size, num_classes)
    onehot_labels: one-hot 格式标签 (batch_size, num_classes)
    """
    preds = torch.argmax(logits, dim=1)
    true = onehot_to_index(onehot_labels)
    correct = (preds == true).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy

def calculate_recall(logits, onehot_labels):
    """
    计算多类别分类的召回率（宏平均）
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    true = onehot_to_index(onehot_labels).cpu().numpy()
    num_classes = onehot_labels.shape[1]
    recalls = []
    for cls in range(num_classes):
        tp = np.sum((true == cls) & (preds == cls))
        fn = np.sum((true == cls) & (preds != cls))
        recall = tp / (tp + fn + 1e-8)
        recalls.append(recall)
    return np.mean(recalls)

def plot_confusion_matrix_multiclass(logits, onehot_labels, num_classes=8):
    """
    绘制多类别混淆矩阵 (8x8)
    将 logits 和 one-hot 标签转换为类别索引后计算混淆矩阵
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    true = onehot_to_index(onehot_labels).cpu().numpy()
    
    cm = confusion_matrix(true, preds, labels=list(range(num_classes)))
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (8x8)')
    plt.show()

# ==========================
# 4. 主程序：数据划分、训练和评估
# ==========================
if __name__ == "__main__":
    # 参数设置
    emb_dim = 1024      # 每只眼睛编码的维度
    hidden_dim = 512
    num_classes = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集，文件路径请自行修改
    dataset = EyeDataset("data/output.json")
    
    # 划分数据集（例如 80% 训练，20% 测试）
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # # 实例化模型
    model = EyeClassifier(emb_dim, hidden_dim, num_classes).to(device)
    
    # # 使用 CrossEntropyLoss 训练模型，目标标签需要为类别索引
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # # 训练模型
    # model.train()
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     total_loss = 0.0
    #     total_accuracy = 0.0
    #     total_recall = 0.0
    #     for batch_idx, (left_embed, right_embed, labels_onehot) in enumerate(train_dataloader):
    #         left_embed = left_embed.to(device)
    #         right_embed = right_embed.to(device)
    #         labels_onehot = labels_onehot.to(device)
    #         # 将 one-hot 标签转换为类别索引
    #         labels = onehot_to_index(labels_onehot)
            
    #         optimizer.zero_grad()
    #         logits = model(left_embed, right_embed)  # 输出 shape: (batch_size, num_classes)

    #         print(logits.shape)
    #         print(labels.shape)

    #         loss = criterion(logits, labels)
    #         loss.backward()
    #         optimizer.step()
            
    #         batch_acc = calculate_accuracy(logits, labels_onehot)
    #         batch_rec = calculate_recall(logits, labels_onehot)
    #         total_loss += loss.item()
    #         total_accuracy += batch_acc.item()
    #         total_recall += batch_rec
        
    #     avg_loss = total_loss / len(train_dataloader)
    #     avg_acc = total_accuracy / len(train_dataloader)
    #     avg_rec = total_recall / len(train_dataloader)
    #     print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Train Accuracy = {avg_acc:.4f}, Train Recall = {avg_rec:.4f}")
    # torch.save(model.state_dict(), "model.pth")
    # 在测试集上评估并绘制混淆矩阵
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for left_embed, right_embed, labels_onehot in test_dataloader:
            left_embed = left_embed.to(device)
            right_embed = right_embed.to(device)
            labels_onehot = labels_onehot.to(device)
            logits = model(left_embed, right_embed)
            if logits.shape[0] != 16:
                break
            all_logits.append(logits)
            all_labels.append(labels_onehot)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    test_acc = calculate_accuracy(all_logits, all_labels)
    test_rec = calculate_recall(all_logits, all_labels)
    print(f"Test Accuracy = {test_acc:.4f}, Test Recall = {test_rec:.4f}")
    
    plot_confusion_matrix_multiclass(all_logits, all_labels, num_classes=num_classes)
