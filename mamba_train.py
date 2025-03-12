import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from utils.dataset import loader  # 数据加载
from model.MedMamba import VSSM  # 导入 VSSM 模型
from tools.focal_loss import FocalLoss
from config import conf
from tools.eval import eval
from tqdm import tqdm

# 1. 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载模型
model = VSSM(depths=[2, 2, 4, 2],dims=[96,192,384,768],num_classes=8).to("cuda")
if conf.train.resume:
    model.load_state_dict(torch.load(conf.train.resume))

# 3. 数据加载
train_dataloader = loader(train=True)   # 训练集
val_dataloader = loader(train=False)    # 验证集

# 4. 定义损失函数 & 优化器
criterion = FocalLoss()  # 适用于多分类任务
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # AdamW 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率调整

# 5. 训练参数
num_epochs = 30   # 训练轮数
best_val_acc = 0  # 记录最佳验证准确率

# 6. 创建日志文件夹
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 7. 训练循环
with open(log_file, "w") as log:
    log.write("Epoch,Loss,Val_Acc,Macro_Precision,Macro_Recall,Precision_Per_Class,Recall_Per_Class\n")  # 写入表头

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for images, labels in tqdm(train_dataloader, desc='Traing'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 梯度清零
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        scheduler.step()

        # 8. 评估模型
        result = eval(model, val_dataloader, criterion)

        log_msg = (
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Loss: {result['loss']:.4f} - "
            f"Val Acc: {result['accuracy']:.4f} - "
            f"Macro Precision: {result['macro_precision']:.4f} - "
            f"Macro Recall: {result['macro_recall']:.4f}"
        )
        print(log_msg)
        
        log.write(
            f"{epoch+1},{result['loss']:.4f},{result['accuracy']:.4f},"
            f"{result['macro_precision']:.4f},{result['macro_recall']:.4f},"
            f"{result['precision_per_class']},{result['recall_per_class']}\n"
        )

        # 9. 保存最佳模型
        if result['accuracy'] > best_val_acc:
            best_val_acc = result['accuracy']
            torch.save(model.state_dict(), "./checkpoint/best_vssm_model.pth")
            print("✅ Best model saved!")
            log.write("✅ Best model saved!\n")

print("🎉 Training complete!")
