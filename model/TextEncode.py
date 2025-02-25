import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import json
from tqdm import tqdm  # 添加进度条显示
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import conf
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertModel.from_pretrained("bert-large-uncased")
model = model.to(device)

# 定义文本编码函数
def TextEncode(text):
    inputs = tokenizer(
        text,
        padding=True,        # 添加填充
        truncation=True,     # 截断
        max_length=128,      # 最大长度
        return_tensors="pt"  # 返回 PyTorch 张量
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():  # 不计算梯度，节省资源
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # 提取 [CLS] 向量
    return cls_embeddings.cpu().numpy().tolist()  # 转为 NumPy，再转为列表

# 读取 Excel 文件
df = pd.read_csv(conf.data_path.csv)

# 选择所需的列
df = df[['ID', 'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords', 
         'label']]  # 包括标签列

labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# 构建数据列表
data_list = []
for i in tqdm(range(len(df)), desc="Processing rows"):  # 使用 tqdm 显示进度条
    row = df.iloc[i]  # 按行提取数据

    # 解析标签列，支持多个标签（如 "N,D,C"）
    row_labels = str(row['label']).split(',')  # 确保为字符串再拆分
    label_list = [1 if lbl in row_labels else 0 for lbl in labels]  # 生成 one-hot 编码

    data_list.append({
        'ID': int(row['ID']),  # 确保 ID 转为 Python 原生 int 类型
        'label': label_list,  # 存储 one-hot 编码
        'Left-Keywords': TextEncode(row['Left-Diagnostic Keywords']),
        'Right-Keywords': TextEncode(row['Right-Diagnostic Keywords'])
    })

# 保存为 JSON 文件
with open(conf.data_path.text_embedding, "w", encoding="utf-8") as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)

print(f"数据已成功保存到 {conf.data_path.text_embedding}")
