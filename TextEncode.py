import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import json
from tqdm import tqdm  # 添加进度条显示

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
df = pd.read_excel("data/Traning_Dataset.xlsx")

# 选择所需的列
df = df[['ID', 'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords', 
         'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]  # 包括标签列

# 构建数据列表
data_list = []
for i in tqdm(range(len(df)), desc="Processing rows"):  # 使用 tqdm 显示进度条
    row = df.iloc[i]  # 按行提取数据
    
    # 生成标签列表
    labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    label_list = [int(row[label]) for label in labels]  # 根据每列值生成 0/1 标签列表
    
    data_list.append({
        'ID': int(row['ID']),  # 确保 ID 转为 Python 原生 int 类型
        'Left-Keywords': TextEncode(row['Left-Diagnostic Keywords']),
        'Right-Keywords': TextEncode(row['Right-Diagnostic Keywords']),
        'label': label_list  # 添加标签字段
    })

# 保存为 JSON 文件
with open("data/output.json", "w", encoding="utf-8") as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)

print("数据已成功保存到 data/output.json")
