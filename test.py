import pandas as pd

# 读取 Excel 文件（假设文件名为 data.xlsx）
df = pd.read_excel("data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx")

# 需要统计的类别列（N, D, G, C, A, H, M, O）
category_columns = ["N", "D", "G", "C", "A", "H", "M", "O"]

# 计算每个类别的数量
category_counts = df[category_columns].sum()

# 计算比例（归一化，使总和为1）
category_ratios = category_counts / category_counts.sum()

# 打印结果
print("类别数量:\n", category_counts)
print("\n类别比例:\n", category_ratios)
