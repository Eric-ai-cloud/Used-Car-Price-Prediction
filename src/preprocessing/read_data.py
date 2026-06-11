import pandas as pd

# 设置显示选项，显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 读取CSV文件（尝试不同的分隔符）
try:
    df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
except:
    df = pd.read_csv('used_car_train_20200313.csv')

# 打印前5行数据
print("前5行数据：")
print(df.head())

# 显示数据形状和列信息
print(f"\n数据形状: {df.shape}")
print(f"\n列名列表:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")
