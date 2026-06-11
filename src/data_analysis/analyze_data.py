import pandas as pd

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 读取CSV文件
df = pd.read_csv('used_car_train_20200313.csv', sep=' ')

print("=== 数据集基本信息 ===")
print(f"数据形状: {df.shape}")
print(f"\n列数: {len(df.columns)}")
print(f"\n各列数据类型:\n")
print(df.dtypes)
print(f"\n=== 缺失值统计 ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing,
    '缺失比例(%)': round(missing_pct, 2)
})
print(missing_df[missing_df['缺失数量'] > 0])
print(f"\n=== 数值型字段统计信息 ===")
print(df.describe())
