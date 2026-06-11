import pandas as pd

print("验证预处理后的数据...")

train = pd.read_csv('train_preprocessed.csv')
test = pd.read_csv('test_preprocessed.csv')

print(f'\n训练集形状: {train.shape}')
print(f'测试集形状: {test.shape}')

print('\n缺失值检查:')
print(f'训练集缺失值总数: {train.isnull().sum().sum()}')
print(f'测试集缺失值总数: {test.isnull().sum().sum()}')

print('\n标准化验证（前5个数值特征）:')
num_cols = [col for col in train.columns if col not in ['SaleID', 'price'] and train[col].dtype in ['float64', 'int64']][:5]
for col in num_cols:
    print(f'  {col}: 均值={train[col].mean():.4f}, 标准差={train[col].std():.4f}')

print('\n✅ 验证完成！')
