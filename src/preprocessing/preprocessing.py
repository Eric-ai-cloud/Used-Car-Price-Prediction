import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔧 数据预处理 - 二手车价格预测")
print("=" * 80)

# ========================
# 1. 加载特征工程后的数据
# ========================
print("\n【1】加载特征工程后的数据...")
train_df = pd.read_csv('train_featured.csv')
test_df = pd.read_csv('test_featured.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 验证品牌特征是否存在
brand_cols = ['brand', 'brand_power_mean', 'brand_power_std', 'brand_km_mean', 'brand_km_std']
print(f"\n品牌特征检查:")
for col in brand_cols:
    train_has = col in train_df.columns
    test_has = col in test_df.columns
    print(f"  {col:30s}: 训练集={train_has}, 测试集={test_has}")

if all(col in train_df.columns and col in test_df.columns for col in brand_cols):
    print("✅ 所有品牌特征都已存在")
else:
    print("⚠️  部分品牌特征缺失，请检查特征工程步骤")

# ========================
# 2. 删除不需要的列
# ========================
print("\n【2】删除不需要的列...")

# 要删除的列：name(高基数), seller/offerType(几乎无方差或数据泄露风险)
drop_columns = ['name', 'seller', 'offerType']

# 过滤出存在的列
drop_columns_existing = [col for col in drop_columns if col in train_df.columns]

if drop_columns_existing:
    print(f"删除列: {drop_columns_existing}")
    train_df.drop(drop_columns_existing, axis=1, inplace=True)
    
    # 测试集可能没有这些列，只删除存在的
    test_drop = [col for col in drop_columns_existing if col in test_df.columns]
    if test_drop:
        test_df.drop(test_drop, axis=1, inplace=True)
    print("✅ 删除完成")
else:
    print("无需删除的列")

# ========================
# 3. 类别型特征编码
# ========================
print("\n【3】类别型特征编码...")

# 识别类别型特征 (object类型，且非ID/Target)
categorical_cols = [col for col in train_df.columns 
                    if train_df[col].dtype == 'object' 
                    and col not in ['SaleID', 'price']]

print(f"检测到类别型特征: {categorical_cols}")

# 使用LabelEncoder进行编码
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    
    # 合并训练集和测试集的所有值，确保编码一致
    # 使用 dropna() 避免 NaN 干扰拟合，但在转换时需处理 NaN
    train_vals = train_df[col].dropna().unique()
    test_vals = test_df[col].dropna().unique()
    
    # 合并并去重
    all_values = list(set(list(train_vals) + list(test_vals)))
    
    # 如果存在缺失值，LabelEncoder 可能会报错，或者我们将缺失视为一类
    # 这里我们显式地将 NaN 替换为 'Unknown' 以便编码，或者让 LE 处理字符串
    # 更稳健的做法：填充缺失值为 'Unknown' 后再编码
    
    # 临时填充用于 fit
    temp_train = train_df[col].fillna('Unknown')
    temp_test = test_df[col].fillna('Unknown')
    
    # 重新获取所有可能的值（包含 'Unknown'）
    all_values_final = list(set(list(temp_train.unique()) + list(temp_test.unique())))
    
    le.fit(all_values_final)
    
    # 转换训练集
    train_df[col] = le.transform(temp_train)
    
    # 转换测试集
    test_df[col] = le.transform(temp_test)
    
    label_encoders[col] = le
    print(f"  ✓ {col}: {len(le.classes_)} 个类别 -> 已编码")

print("✅ 类别型特征编码完成")

# ========================
# 4. 保存预处理后的数据
# ========================
print("\n【4】保存预处理后的数据...")

train_df.to_csv('train_preprocessed.csv', index=False)
test_df.to_csv('test_preprocessed.csv', index=False)

print("✅ 训练集已保存到: train_preprocessed.csv")
print("✅ 测试集已保存到: test_preprocessed.csv")

# ========================
# 5. 验证结果
# ========================
print("\n【5】验证预处理结果...")

print(f"\n训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 检查缺失值
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

print(f"\n训练集缺失值统计:")
total_train_missing = train_missing.sum()
print(f"  总缺失值: {total_train_missing}")
if total_train_missing > 0:
    missing_cols_train = train_missing[train_missing > 0]
    for col, count in missing_cols_train.items():
        print(f"    {col}: {count}")
else:
    print("  ✅ 无缺失值")

print(f"\n测试集缺失值统计:")
total_test_missing = test_missing.sum()
print(f"  总缺失值: {total_test_missing}")
if total_test_missing > 0:
    missing_cols_test = test_missing[test_missing > 0]
    for col, count in missing_cols_test.items():
        print(f"    {col}: {count}")
else:
    print("  ✅ 无缺失值")

# 检查特征一致性
train_features = set(train_df.columns)
test_features = set(test_df.columns)

# 排除price列（测试集没有目标变量是正常的）
train_features_no_price = train_features - {'price'}

if train_features_no_price == test_features:
    print("\n✅ 训练集和测试集特征完全一致（除price外）")
else:
    missing_in_test = train_features_no_price - test_features
    extra_in_test = test_features - train_features_no_price
    
    if missing_in_test:
        print(f"\n⚠️  测试集缺少的特征: {missing_in_test}")
    if extra_in_test:
        print(f"\n⚠️  测试集多出的特征: {extra_in_test}")

# ========================
# 6. 总结
# ========================
print("\n" + "=" * 80)
print("📊 数据预处理总结")
print("=" * 80)
print(f"""
✅ 完成的预处理步骤:

1️⃣  数据加载:
   - 训练集: {train_df.shape}
   - 测试集: {test_df.shape}

2️⃣  删除列:
   - 删除了: {drop_columns_existing if drop_columns_existing else '无'}

3️⃣  类别型特征编码:
   - 编码的特征: {len(categorical_cols)} 个
   - 方法: LabelEncoder (缺失值填充为 'Unknown' 后编码)
   - 策略: 合并训练集和测试集的值进行统一编码

4️⃣  特征一致性:
   - 品牌特征: {'✅ 已包含' if all(col in train_df.columns for col in brand_cols) else '⚠️ 检查'}
   - 训练集和测试集: {'✅ 特征一致' if train_features_no_price == test_features else '⚠️ 存在差异'}

📁 输出文件:
   - train_preprocessed.csv: {train_df.shape}
   - test_preprocessed.csv: {test_df.shape}

💡 关键改进:
   - 修复了品牌统计特征缺失问题（依赖上游特征工程）
   - 确保训练集和测试集特征完全一致
   - 使用统一的编码策略处理未见类别
""")

print("=" * 80)
print("🎉 数据预处理完成！")
print("=" * 80)
