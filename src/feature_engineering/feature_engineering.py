import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔧 特征工程 - 二手车价格预测")
print("=" * 80)

# ========================
# 1. 数据加载
# ========================
print("\n【1】数据加载...")
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# ========================
# 2. 基础特征工程函数（不含品牌统计）
# ========================
def basic_feature_engineering(df):
    """
    基础特征工程函数（不包含品牌统计特征）
    
    参数:
        df: 输入DataFrame
    
    返回:
        处理后的DataFrame
    """
    # -----------------------
    # 2.1 车龄特征构造
    # -----------------------
    print("  → 构造车龄特征...")
    
    # 确保日期字段为字符串类型
    df['regDate'] = df['regDate'].astype(str)
    df['creatDate'] = df['creatDate'].astype(str)
    
    # 提取注册日期信息
    df['regYear'] = df['regDate'].str[:4].astype(int)
    df['regMonth'] = df['regDate'].str[4:6].astype(int)
    df['regDay'] = df['regDate'].str[6:8].astype(int)
    
    # 提取交易日期信息
    df['creatYear'] = df['creatDate'].str[:4].astype(int)
    df['creatMonth'] = df['creatDate'].str[4:6].astype(int)
    df['creatDay'] = df['creatDate'].str[6:8].astype(int)
    
    # 计算车龄（年）- creatDate - regDate
    df['carAge'] = df['creatYear'] - df['regYear']
    
    # 计算车龄（月）- 更精确的车龄计算
    df['carAgeMonth'] = (df['creatYear'] - df['regYear']) * 12 + (df['creatMonth'] - df['regMonth'])
    
    # 删除原始日期列
    df.drop(['regDate', 'creatDate'], axis=1, inplace=True)
    
    print(f"    ✓ 新增特征: regYear, regMonth, regDay, creatYear, creatMonth, creatDay")
    print(f"    ✓ 新增特征: carAge (车龄-年), carAgeMonth (车龄-月)")
    
    # -----------------------
    # 2.2 其他时间相关特征
    # -----------------------
    print("  → 构造其他时间特征...")
    
    # 车辆是否在年初/年末交易
    df['isYearStart'] = (df['creatMonth'] <= 3).astype(int)
    df['isYearEnd'] = (df['creatMonth'] >= 10).astype(int)
    
    # 车辆注册季度
    df['regQuarter'] = ((df['regMonth'] - 1) // 3 + 1).astype(int)
    
    # 车辆交易季度
    df['creatQuarter'] = ((df['creatMonth'] - 1) // 3 + 1).astype(int)
    
    print(f"    ✓ 新增特征: isYearStart, isYearEnd, regQuarter, creatQuarter")
    
    # -----------------------
    # 2.3 交互特征
    # -----------------------
    print("  → 构造交互特征...")
    
    # 功率与里程的比值
    df['power_per_km'] = df['power'] / (df['kilometer'] + 1)  # 加1避免除零
    
    # 车龄与里程的关系
    df['km_per_year'] = df['kilometer'] / (df['carAge'] + 1)  # 年均行驶里程
    
    # 功率与车龄的交互
    df['power_age_ratio'] = df['power'] / (df['carAge'] + 1)
    
    print(f"    ✓ 新增特征: power_per_km, km_per_year, power_age_ratio")
    
    return df

# ========================
# 3. 应用基础特征工程
# ========================
print("\n【2】应用基础特征工程...")

print("\n训练集基础特征工程:")
train_df = basic_feature_engineering(train_df)

print("\n测试集基础特征工程:")
test_df = basic_feature_engineering(test_df)

print(f"\n✅ 基础特征工程完成！")
print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# ========================
# 4. 构造品牌统计特征（关键修复）
# ========================
print("\n【3】构造品牌统计特征...")

# 使用训练集计算品牌统计信息
print("  → 基于训练集计算品牌统计信息...")
brand_stats = train_df.groupby('brand').agg({
    'power': ['mean', 'std'],
    'kilometer': ['mean', 'std']
}).reset_index()

brand_stats.columns = ['brand', 'brand_power_mean', 'brand_power_std', 
                       'brand_km_mean', 'brand_km_std']

print(f"    ✓ 计算了 {len(brand_stats)} 个品牌的统计信息")

# 将品牌统计信息应用到训练集
print("  → 应用到训练集...")
train_df = train_df.merge(brand_stats, on='brand', how='left')

# 填充训练集中可能的缺失值（如果某个品牌在训练集中只有一个样本，std会为NaN）
train_df['brand_power_mean'] = train_df['brand_power_mean'].fillna(train_df['power'].mean())
train_df['brand_power_std'] = train_df['brand_power_std'].fillna(train_df['power'].std())
train_df['brand_km_mean'] = train_df['brand_km_mean'].fillna(train_df['kilometer'].mean())
train_df['brand_km_std'] = train_df['brand_km_std'].fillna(train_df['kilometer'].std())

print(f"    ✓ 训练集已添加品牌统计特征")

# 将品牌统计信息应用到测试集（关键修复点）
print("  → 应用到测试集...")
test_df = test_df.merge(brand_stats, on='brand', how='left')

# 对于测试集中新出现的品牌或没有匹配的品牌，用全局统计信息填充
global_power_mean = train_df['power'].mean()
global_power_std = train_df['power'].std()
global_km_mean = train_df['kilometer'].mean()
global_km_std = train_df['kilometer'].std()

test_df['brand_power_mean'] = test_df['brand_power_mean'].fillna(global_power_mean)
test_df['brand_power_std'] = test_df['brand_power_std'].fillna(global_power_std)
test_df['brand_km_mean'] = test_df['brand_km_mean'].fillna(global_km_mean)
test_df['brand_km_std'] = test_df['brand_km_std'].fillna(global_km_std)

print(f"    ✓ 测试集已添加品牌统计特征")
print(f"    ✓ 对于未知品牌，使用全局统计信息作为兜底")

print(f"\n✅ 品牌统计特征构造完成！")

# ========================
# 5. 保存特征工程后的数据
# ========================
print("\n【4】保存特征工程后的数据...")

train_df.to_csv('train_featured.csv', index=False)
test_df.to_csv('test_featured.csv', index=False)

print("✅ 训练集已保存到: train_featured.csv")
print("✅ 测试集已保存到: test_featured.csv")

# ========================
# 6. 验证特征一致性
# ========================
print("\n【5】验证训练集和测试集特征一致性...")

train_features = set(train_df.columns)
test_features = set(test_df.columns)

missing_in_test = train_features - test_features
missing_in_train = test_features - train_features

if missing_in_test:
    print(f"⚠️  测试集缺少的特征: {missing_in_test}")
else:
    print("✅ 测试集包含所有训练集的特征")

if missing_in_train:
    print(f"⚠️  训练集缺少的特征: {missing_in_train}")
else:
    print("✅ 训练集包含所有测试集的特征")

if not missing_in_test and not missing_in_train:
    print("\n🎉 训练集和测试集特征完全一致！")

print(f"\n最终训练集形状: {train_df.shape}")
print(f"最终测试集形状: {test_df.shape}")

# ========================
# 7. 总结
# ========================
print("\n" + "=" * 80)
print("📊 特征工程总结")
print("=" * 80)
print(f"""
✅ 完成的特征工程步骤:

1️⃣  车龄特征:
   - regYear, regMonth, regDay
   - creatYear, creatMonth, creatDay
   - carAge (车龄-年)
   - carAgeMonth (车龄-月)

2️⃣  时间相关特征:
   - isYearStart, isYearEnd
   - regQuarter, creatQuarter

3️⃣  交互特征:
   - power_per_km (功率/里程比)
   - km_per_year (年均行驶里程)
   - power_age_ratio (功率/车龄比)

4️⃣  品牌统计特征:
   - brand_power_mean, brand_power_std
   - brand_km_mean, brand_km_std
   - ✅ 训练集和测试集都已添加

📁 输出文件:
   - train_featured.csv: {train_df.shape}
   - test_featured.csv: {test_df.shape}

💡 关键改进:
   - 使用训练集的统计信息应用到测试集
   - 对于测试集中的未知品牌，使用全局统计信息兜底
   - 确保训练集和测试集特征完全一致
""")

print("=" * 80)
print("🎉 特征工程完成！")
print("=" * 80)
