"""
特征优化分析报告
全面分析当前数据，找出可以优化的特征
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔍 特征优化分析报告")
print("=" * 80)

# ========================
# 1. 数据加载
# ========================
print("\n【1】数据加载...")
train_df = pd.read_csv('train_final_k5.csv')
print(f"训练集形状: {train_df.shape}")
print(f"特征数量: {train_df.shape[1] - 2}")  # 减去SaleID和price

# ========================
# 2. 特征分类
# ========================
print("\n【2】特征分类...")

# 基础业务特征
basic_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 
                  'power', 'kilometer', 'notRepairedDamage', 'regionCode']

# 时间特征
time_features = ['regYear', 'regMonth', 'regDay', 'creatYear', 'creatMonth', 
                 'creatDay', 'carAge', 'carAgeMonth']

# 布尔/标志特征
binary_features = ['isYearStart', 'isYearEnd', 'regQuarter', 'creatQuarter']

# 衍生特征
derived_features = ['power_per_km', 'km_per_year', 'power_age_ratio']

# 品牌统计特征
brand_stats = ['brand_power_mean', 'brand_power_std', 'brand_km_mean', 'brand_km_std']

# 匿名特征
anonymous_features = [f'v_{i}' for i in range(15)]

# 聚类标签
cluster_feature = ['cluster_label']

print(f"\n特征分类统计:")
print(f"  • 基础业务特征: {len(basic_features)}")
print(f"  • 时间特征: {len(time_features)}")
print(f"  • 布尔/标志特征: {len(binary_features)}")
print(f"  • 衍生特征: {len(derived_features)}")
print(f"  • 品牌统计特征: {len(brand_stats)}")
print(f"  • 匿名特征: {len(anonymous_features)}")
print(f"  • 聚类标签: {len(cluster_feature)}")

# ========================
# 3. 缺失值检查
# ========================
print("\n【3】缺失值检查...")
missing = train_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"发现{len(missing)}个特征有缺失值:")
    for col, count in missing.items():
        pct = count / len(train_df) * 100
        print(f"  {col}: {count} ({pct:.2f}%)")
else:
    print("✅ 无缺失值")

# ========================
# 4. 低方差特征检测
# ========================
print("\n【4】低方差特征检测...")

numeric_cols = train_df.select_dtypes(include=[np.number]).columns
low_var_features = []

for col in numeric_cols:
    if col not in ['SaleID', 'price']:
        variance = train_df[col].var()
        if variance < 0.01:  # 方差阈值
            low_var_features.append((col, variance))

if low_var_features:
    print(f"⚠️ 发现{len(low_var_features)}个低方差特征（可能信息量少）:")
    for col, var in sorted(low_var_features, key=lambda x: x[1])[:10]:
        unique_count = train_df[col].nunique()
        print(f"  {col}: 方差={var:.6f}, 唯一值数={unique_count}")
else:
    print("✅ 无低方差特征")

# ========================
# 5. 高相关性特征检测
# ========================
print("\n【5】高相关性特征检测...")

# 计算数值特征的相关性矩阵
numeric_for_corr = train_df[numeric_cols].drop(columns=['SaleID', 'price'])
corr_matrix = numeric_for_corr.corr()

# 找出高度相关的特征对（相关系数 > 0.95）
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = abs(corr_matrix.iloc[i, j])
        if corr_value > 0.95:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_value
            ))

if high_corr_pairs:
    print(f"⚠️ 发现{len(high_corr_pairs)}对高度相关特征（可能存在冗余）:")
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
        print(f"  {feat1} ↔ {feat2}: r={corr:.4f}")
    
    print(f"\n💡 建议:")
    print(f"  • carAge和carAgeMonth高度相关，保留一个即可")
    print(f"  • regYear和carAge高度相关，考虑移除regYear")
else:
    print("✅ 无高度相关特征对")

# ========================
# 6. 与目标变量的相关性分析
# ========================
print("\n【6】与目标变量(price)的相关性分析...")

correlations = []
for col in numeric_for_corr.columns:
    corr = train_df[col].corr(train_df['price'])
    correlations.append((col, abs(corr), corr))

# 按绝对值排序
correlations.sort(key=lambda x: x[1], reverse=True)

print(f"\nTop 20 与price相关性最强的特征:")
print(f"{'排名':<6} {'特征':<25} {'相关系数':<12} {'方向'}")
print("-" * 60)

for i, (col, abs_corr, corr) in enumerate(correlations[:20]):
    direction = "正相关" if corr > 0 else "负相关"
    bar = "█" * int(abs_corr * 50)
    print(f"{i+1:<6} {col:<25} {corr:<12.4f} {direction} {bar}")

# 低相关特征
low_corr_features = [(col, abs_corr) for col, abs_corr, _ in correlations if abs_corr < 0.05]
if low_corr_features:
    print(f"\n⚠️ 发现{len(low_corr_features)}个与price低相关特征（|r|<0.05）:")
    for col, abs_corr in sorted(low_corr_features, key=lambda x: x[1])[:10]:
        print(f"  {col}: |r|={abs_corr:.4f}")

# ========================
# 7. 特征分布分析
# ========================
print("\n【7】特征分布异常检测...")

skewed_features = []
for col in numeric_for_corr.columns:
    skewness = train_df[col].skew()
    if abs(skewness) > 2:  # 偏度阈值
        skewed_features.append((col, skewness))

if skewed_features:
    print(f"⚠️ 发现{len(skewed_features)}个严重偏态分布特征:")
    for col, skew in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
        distribution = "右偏" if skew > 0 else "左偏"
        print(f"  {col}: 偏度={skew:.2f} ({distribution})")
else:
    print("✅ 无严重偏态分布特征")

# ========================
# 8. 类别特征基数分析
# ========================
print("\n【8】类别特征基数分析...")

categorical_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 
                    'notRepairedDamage', 'regionCode']

print(f"\n{'特征':<25} {'唯一值数':<12} {'基数类型'}")
print("-" * 60)

for col in categorical_cols:
    nunique = train_df[col].nunique()
    if nunique > 50:
        cardinality = "高基数 ⚠️"
    elif nunique > 10:
        cardinality = "中基数"
    else:
        cardinality = "低基数 ✅"
    print(f"{col:<25} {nunique:<12} {cardinality}")

# ========================
# 9. 衍生特征质量评估
# ========================
print("\n【9】衍生特征质量评估...")

print(f"\n衍生特征与price的相关性:")
for col in derived_features:
    corr = train_df[col].corr(train_df['price'])
    print(f"  {col:<25} r={corr:.4f}")

# 检查衍生特征的合理性
print(f"\n衍生特征统计:")
for col in derived_features:
    mean_val = train_df[col].mean()
    std_val = train_df[col].std()
    min_val = train_df[col].min()
    max_val = train_df[col].max()
    has_negative = (train_df[col] < 0).sum()
    
    print(f"\n  {col}:")
    print(f"    均值={mean_val:.2f}, 标准差={std_val:.2f}")
    print(f"    范围=[{min_val:.2f}, {max_val:.2f}]")
    if has_negative > 0:
        print(f"    ⚠️ 存在{has_negative}个负值")
    else:
        print(f"    ✅ 无非负值")

# ========================
# 10. 时间特征分析
# ========================
print("\n【10】时间特征分析...")

print(f"\n时间特征与price的相关性:")
for col in time_features:
    corr = train_df[col].corr(train_df['price'])
    print(f"  {col:<25} r={corr:.4f}")

# 检查时间特征的冗余
print(f"\n⚠️ 时间特征冗余分析:")
print(f"  • carAge vs carAgeMonth: r={train_df['carAge'].corr(train_df['carAgeMonth']):.4f}")
print(f"  • regYear vs carAge: r={train_df['regYear'].corr(train_df['carAge']):.4f}")
print(f"  • creatYear vs carAge: r={train_df['creatYear'].corr(train_df['carAge']):.4f}")

# ========================
# 11. 品牌统计特征分析
# ========================
print("\n【11】品牌统计特征分析...")

print(f"\n品牌统计特征与price的相关性:")
for col in brand_stats:
    corr = train_df[col].corr(train_df['price'])
    print(f"  {col:<25} r={corr:.4f}")

# 检查品牌统计特征的必要性
print(f"\n💡 品牌统计特征分析:")
print(f"  • 这些特征提供了品牌层面的聚合信息")
print(f"  • 可能与原始特征（power, kilometer）存在共线性")
print(f"  • 建议：通过模型特征重要性判断是否保留")

# ========================
# 12. 聚类标签分析
# ========================
print("\n【12】聚类标签分析...")

cluster_corr = train_df['cluster_label'].corr(train_df['price'])
print(f"  cluster_label与price的相关性: r={cluster_corr:.4f}")

print(f"\n各簇的price统计:")
cluster_stats = train_df.groupby('cluster_label')['price'].agg(['mean', 'std', 'count'])
for cluster_id in cluster_stats.index:
    mean_price = cluster_stats.loc[cluster_id, 'mean']
    std_price = cluster_stats.loc[cluster_id, 'std']
    count = int(cluster_stats.loc[cluster_id, 'count'])
    pct = count / len(train_df) * 100
    print(f"  簇{int(cluster_id)}: 均值={mean_price:.2f}, 标准差={std_price:.2f}, "
          f"样本数={count} ({pct:.2f}%)")

# ========================
# 13. 综合优化建议
# ========================
print("\n\n" + "=" * 80)
print("📋 综合优化建议")
print("=" * 80)

print(f"\n🔴 高优先级优化（立即可做）:")
print(f"  1. 移除冗余时间特征")
print(f"     • 保留carAge，移除carAgeMonth（高度相关）")
print(f"     • 或者保留carAgeMonth，移除carAge")
print(f"  2. 处理低相关特征")
print(f"     • 识别并移除与price相关性<0.05的特征")
print(f"     • 但需通过模型验证其实际价值")

print(f"\n🟡 中优先级优化（需要实验）:")
print(f"  3. 特征变换")
print(f"     • 对严重偏态分布的特征进行log变换或Box-Cox变换")
print(f"     • 例如：power, kilometer等")
print(f"  4. 交互特征")
print(f"     • 基于业务理解创建新的交互特征")
print(f"     • 例如：power × carAge, brand × fuelType等")
print(f"  5. 多项式特征")
print(f"     • 为重要特征添加平方项、立方项")
print(f"     • 例如：power², carAge²等")

print(f"\n🟢 低优先级优化（长期探索）:")
print(f"  6. 特征选择")
print(f"     • 使用Recursive Feature Elimination (RFE)")
print(f"     • 或使用基于树模型的特征重要性")
print(f"  7. 降维技术")
print(f"     • PCA用于匿名特征（v_0到v_14）")
print(f"     • 减少维度同时保留主要信息")
print(f"  8. 目标编码")
print(f"     • 对高基数类别特征（如model, regionCode）")
print(f"     • 使用target encoding替代one-hot编码")

print(f"\n💡 具体建议:")
print(f"  • 当前特征总数: {train_df.shape[1] - 2}")
print(f"  • 建议尝试移除: carAgeMonth（与carAge高度相关）")
print(f"  • 建议尝试添加: power², carAge², power×carAge等交互项")
print(f"  • 建议尝试验证: 品牌统计特征的必要性")

print("\n" + "=" * 80)
print("✅ 特征优化分析完成！")
print("=" * 80)
