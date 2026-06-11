"""
深入分析v_14匿名特征的异常值情况
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🔍 v_14 匿名特征异常值深度分析")
print("=" * 80)

# 加载数据
train_k5 = pd.read_csv('train_refined_k5.csv')
test_k5 = pd.read_csv('test_refined_k5.csv')

print(f"\n📊 数据集信息:")
print(f"  训练集: {train_k5.shape}")
print(f"  测试集: {test_k5.shape}")

# ========================
# 1. 基本统计分析
# ========================
print(f"\n{'='*80}")
print("📈 v_14 基本统计分析")
print(f"{'='*80}")

for dataset_name, df in [('训练集', train_k5), ('测试集', test_k5)]:
    print(f"\n{dataset_name}:")
    col_data = df['v_14']
    
    print(f"  样本数: {len(col_data):,}")
    print(f"  均值: {col_data.mean():.6f}")
    print(f"  中位数: {col_data.median():.6f}")
    print(f"  标准差: {col_data.std():.6f}")
    print(f"  最小值: {col_data.min():.6f}")
    print(f"  最大值: {col_data.max():.6f}")
    print(f"  偏度: {col_data.skew():.4f}")
    print(f"  峰度: {col_data.kurtosis():.4f}")
    
    # 分位数
    print(f"\n  分位数:")
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"    {q*100:5.0f}%: {col_data.quantile(q):.6f}")

# ========================
# 2. IQR异常值详细分析
# ========================
print(f"\n{'='*80}")
print("🔍 IQR异常值详细分析")
print(f"{'='*80}")

Q1 = train_k5['v_14'].quantile(0.25)
Q3 = train_k5['v_14'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n训练集 IQR统计:")
print(f"  Q1 (25%): {Q1:.6f}")
print(f"  Q3 (75%): {Q3:.6f}")
print(f"  IQR: {IQR:.6f}")
print(f"  下界: {lower_bound:.6f}")
print(f"  上界: {upper_bound:.6f}")

outliers_train = train_k5[(train_k5['v_14'] < lower_bound) | (train_k5['v_14'] > upper_bound)]
normal_train = train_k5[(train_k5['v_14'] >= lower_bound) & (train_k5['v_14'] <= upper_bound)]

print(f"\n  正常值数量: {len(normal_train):,} ({len(normal_train)/len(train_k5)*100:.2f}%)")
print(f"  异常值数量: {len(outliers_train):,} ({len(outliers_train)/len(train_k5)*100:.2f}%)")

if len(outliers_train) > 0:
    print(f"\n  异常值统计:")
    print(f"    均值: {outliers_train['v_14'].mean():.6f}")
    print(f"    中位数: {outliers_train['v_14'].median():.6f}")
    print(f"    最小值: {outliers_train['v_14'].min():.6f}")
    print(f"    最大值: {outliers_train['v_14'].max():.6f}")
    
    # 上下界异常值分别统计
    lower_outliers = outliers_train[outliers_train['v_14'] < lower_bound]
    upper_outliers = outliers_train[outliers_train['v_14'] > upper_bound]
    
    print(f"\n  下界以下异常值: {len(lower_outliers):,} ({len(lower_outliers)/len(train_k5)*100:.2f}%)")
    print(f"  上界以上异常值: {len(upper_outliers):,} ({len(upper_outliers)/len(train_k5)*100:.2f}%)")

# ========================
# 3. 与目标变量price的相关性分析
# ========================
print(f"\n{'='*80}")
print("📊 与目标变量price的相关性分析")
print(f"{'='*80}")

corr_normal = normal_train['v_14'].corr(normal_train['price'])
corr_outlier = outliers_train['v_14'].corr(outliers_train['price']) if len(outliers_train) > 0 else 0
corr_all = train_k5['v_14'].corr(train_k5['price'])

print(f"\n  整体相关性: {corr_all:.4f}")
print(f"  正常值相关性: {corr_normal:.4f}")
print(f"  异常值相关性: {corr_outlier:.4f}")

# 异常值的平均价格
if len(outliers_train) > 0:
    avg_price_normal = normal_train['price'].mean()
    avg_price_outlier = outliers_train['price'].mean()
    
    print(f"\n  正常值平均价格: {avg_price_normal:.2f}")
    print(f"  异常值平均价格: {avg_price_outlier:.2f}")
    print(f"  差异: {avg_price_outlier - avg_price_normal:.2f} ({(avg_price_outlier/avg_price_normal-1)*100:.2f}%)")

# ========================
# 4. 在聚类簇中的分布
# ========================
if 'cluster_label' in train_k5.columns:
    print(f"\n{'='*80}")
    print("🔍 在各聚类簇中的分布")
    print(f"{'='*80}")
    
    cluster_dist = train_k5.groupby('cluster_label')['v_14'].agg(['count', 'mean', 'std']).round(4)
    cluster_dist['outlier_count'] = train_k5.groupby('cluster_label').apply(
        lambda x: len(x[(x['v_14'] < lower_bound) | (x['v_14'] > upper_bound)])
    )
    cluster_dist['outlier_pct'] = (cluster_dist['outlier_count'] / cluster_dist['count'] * 100).round(2)
    
    print(f"\n各簇的v_14分布:")
    print(cluster_dist.to_string())
    
    # 检查是否有簇的异常值比例特别高
    high_outlier_clusters = cluster_dist[cluster_dist['outlier_pct'] > 10]
    if len(high_outlier_clusters) > 0:
        print(f"\n⚠️  异常值比例>10%的簇:")
        print(high_outlier_clusters[['outlier_count', 'outlier_pct']])

# ========================
# 5. 其他匿名特征的对比
# ========================
print(f"\n{'='*80}")
print("📊 所有匿名特征的异常值比例对比")
print(f"{'='*80}")

v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in train_k5.columns]

print(f"\n{'特征':<10} {'异常值数量':<15} {'异常值比例':<15} {'偏度':<10} {'峰度':<10}")
print("-" * 80)

v_outlier_stats = []
for col in v_cols:
    Q1_c = train_k5[col].quantile(0.25)
    Q3_c = train_k5[col].quantile(0.75)
    IQR_c = Q3_c - Q1_c
    lower_c = Q1_c - 1.5 * IQR_c
    upper_c = Q3_c + 1.5 * IQR_c
    
    outlier_count = len(train_k5[(train_k5[col] < lower_c) | (train_k5[col] > upper_c)])
    outlier_pct = outlier_count / len(train_k5) * 100
    skewness = train_k5[col].skew()
    kurtosis = train_k5[col].kurtosis()
    
    v_outlier_stats.append({
        'feature': col,
        'outlier_count': outlier_count,
        'outlier_pct': outlier_pct,
        'skewness': skewness,
        'kurtosis': kurtosis
    })
    
    marker = " ⚠️" if outlier_pct > 5 else ""
    print(f"{col:<10} {outlier_count:<15,} {outlier_pct:<14.2f}% {skewness:<10.4f} {kurtosis:<10.4f}{marker}")

# 排序找出异常值最多的特征
v_outlier_df = pd.DataFrame(v_outlier_stats).sort_values('outlier_pct', ascending=False)
print(f"\n异常值比例Top5:")
print(v_outlier_df.head(5)[['feature', 'outlier_pct', 'skewness']].to_string(index=False))

# ========================
# 6. 处理建议
# ========================
print(f"\n{'='*80}")
print("💡 v_14 异常值处理建议")
print(f"{'='*80}")

print(f"\n基于分析结果，给出以下建议:\n")

# 判断严重程度
if len(outliers_train) / len(train_k5) * 100 < 5:
    severity = "轻微"
    recommendation = "✅ 建议保留，无需特殊处理"
elif len(outliers_train) / len(train_k5) * 100 < 10:
    severity = "中等"
    recommendation = "🟡 可选优化：添加标识特征或log变换"
else:
    severity = "严重"
    recommendation = "🔴 需要处理：考虑截断、删除或变换"

print(f"1️⃣ 严重程度评估: {severity} ({len(outliers_train)/len(train_k5)*100:.2f}%异常值)")
print(f"   {recommendation}\n")

print(f"2️⃣ 推荐方案（按优先级）:")
print(f"   A. 【优先】保留原样 - CatBoost/XGBoost对异常值鲁棒性强")
print(f"      • 树模型能自动学习异常值的模式")
print(f"      • v_14与price相关性: {corr_all:.4f}")
print(f"      • 异常值平均价格差异: {(avg_price_outlier/avg_price_normal-1)*100:.2f}%\n")

print(f"   B. 【可选】添加二元标识特征")
print(f"      • 创建 is_v14_outlier 特征（1=异常值，0=正常值）")
print(f"      • 帮助模型识别异常值样本")
print(f"      • 不改变原始数值，保留完整信息\n")

print(f"   C. 【谨慎】Log变换（如果偏度>2）")
print(f"      • 当前偏度: {train_k5['v_14'].skew():.4f}")
if abs(train_k5['v_14'].skew()) > 2:
    print(f"      • ⚠️  偏度较大，可以考虑log变换")
    print(f"      • 注意：需要处理负值和零值")
else:
    print(f"      • ✅ 偏度适中，不建议log变换\n")

print(f"   D. 【避免】直接截断或删除")
print(f"      • ❌ 匿名特征缺乏业务含义，不宜盲目截断")
print(f"      • ❌ 删除会损失5.15%的样本，影响模型性能")
print(f"      • ❌ 可能丢失有价值的预测信号\n")

print(f"3️⃣ 验证方法:")
print(f"   • 对比有无处理时的模型RMSE变化")
print(f"   • 使用交叉验证确保稳定性")
print(f"   • 监控v_14的特征重要性排名")

print(f"\n{'='*80}")
print("✅ 分析完成！")
print(f"{'='*80}")
