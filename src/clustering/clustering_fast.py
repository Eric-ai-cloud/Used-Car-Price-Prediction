import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🔍 第二步：基于粗填特征的聚类分析（快速版）")
print("=" * 80)

# ========================
# 1. 数据加载
# ========================
print("\n【1】加载预处理后的数据...")
train_df = pd.read_csv('train_preprocessed.csv')
test_df = pd.read_csv('test_preprocessed.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 保存原始ID和目标变量
train_saleid = train_df['SaleID'].copy() if 'SaleID' in train_df.columns else None
test_saleid = test_df['SaleID'].copy() if 'SaleID' in test_df.columns else None
train_price = train_df['price'].copy() if 'price' in train_df.columns else None

# ========================
# 2. 定义核心业务特征并筛选
# ========================
print("\n【2】特征筛选 - 保留与核心业务特征相关性>0.1的特征...")

target_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3']

# 获取所有数值型特征
exclude_cols = ['SaleID', 'price', 'name']
numerical_cols = [col for col in train_df.columns 
                  if col not in exclude_cols 
                  and train_df[col].dtype in ['float64', 'int64']]

# 计算相关性并筛选
selected_features = set()
for target_feat in target_features:
    if target_feat not in numerical_cols:
        continue
    correlations = train_df[numerical_cols].corr()[target_feat].abs()
    correlated_feats = correlations[
        (correlations > 0.1) & (correlations.index != target_feat)
    ].index.tolist()
    selected_features.update(correlated_feats)
    selected_features.add(target_feat)

# 只保留训练集和测试集都有的特征
common_features = [feat for feat in selected_features if feat in test_df.columns]

print(f"原始特征数: {len(numerical_cols)}")
print(f"筛选后特征数: {len(selected_features)}")
print(f"共同特征数: {len(common_features)}")

# ========================
# 3. 准备聚类数据（抽样加速）
# ========================
print("\n【3】准备聚类数据（使用抽样加速）...")

# 抽取样本加速计算
sample_size = 15000
np.random.seed(42)
sample_indices = np.random.choice(len(train_df), sample_size, replace=False)

X_train_sample = train_df.loc[sample_indices, common_features].values
X_train_full = train_df[common_features].values
X_test_full = test_df[common_features].values

print(f"抽样用于确定k值: {X_train_sample.shape}")
print(f"完整训练集: {X_train_full.shape}")
print(f"完整测试集: {X_test_full.shape}")

# ========================
# 4. 确定最优簇数
# ========================
print("\n【4】确定最优簇数（手肘法 + 轮廓系数）...")

k_range = range(5, 13)
inertias = []
silhouette_scores = []

for k in k_range:
    print(f"  计算 k={k}...", end=' ')
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_train_sample)
    
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_train_sample, labels)
    silhouette_scores.append(sil_score)
    
    print(f"SSE={kmeans.inertia_:,.0f}, 轮廓系数={sil_score:.4f}")

best_k_idx = np.argmax(silhouette_scores)
best_k = list(k_range)[best_k_idx]
best_silhouette = silhouette_scores[best_k_idx]

print(f"\n✅ 最优簇数: k={best_k} (轮廓系数={best_silhouette:.4f})")

# ========================
# 5. 可视化
# ========================
print("\n【5】绘制优化曲线...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'最优k={best_k}')
axes[0].set_xlabel('簇数 (k)', fontsize=12)
axes[0].set_ylabel('SSE', fontsize=12)
axes[0].set_title('手肘法', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].axvline(x=best_k, color='blue', linestyle='--', linewidth=2, label=f'最优k={best_k}')
axes[1].set_xlabel('簇数 (k)', fontsize=12)
axes[1].set_ylabel('轮廓系数', fontsize=12)
axes[1].set_title('轮廓系数', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存: cluster_optimization.png")
plt.show()

# ========================
# 6. 使用最优k值进行完整聚类
# ========================
print(f"\n【6】执行K-Means聚类 (k={best_k})...")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)

# 在完整训练集上拟合
train_labels = kmeans_final.fit_predict(X_train_full)
# 预测测试集
test_labels = kmeans_final.predict(X_test_full)

print(f"训练集聚类完成: {len(train_labels)} 个样本")
print(f"测试集聚类完成: {len(test_labels)} 个样本")

# ========================
# 7. 添加簇标签
# ========================
print("\n【7】添加簇标签到数据集...")

train_df['cluster_label'] = train_labels
test_df['cluster_label'] = test_labels

# 统计簇分布
print("\n各簇样本分布:")
cluster_counts = train_df['cluster_label'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    pct = count / len(train_df) * 100
    print(f"  簇 {cluster_id}: {count:6d} 样本 ({pct:5.2f}%)")

# 价格统计
if 'price' in train_df.columns:
    print("\n各簇的价格统计:")
    cluster_stats = train_df.groupby('cluster_label')['price'].agg(['mean', 'median', 'std', 'count'])
    cluster_stats.columns = ['平均价格', '中位数', '标准差', '样本数']
    print(cluster_stats.round(2))

# ========================
# 8. 保存结果
# ========================
print("\n【8】保存聚类结果...")

if train_saleid is not None:
    train_df['SaleID'] = train_saleid
if train_price is not None:
    train_df['price'] = train_price
if test_saleid is not None:
    test_df['SaleID'] = test_saleid

train_df.to_csv('train_with_clusters.csv', index=False)
test_df.to_csv('test_with_clusters.csv', index=False)

print("✅ 训练集已保存到: train_with_clusters.csv")
print("✅ 测试集已保存到: test_with_clusters.csv")

# ========================
# 9. 总结
# ========================
print("\n" + "=" * 80)
print("📊 聚类分析总结")
print("=" * 80)

print(f"""
✅ 完成的步骤:

1️⃣  特征筛选:
   - 原始特征: {len(numerical_cols)} 个
   - 筛选标准: 相关性 > 0.1
   - 保留特征: {len(common_features)} 个

2️⃣  最优簇数:
   - 尝试范围: k = 5 ~ 12
   - 最优簇数: k = {best_k}
   - 轮廓系数: {best_silhouette:.4f}

3️⃣  K-Means聚类:
   - 训练集: {len(train_labels)} 样本
   - 测试集: {len(test_labels)} 样本
   - 簇数: {best_k}

📁 输出文件:
   - train_with_clusters.csv
   - test_with_clusters.csv
   - cluster_optimization.png

💡 下一步:
   → 将cluster_label作为新特征加入模型
   → 可以针对不同簇分别建模
""")

print("=" * 80)
print("🎉 聚类分析完成！")
print("=" * 80)
