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
print("🔍 第二步：基于粗填特征的聚类分析")
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
# 2. 定义待填充的核心业务特征
# ========================
print("\n【2】定义核心业务特征...")

# 这些是我们认为需要重点关注的业务特征（可能用于后续精细填补）
target_features = {
    'power': '功率',
    'kilometer': '行驶里程',
    'v_0': '匿名特征0',
    'v_1': '匿名特征1',
    'v_2': '匿名特征2',
    'v_3': '匿名特征3',
}

print("待分析的核心业务特征:")
for feat, desc in target_features.items():
    print(f"  - {feat}: {desc}")

# ========================
# 3. 特征筛选：保留与目标特征相关性>0.1的特征
# ========================
print("\n【3】特征筛选 - 保留相关性>0.1的核心特征...")

# 获取所有数值型特征（排除ID和目标变量）
exclude_cols = ['SaleID', 'price', 'name']
numerical_cols = [col for col in train_df.columns 
                  if col not in exclude_cols 
                  and train_df[col].dtype in ['float64', 'int64']]

print(f"候选数值特征总数: {len(numerical_cols)}")

# 计算每个目标特征与其他特征的相关性
selected_features = set()

for target_feat in target_features.keys():
    if target_feat not in numerical_cols:
        continue
    
    # 计算与所有其他特征的相关性
    correlations = train_df[numerical_cols].corr()[target_feat].abs()
    
    # 筛选相关性>0.1的特征（排除自身）
    correlated_feats = correlations[
        (correlations > 0.1) & (correlations.index != target_feat)
    ].index.tolist()
    
    selected_features.update(correlated_feats)
    selected_features.add(target_feat)  # 包含目标特征本身
    
    print(f"\n  {target_feat} 的相关特征 (>0.1): {len(correlated_feats)}个")
    if len(correlated_feats) <= 10:
        for feat in correlated_feats[:10]:
            corr_val = correlations[feat]
            print(f"    - {feat}: {corr_val:.3f}")

# 转换为列表
selected_features = list(selected_features)
print(f"\n✅ 筛选后保留的特征数: {len(selected_features)}")
print(f"   从 {len(numerical_cols)} 个特征中筛选出 {len(selected_features)} 个核心特征")

# ========================
# 4. 准备聚类数据
# ========================
print("\n【4】准备聚类数据...")

# 提取筛选后的特征（只保留训练集和测试集都存在的特征）
common_features = [feat for feat in selected_features if feat in test_df.columns]

print(f"筛选后特征数: {len(selected_features)}")
print(f"共同特征数: {len(common_features)}")
print(f"排除的特征: {set(selected_features) - set(common_features)}")

# 提取共同特征
X_train_cluster = train_df[common_features].values
X_test_cluster = test_df[common_features].values

print(f"训练集聚类数据形状: {X_train_cluster.shape}")
print(f"测试集聚类数据形状: {X_test_cluster.shape}")

# 检查是否有缺失值
print(f"训练集缺失值: {np.isnan(X_train_cluster).sum()}")
print(f"测试集缺失值: {np.isnan(X_test_cluster).sum()}")

# ========================
# 5. 确定最优簇数（手肘法 + 轮廓系数）
# ========================
print("\n【5】确定最优簇数...")
print("  → 使用手肘法和轮廓系数评估不同k值...")
print("  ⚠️  注意：由于数据量较大，使用抽样加速计算...")

# 为了加速计算，使用抽样数据（如果数据量太大）
sample_size = min(20000, len(X_train_cluster))  # 最多使用20000样本
if sample_size < len(X_train_cluster):
    print(f"  → 从 {len(X_train_cluster)} 样本中抽取 {sample_size} 个进行快速评估...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_train_cluster), sample_size, replace=False)
    X_sample = X_train_cluster[sample_indices]
else:
    X_sample = X_train_cluster
    print(f"  → 使用全部 {len(X_train_cluster)} 样本进行评估")

# 尝试的簇数范围（缩小范围以加速）
k_range = range(5, 13)  # 5-12（减少计算量）

inertias = []      # 惯性（SSE）
silhouette_scores = []  # 轮廓系数

for k in k_range:
    print(f"    计算 k={k}...", end=' ')
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    labels = kmeans.fit_predict(X_sample)
    
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_sample, labels)
    silhouette_scores.append(sil_score)
    
    print(f"SSE={kmeans.inertia_:,.0f}, 轮廓系数={sil_score:.4f}")

# 找到最优簇数（轮廓系数最大）
best_k_idx = np.argmax(silhouette_scores)
best_k = list(k_range)[best_k_idx]
best_silhouette = silhouette_scores[best_k_idx]

print(f"\n✅ 最优簇数: k={best_k} (轮廓系数={best_silhouette:.4f})")

# ========================
# 6. 可视化手肘法和轮廓系数
# ========================
print("\n【6】绘制手肘法和轮廓系数图...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 手肘法
axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'最优k={best_k}')
axes[0].set_xlabel('簇数 (k)', fontsize=12)
axes[0].set_ylabel('SSE (惯性)', fontsize=12)
axes[0].set_title('手肘法 - 确定最优簇数', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 轮廓系数
axes[1].plot(list(k_range), silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].axvline(x=best_k, color='blue', linestyle='--', linewidth=2, label=f'最优k={best_k}')
axes[1].set_xlabel('簇数 (k)', fontsize=12)
axes[1].set_ylabel('轮廓系数', fontsize=12)
axes[1].set_title('轮廓系数 - 确定最优簇数', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存: cluster_optimization.png")
plt.show()

# ========================
# 7. 使用最优簇数进行K-Means聚类
# ========================
print(f"\n【7】执行K-Means聚类 (k={best_k})...")

# 训练集聚类
kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
    max_iter=300
)

train_labels = kmeans_final.fit_predict(X_train_cluster)
test_labels = kmeans_final.predict(X_test_cluster)

print(f"训练集聚类完成: {len(train_labels)} 个样本")
print(f"测试集聚类完成: {len(test_labels)} 个样本")

# ========================
# 8. 聚类结果分析
# ========================
print("\n【8】聚类结果分析...")

# 添加簇标签到DataFrame
train_df['cluster_label'] = train_labels
test_df['cluster_label'] = test_labels

# 统计每个簇的样本数
print("\n各簇样本分布:")
cluster_counts = train_df['cluster_label'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    pct = count / len(train_df) * 100
    print(f"  簇 {cluster_id}: {count:6d} 样本 ({pct:5.2f}%)")

# 分析每个簇的特征（以price为例）
if 'price' in train_df.columns:
    print("\n各簇的价格统计:")
    cluster_stats = train_df.groupby('cluster_label')['price'].agg(['mean', 'median', 'std', 'count'])
    cluster_stats.columns = ['平均价格', '中位数价格', '标准差', '样本数']
    print(cluster_stats.round(2))

# ========================
# 9. 可视化聚类结果（使用前2个主成分）
# ========================
print("\n【9】可视化聚类结果...")

from sklearn.decomposition import PCA

# 使用PCA降维到2D以便可视化
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_cluster)
X_test_pca = pca.transform(X_test_cluster)

print(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.2%}")

# 绘制训练集聚类结果
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    X_train_pca[:, 0], 
    X_train_pca[:, 1],
    c=train_labels,
    cmap='viridis',
    alpha=0.6,
    s=10
)
plt.colorbar(scatter, label='簇标签')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)', fontsize=12)
plt.title(f'K-Means聚类结果 (k={best_k}) - 训练集', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_visualization_train.png', dpi=300, bbox_inches='tight')
print("✅ 训练集聚类可视化已保存: cluster_visualization_train.png")
plt.show()

# ========================
# 10. 保存聚类结果
# ========================
print("\n【10】保存聚类结果...")

# 恢复ID和目标变量
if train_saleid is not None:
    train_df['SaleID'] = train_saleid
if train_price is not None:
    train_df['price'] = train_price
if test_saleid is not None:
    test_df['SaleID'] = test_saleid

# 保存
train_df.to_csv('train_with_clusters.csv', index=False)
test_df.to_csv('test_with_clusters.csv', index=False)

print("✅ 训练集已保存到: train_with_clusters.csv")
print("✅ 测试集已保存到: test_with_clusters.csv")

# ========================
# 11. 总结报告
# ========================
print("\n" + "=" * 80)
print("📊 聚类分析总结")
print("=" * 80)

print(f"""
✅ 完成的聚类分析步骤:

1️⃣  特征筛选:
   - 原始特征数: {len(numerical_cols)}
   - 筛选标准: 与核心业务特征相关性 > 0.1
   - 保留特征数: {len(selected_features)}
   - 筛选掉噪声特征: {len(numerical_cols) - len(selected_features)} 个

2️⃣  最优簇数确定:
   - 尝试范围: k = 5 ~ 15
   - 评估方法: 手肘法 + 轮廓系数
   - 最优簇数: k = {best_k}
   - 最佳轮廓系数: {best_silhouette:.4f}

3️⃣  K-Means聚类:
   - 算法: K-Means (k={best_k})
   - 初始化次数: 10次
   - 最大迭代: 300次
   - 训练集样本: {len(train_labels)}
   - 测试集样本: {len(test_labels)}

4️⃣  聚类质量:
   - 各簇样本分布均衡性: {'良好' if cluster_counts.std() / cluster_counts.mean() < 0.5 else '需关注'}
   - 最大簇: {cluster_counts.idxmax()} ({cluster_counts.max()} 样本)
   - 最小簇: {cluster_counts.idxmin()} ({cluster_counts.min()} 样本)

📁 输出文件:
   - train_with_clusters.csv: 带簇标签的训练集
   - test_with_clusters.csv: 带簇标签的测试集
   - cluster_optimization.png: 最优簇数选择图
   - cluster_visualization_train.png: 聚类可视化图

💡 下一步建议:
   → 可以将簇标签作为新特征加入模型
   → 可以针对不同簇分别建模（分层建模）
   → 可以分析各簇的业务含义，指导特征工程
""")

print("=" * 80)
print("🎉 聚类分析完成！")
print("=" * 80)
