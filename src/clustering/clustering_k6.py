import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔍 基于k=6的聚类分析")
print("=" * 80)

# ========================
# 1. 加载预处理后的数据
# ========================
print("\n【1】加载预处理后的数据...")
train_df = pd.read_csv('train_preprocessed.csv')
test_df = pd.read_csv('test_preprocessed.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# ========================
# 2. 特征筛选
# ========================
print("\n【2】特征筛选 - 保留与核心业务特征相关性>0.1的特征...")

# 选择核心业务特征用于相关性计算
core_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3']

# 获取所有数值型特征
numerical_cols = [col for col in train_df.columns 
                  if train_df[col].dtype in ['float64', 'int64']
                  and col not in ['SaleID', 'price', 'cluster_label']]

# 计算与核心特征的相关性
selected_features = set()
for core_feat in core_features:
    if core_feat in numerical_cols:
        correlations = train_df[numerical_cols].corr()[core_feat].abs()
        # 选择相关性 > 0.1 的特征
        relevant = correlations[correlations > 0.1].index.tolist()
        selected_features.update(relevant)

# 转换为列表并排序
selected_features = sorted(list(selected_features))
print(f"原始特征数: {len(numerical_cols)}")
print(f"筛选后特征数: {len(selected_features)}")

# 确保训练集和测试集有共同的特征
common_features = [col for col in selected_features if col in test_df.columns]
print(f"共同特征数: {len(common_features)}")

# ========================
# 3. 准备聚类数据
# ========================
print("\n【3】准备聚类数据...")

scaler = StandardScaler()
X_train_full = scaler.fit_transform(train_df[common_features])
X_test_full = scaler.transform(test_df[common_features])

print(f"完整训练集: {X_train_full.shape}")
print(f"完整测试集: {X_test_full.shape}")

# ========================
# 4. 使用k=6进行聚类
# ========================
print("\n【4】执行K-Means聚类 (k=6)...")

k = 6
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)

# 在完整训练集上拟合
train_labels = kmeans.fit_predict(X_train_full)
# 预测测试集
test_labels = kmeans.predict(X_test_full)

print(f"训练集聚类完成: {len(train_labels)} 个样本")
print(f"测试集聚类完成: {len(test_labels)} 个样本")

# SSE和轮廓系数
inertia = kmeans.inertia_
from sklearn.metrics import silhouette_score
sil_score = silhouette_score(X_train_full, train_labels)
print(f"SSE: {inertia:,.0f}")
print(f"轮廓系数: {sil_score:.4f}")

# ========================
# 5. 添加簇标签
# ========================
print("\n【5】添加簇标签到数据集...")

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
    print(cluster_stats)

# ========================
# 6. 保存带簇标签的数据
# ========================
print("\n【6】保存聚类结果...")

train_df.to_csv('train_with_clusters_k6.csv', index=False)
test_df.to_csv('test_with_clusters_k6.csv', index=False)

print("✅ 训练集已保存到: train_with_clusters_k6.csv")
print("✅ 测试集已保存到: test_with_clusters_k6.csv")

# ========================
# 7. 总结
# ========================
print("\n" + "=" * 80)
print("📊 聚类分析总结 (k=6)")
print("=" * 80)

print(f"""
✅ 完成的步骤:

1️⃣  特征筛选:
   - 原始特征: {len(numerical_cols)} 个
   - 筛选标准: 相关性 > 0.1
   - 保留特征: {len(common_features)} 个

2️⃣  K-Means聚类:
   - 簇数: k = {k}
   - SSE: {inertia:,.0f}
   - 轮廓系数: {sil_score:.4f}

3️⃣  聚类规模:
   - 训练集: {len(train_labels)} 样本
   - 测试集: {len(test_labels)} 样本
   - 簇数: {k}

📁 输出文件:
   - train_with_clusters_k6.csv
   - test_with_clusters_k6.csv

💡 下一步:
   → 将cluster_label作为新特征加入模型
   → 可以进行簇内精填修正

================================================================================
🎉 聚类分析完成（k=6）！
================================================================================
""")