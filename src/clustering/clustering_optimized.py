import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🔍 基于粗填特征的聚类分析 - 完整流程")
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

# 检查缺失值
print(f"\n粗填前缺失值:")
print(f"  训练集: {train_df.isnull().sum().sum()}")
print(f"  测试集: {test_df.isnull().sum().sum()}")

# ========================
# 1.5 KNN粗填（关键步骤）
# ========================
print("\n【1.5】执行KNN粗填缺失值...")

# 识别需要填充的数值型特征
numerical_cols_for_impute = [col for col in train_df.columns 
                              if train_df[col].dtype in ['float64', 'int64']
                              and col not in ['SaleID', 'price']]

print(f"需要填充的数值型特征数: {len(numerical_cols_for_impute)}")

# 使用KNNImputer进行粗填
print("执行KNN粗填 (k=7, 距离加权)...")
imputer = KNNImputer(n_neighbors=7, weights='distance')

# 对训练集进行粗填
train_numerical = train_df[numerical_cols_for_impute]
train_filled = imputer.fit_transform(train_numerical)
train_df[numerical_cols_for_impute] = train_filled

# 对测试集进行粗填（使用训练集的imputer）
test_numerical = test_df[numerical_cols_for_impute]
test_filled = imputer.transform(test_numerical)
test_df[numerical_cols_for_impute] = test_filled

print(f"粗填后缺失值:")
print(f"  训练集: {train_df.isnull().sum().sum()}")
print(f"  测试集: {test_df.isnull().sum().sum()}")

if train_df.isnull().sum().sum() == 0 and test_df.isnull().sum().sum() == 0:
    print("✅ KNN粗填完成，无缺失值")
else:
    print("⚠️  警告：仍存在缺失值")

# ========================
# 2. 定义核心业务特征（待填充的关键特征）
# ========================
print("\n【2】定义核心业务特征...")

# 这些是需要重点关注的业务特征（可能用于后续精细填补或建模）
core_features = [
    'power',        # 功率
    'kilometer',    # 行驶里程
    'v_0',          # 匿名特征0
    'v_1',          # 匿名特征1
    'v_2',          # 匿名特征2
    'v_3',          # 匿名特征3
]

print("核心业务特征列表:")
feature_descriptions = {
    'power': '发动机功率（马力）',
    'kilometer': '行驶里程（万公里）',
    'v_0': '匿名特征0（车辆技术参数）',
    'v_1': '匿名特征1（车辆技术参数）',
    'v_2': '匿名特征2（车辆技术参数）',
    'v_3': '匿名特征3（车辆技术参数）',
}

for feat in core_features:
    desc = feature_descriptions.get(feat, '未知特征')
    exists_in_train = feat in train_df.columns
    exists_in_test = feat in test_df.columns
    status = "✅" if (exists_in_train and exists_in_test) else "❌"
    print(f"  {status} {feat:15s}: {desc}")

# ========================
# 3. 特征筛选：保留与核心业务特征相关性>0.1的特征
# ========================
print("\n【3】特征筛选 - 只保留与核心业务特征相关性>0.1的特征...")

# 获取所有数值型特征（排除ID、目标变量和name）
exclude_cols = ['SaleID', 'price', 'name', 'cluster_label']
numerical_cols = [col for col in train_df.columns 
                  if col not in exclude_cols 
                  and train_df[col].dtype in ['float64', 'int64']]

print(f"\n候选数值特征总数: {len(numerical_cols)}")
print(f"候选特征列表（前20个）: {numerical_cols[:20]}")

# 计算每个核心特征与其他特征的相关性
selected_features = set()
correlation_details = {}

for core_feat in core_features:
    if core_feat not in numerical_cols:
        print(f"  ⚠️  核心特征 '{core_feat}' 不存在于数据中，跳过")
        continue
    
    # 计算与所有其他特征的相关性（绝对值）
    correlations = train_df[numerical_cols].corr()[core_feat].abs()
    
    # 筛选相关性>0.1的特征（排除自身）
    correlated_feats = correlations[
        (correlations > 0.1) & (correlations.index != core_feat)
    ].index.tolist()
    
    selected_features.update(correlated_feats)
    selected_features.add(core_feat)  # 包含核心特征本身
    
    # 记录详细信息
    correlation_details[core_feat] = {
        'correlated_count': len(correlated_feats),
        'top_correlations': correlations[correlated_feats].sort_values(ascending=False).head(10)
    }
    
    print(f"\n  📊 核心特征 '{core_feat}' 的相关分析:")
    print(f"     找到 {len(correlated_feats)} 个相关特征 (>0.1)")
    print(f"     Top 5 强相关特征:")
    for i, (feat, corr_val) in enumerate(correlations[correlated_feats].nlargest(5).items()):
        print(f"       {i+1}. {feat:20s}: {corr_val:.3f}")

# 转换为排序列表
selected_features = sorted(list(selected_features))

print(f"\n{'='*80}")
print(f"✅ 特征筛选完成!")
print(f"   原始特征数: {len(numerical_cols)}")
print(f"   筛选后特征数: {len(selected_features)}")
print(f"   筛选掉噪声特征: {len(numerical_cols) - len(selected_features)} 个")
print(f"   筛选比例: {len(selected_features)/len(numerical_cols)*100:.1f}%")
print(f"{'='*80}")

# ========================
# 4. 确保训练集和测试集特征一致性
# ========================
print("\n【4】确保训练集和测试集特征一致性...")

# 只保留训练集和测试集都存在的特征
common_features = [feat for feat in selected_features if feat in test_df.columns]

missing_in_test = set(selected_features) - set(common_features)
if missing_in_test:
    print(f"⚠️  测试集缺少的特征: {missing_in_test}")
    print(f"   将从聚类特征中排除这些特征")

print(f"最终用于聚类的共同特征数: {len(common_features)}")

# 提取共同特征数据
X_train_cluster = train_df[common_features].values
X_test_cluster = test_df[common_features].values

print(f"训练集聚类数据形状: {X_train_cluster.shape}")
print(f"测试集聚类数据形状: {X_test_cluster.shape}")

# 检查是否有缺失值
nan_count_train = np.isnan(X_train_cluster).sum()
nan_count_test = np.isnan(X_test_cluster).sum()
print(f"训练集缺失值: {nan_count_train}")
print(f"测试集缺失值: {nan_count_test}")

if nan_count_train > 0 or nan_count_test > 0:
    print("⚠️  警告：聚类数据存在缺失值！")
else:
    print("✅ 聚类数据完整")

# ========================
# 5. 标准化处理
# ========================
print("\n【5】标准化处理（Z-score）...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_cluster)
X_test_scaled = scaler.transform(X_test_cluster)

print(f"标准化后训练集统计:")
print(f"  均值范围: [{X_train_scaled.mean(axis=0).min():.6f}, {X_train_scaled.mean(axis=0).max():.6f}]")
print(f"  标准差范围: [{X_train_scaled.std(axis=0).min():.6f}, {X_train_scaled.std(axis=0).max():.6f}]")
print("✅ 标准化完成")

# ========================
# 6. 确定最优簇数（手肘法 + 轮廓系数）
# ========================
print("\n【6】确定最优簇数 - 手肘法 + 轮廓系数...")
print("  → 尝试簇数范围: k = 5 ~ 15")

# 为了加速计算，使用抽样数据
sample_size = min(20000, len(X_train_scaled))
if sample_size < len(X_train_scaled):
    print(f"  → 从 {len(X_train_scaled)} 样本中抽取 {sample_size} 个进行快速评估...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    X_sample = X_train_scaled[sample_indices]
else:
    X_sample = X_train_scaled
    print(f"  → 使用全部 {len(X_train_scaled)} 样本进行评估")

# 尝试的簇数范围
k_range = range(5, 16)  # 5-15

inertias = []           # SSE（惯性）
silhouette_scores = []  # 轮廓系数

print("\n  开始计算不同k值的聚类质量...")
for k in k_range:
    print(f"    计算 k={k:2d}...", end=' ')
    
    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,      # 运行10次取最优
        max_iter=300,
        algorithm='lloyd'
    )
    
    labels_temp = kmeans_temp.fit_predict(X_sample)
    
    inertia = kmeans_temp.inertia_
    sil_score = silhouette_score(X_sample, labels_temp)
    
    inertias.append(inertia)
    silhouette_scores.append(sil_score)
    
    print(f"SSE={inertia:12,.0f}, 轮廓系数={sil_score:.4f}")

# 找到最优簇数（轮廓系数最大）
best_k_idx = np.argmax(silhouette_scores)
best_k = list(k_range)[best_k_idx]
best_silhouette = silhouette_scores[best_k_idx]
best_inertia = inertias[best_k_idx]

print(f"\n{'='*80}")
print(f"✅ 最优簇数确定完成!")
print(f"   最优簇数: k = {best_k}")
print(f"   最佳轮廓系数: {best_silhouette:.4f}")
print(f"   对应SSE: {best_inertia:,.0f}")
print(f"{'='*80}")

# ========================
# 7. 可视化手肘法和轮廓系数
# ========================
print("\n【7】绘制最优簇数选择图...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：手肘法
axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2.5, markersize=10, label='SSE')
axes[0].axvline(x=best_k, color='red', linestyle='--', linewidth=2.5, label=f'最优k={best_k}')
axes[0].scatter([best_k], [best_inertia], color='red', s=200, zorder=5, marker='*')
axes[0].set_xlabel('簇数 (k)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('SSE (惯性)', fontsize=13, fontweight='bold')
axes[0].set_title('手肘法 - 确定最优簇数', fontsize=15, fontweight='bold')
axes[0].legend(fontsize=12, loc='best')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].tick_params(labelsize=11)

# 右图：轮廓系数
axes[1].plot(list(k_range), silhouette_scores, 'ro-', linewidth=2.5, markersize=10, label='轮廓系数')
axes[1].axvline(x=best_k, color='blue', linestyle='--', linewidth=2.5, label=f'最优k={best_k}')
axes[1].scatter([best_k], [best_silhouette], color='blue', s=200, zorder=5, marker='*')
axes[1].set_xlabel('簇数 (k)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('轮廓系数', fontsize=13, fontweight='bold')
axes[1].set_title('轮廓系数 - 确定最优簇数', fontsize=15, fontweight='bold')
axes[1].legend(fontsize=12, loc='best')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('cluster_optimization_k5_15.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存: cluster_optimization_k5_15.png")
plt.show()

# ========================
# 8. 使用最优簇数进行K-Means聚类（全量数据）
# ========================
print(f"\n【8】执行K-Means聚类 (k={best_k}) - 全量数据...")

kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
    max_iter=300,
    algorithm='lloyd'
)

# 在完整训练集上拟合
train_labels = kmeans_final.fit_predict(X_train_scaled)
# 预测测试集
test_labels = kmeans_final.predict(X_test_scaled)

print(f"训练集聚类完成: {len(train_labels)} 个样本")
print(f"测试集聚类完成: {len(test_labels)} 个样本")

# 最终聚类质量指标
final_inertia = kmeans_final.inertia_
final_silhouette = silhouette_score(X_train_scaled, train_labels)

print(f"\n最终聚类质量:")
print(f"  SSE (惯性): {final_inertia:,.0f}")
print(f"  轮廓系数: {final_silhouette:.4f}")
print(f"  迭代次数: {kmeans_final.n_iter_}")
print(f"  收敛状态: {'✅ 已收敛' if kmeans_final.n_iter_ < 300 else '⚠️  达到最大迭代'}")

# ========================
# 9. 聚类结果分析
# ========================
print("\n【9】聚类结果详细分析...")

# 添加簇标签到DataFrame
train_df['cluster_label'] = train_labels
test_df['cluster_label'] = test_labels

# 统计每个簇的样本数
print("\n📊 各簇样本分布:")
cluster_counts = train_df['cluster_label'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    pct = count / len(train_df) * 100
    bar = '█' * int(pct / 2)
    print(f"  簇 {cluster_id}: {count:6d} 样本 ({pct:5.2f}%) {bar}")

# 均衡性检查
balance_ratio = cluster_counts.min() / cluster_counts.max()
print(f"\n  簇均衡性: {balance_ratio:.2f} ({'良好' if balance_ratio > 0.3 else '需关注'})")

# 分析每个簇的价格统计（如果存在price列）
if 'price' in train_df.columns:
    print("\n💰 各簇的价格统计:")
    cluster_stats = train_df.groupby('cluster_label')['price'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ])
    cluster_stats.columns = ['平均价格', '中位数', '标准差', '最低价', '最高价', '样本数']
    cluster_stats = cluster_stats.round(2)
    print(cluster_stats)
    
    # 价格差异分析
    price_range = cluster_stats['平均价格'].max() - cluster_stats['平均价格'].min()
    print(f"\n  各簇平均价格差异: {price_range:.2f} 元")

# 分析核心业务特征在各簇的分布
print("\n📈 核心业务特征在各簇的均值:")
for feat in core_features:
    if feat in train_df.columns:
        print(f"\n  {feat}:")
        cluster_means = train_df.groupby('cluster_label')[feat].mean()
        for cluster_id, mean_val in cluster_means.items():
            print(f"    簇 {cluster_id}: {mean_val:.2f}")

# ========================
# 10. 可视化聚类结果（PCA降维）
# ========================
print("\n【10】可视化聚类结果（PCA降维到2D）...")

from sklearn.decomposition import PCA

# PCA降维
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_var = pca.explained_variance_ratio_.sum()
print(f"PCA解释方差比: {explained_var:.2%} (PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%})")

# 绘制训练集聚类结果
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    X_train_pca[:, 0], 
    X_train_pca[:, 1],
    c=train_labels,
    cmap='viridis',
    alpha=0.6,
    s=15,
    edgecolors='none'
)
plt.colorbar(scatter, label='簇标签')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)', fontsize=13, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)', fontsize=13, fontweight='bold')
plt.title(f'K-Means聚类结果 (k={best_k}) - 训练集\n轮廓系数={final_silhouette:.4f}', 
          fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('cluster_visualization_train_pca.png', dpi=300, bbox_inches='tight')
print("✅ 训练集聚类可视化已保存: cluster_visualization_train_pca.png")
plt.show()

# ========================
# 11. 保存聚类结果
# ========================
print("\n【11】保存聚类结果...")

# 恢复ID和目标变量
if train_saleid is not None:
    train_df['SaleID'] = train_saleid
if train_price is not None:
    train_df['price'] = train_price
if test_saleid is not None:
    test_df['SaleID'] = test_saleid

# 保存带簇标签的数据
train_df.to_csv('train_with_clusters.csv', index=False)
test_df.to_csv('test_with_clusters.csv', index=False)

print("✅ 训练集已保存到: train_with_clusters.csv")
print("✅ 测试集已保存到: test_with_clusters.csv")

# 保存聚类模型
import joblib
joblib.dump(kmeans_final, 'kmeans_model.pkl')
joblib.dump(scaler, 'cluster_scaler.pkl')
print("✅ 聚类模型已保存到: kmeans_model.pkl")
print("✅ 标准化器已保存到: cluster_scaler.pkl")

# ========================
# 12. 总结报告
# ========================
print("\n" + "=" * 80)
print("📊 聚类分析完整总结报告")
print("=" * 80)

print(f"""
✅ 完成的聚类分析流程:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣  数据准备
   - 训练集: {train_df.shape[0]:,} 样本 × {train_df.shape[1]} 特征
   - 测试集: {test_df.shape[0]:,} 样本 × {test_df.shape[1]} 特征
   - 缺失值: 训练集={train_df.isnull().sum().sum()}, 测试集={test_df.isnull().sum().sum()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2️⃣  特征筛选（关键步骤）
   - 核心业务特征: {', '.join(core_features)}
   - 原始数值特征: {len(numerical_cols)} 个
   - 筛选标准: 与核心特征相关性 > 0.1
   - 保留特征: {len(selected_features)} 个
   - 剔除噪声: {len(numerical_cols) - len(selected_features)} 个
   - 共同特征: {len(common_features)} 个（训练集和测试集一致）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3️⃣  最优簇数确定
   - 尝试范围: k = 5 ~ 15
   - 评估方法: 手肘法 + 轮廓系数
   - 抽样评估: {sample_size:,} 样本
   - ✅ 最优簇数: k = {best_k}
   - 最佳轮廓系数: {best_silhouette:.4f}
   - 对应手肘SSE: {best_inertia:,.0f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4️⃣  K-Means聚类（全量数据）
   - 算法: K-Means (k={best_k})
   - 初始化: 10次随机启动
   - 最大迭代: 300次
   - 实际迭代: {kmeans_final.n_iter_} 次
   - 训练集: {len(train_labels):,} 样本
   - 测试集: {len(test_labels):,} 样本
   - 最终SSE: {final_inertia:,.0f}
   - 最终轮廓系数: {final_silhouette:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5️⃣  聚类质量评估
   - 簇数: {best_k}
   - 最大簇: 簇{cluster_counts.idxmax()} ({cluster_counts.max():,} 样本, {cluster_counts.max()/len(train_df)*100:.1f}%)
   - 最小簇: 簇{cluster_counts.idxmin()} ({cluster_counts.min():,} 样本, {cluster_counts.min()/len(train_df)*100:.1f}%)
   - 均衡性: {balance_ratio:.2f} ({'✅ 良好' if balance_ratio > 0.3 else '⚠️  需关注'})
   - 轮廓系数: {final_silhouette:.4f} ({'✅ 结构清晰' if final_silhouette > 0.5 else '⚠️  结构一般' if final_silhouette > 0.25 else '❌ 结构模糊'})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 输出文件清单:
   1. train_with_clusters.csv - 带簇标签的训练集
   2. test_with_clusters.csv - 带簇标签的测试集
   3. cluster_optimization_k5_15.png - 最优簇数选择图
   4. cluster_visualization_train_pca.png - 聚类可视化图
   5. kmeans_model.pkl - 聚类模型
   6. cluster_scaler.pkl - 标准化器

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 下一步应用建议:
   
   🎯 方案1: 簇标签作为新特征
      → 将 cluster_label 加入特征集，直接用于XGBoost建模
   
   🎯 方案2: 分层建模
      → 针对不同簇分别训练独立的预测模型
      → 适合各簇特征差异明显的场景
   
   🎯 方案3: 簇内精填
      → 基于簇标签，对每个簇内的缺失值进行精细化填充
      → 使用簇内中位数/KNN，提升填充质量
   
   🎯 方案4: 业务洞察
      → 分析各簇的业务含义（如：高端车簇、经济车簇等）
      → 指导后续的特征工程和模型优化

""")

print("=" * 80)
print("🎉 基于粗填特征的聚类分析完成！")
print("=" * 80)
