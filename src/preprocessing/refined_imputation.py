import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔧 第三步：簇内精填修正")
print("=" * 80)

# ========================
# 1. 数据加载
# ========================
print("\n【1】加载带簇标签的数据...")
train_df = pd.read_csv('train_with_clusters.csv')
test_df = pd.read_csv('test_with_clusters.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 保存原始ID和目标变量
train_saleid = train_df['SaleID'].copy() if 'SaleID' in train_df.columns else None
test_saleid = test_df['SaleID'].copy() if 'SaleID' in test_df.columns else None
train_price = train_df['price'].copy() if 'price' in train_df.columns else None

# 获取簇标签
train_clusters = train_df['cluster_label'].copy()
test_clusters = test_df['cluster_label'].copy()

# ========================
# 2. 识别需要精填的特征
# ========================
print("\n【2】识别需要精填的特征...")

# 从原始数据中检查哪些特征有缺失值
# 我们需要回到预处理前的数据来查看原始缺失情况
train_original = pd.read_csv('train_featured.csv')
test_original = pd.read_csv('test_featured.csv')

# 找出在原始数据中有缺失值的特征（只检查训练集和测试集都存在的列）
missing_features = []
common_cols = [col for col in train_original.columns if col in test_original.columns and col not in ['SaleID', 'price', 'name']]

for col in common_cols:
    if train_original[col].isnull().sum() > 0 or test_original[col].isnull().sum() > 0:
        missing_features.append(col)

print(f"需要精填的特征数: {len(missing_features)}")
print("缺失特征列表:")
for feat in missing_features[:15]:  # 只显示前15个
    train_missing = train_original[feat].isnull().sum()
    test_missing = test_original[feat].isnull().sum()
    print(f"  {feat:30s}: 训练集={train_missing}, 测试集={test_missing}")

# ========================
# 3. 定义精填策略函数
# ========================
print("\n【3】定义簇内精填策略...")

def cluster_refined_imputation(train_df, test_df, cluster_col, missing_features, k_neighbors=5):
    """
    簇内精填修正
    
    参数:
        train_df: 训练集DataFrame（带簇标签）
        test_df: 测试集DataFrame（带簇标签）
        cluster_col: 簇标签列名
        missing_features: 需要填充的特征列表
        k_neighbors: KNN的近邻数
    
    返回:
        精填后的训练集和测试集
    """
    
    train_refined = train_df.copy()
    test_refined = test_df.copy()
    
    # 获取所有簇
    clusters = sorted(train_df[cluster_col].unique())
    print(f"\n处理 {len(clusters)} 个簇的精填...")
    
    for cluster_id in clusters:
        print(f"\n{'='*60}")
        print(f"处理簇 {cluster_id}...")
        
        # 获取当前簇的训练集和测试集索引
        train_mask = train_df[cluster_col] == cluster_id
        test_mask = test_df[cluster_col] == cluster_id
        
        train_cluster = train_df[train_mask].copy()
        test_cluster = test_df[test_mask].copy()
        
        print(f"  训练集样本数: {len(train_cluster)}")
        print(f"  测试集样本数: {len(test_cluster)}")
        
        # ========================
        # 数值型特征精填
        # ========================
        numerical_missing = [col for col in missing_features 
                            if col in train_cluster.columns 
                            and train_cluster[col].dtype in ['float64', 'int64']]
        
        if numerical_missing:
            print(f"  → 数值型特征精填 ({len(numerical_missing)}个)...")
            
            for col in numerical_missing:
                # 检查当前簇内的缺失情况
                train_missing_count = train_cluster[col].isnull().sum()
                test_missing_count = test_cluster[col].isnull().sum()
                
                if train_missing_count == 0 and test_missing_count == 0:
                    continue
                
                # 获取当前簇内该特征的完整样本
                train_complete = train_cluster[col].dropna()
                
                # 兜底机制：如果完整样本数不足，保留粗填结果
                if len(train_complete) < k_neighbors:
                    print(f"    ⚠️  {col}: 完整样本不足({len(train_complete)}<{k_neighbors})，保留粗填结果")
                    continue
                
                # 策略1：优先使用簇内中位数填充（更稳健，适配偏态分布）
                cluster_median = train_complete.median()
                
                # 对于缺失比例较低的，直接用中位数
                train_missing_ratio = train_missing_count / len(train_cluster)
                test_missing_ratio = test_missing_count / len(test_cluster) if len(test_cluster) > 0 else 0
                
                if train_missing_ratio < 0.1:  # 缺失比例<10%，用中位数
                    train_refined.loc[train_mask & train_refined[col].isnull(), col] = cluster_median
                    if len(test_cluster) > 0:
                        test_refined.loc[test_mask & test_refined[col].isnull(), col] = cluster_median
                    print(f"    ✓ {col}: 使用中位数填充 (缺失率={train_missing_ratio:.1%})")
                else:
                    # 缺失比例较高，使用簇内KNN精填
                    print(f"    → {col}: 缺失率较高({train_missing_ratio:.1%})，使用簇内KNN精填...")
                    
                    # 选择用于KNN的相关特征（与目标特征相关性高的）
                    candidate_feats = [f for f in train_cluster.columns 
                                      if f not in ['SaleID', 'price', 'name', cluster_col]
                                      and train_cluster[f].dtype in ['float64', 'int64']
                                      and f != col]
                    
                    # 计算相关性，选择Top 10相关特征
                    if len(candidate_feats) > 0:
                        correlations = train_cluster[candidate_feats + [col]].corr()[col].abs()
                        top_correlated = correlations.drop(col).nlargest(min(10, len(candidate_feats))).index.tolist()
                        
                        knn_features = [col] + top_correlated
                        
                        # 确保特征都存在
                        knn_features = [f for f in knn_features if f in train_cluster.columns]
                        
                        if len(knn_features) > 1:
                            try:
                                # 簇内KNN精填
                                knn_imputer = KNNImputer(n_neighbors=min(k_neighbors, len(train_complete)))
                                
                                # 对训练集精填
                                train_cluster_data = train_cluster[knn_features].values
                                train_cluster_filled = knn_imputer.fit_transform(train_cluster_data)
                                train_refined.loc[train_mask, knn_features[0]] = train_cluster_filled[:, 0]
                                
                                # 对测试集精填
                                if len(test_cluster) > 0:
                                    test_cluster_data = test_cluster[knn_features].values
                                    test_cluster_filled = knn_imputer.transform(test_cluster_data)
                                    test_refined.loc[test_mask, knn_features[0]] = test_cluster_filled[:, 0]
                                
                                print(f"      ✓ {col}: 簇内KNN精填完成 (k={min(k_neighbors, len(train_complete))})")
                            except Exception as e:
                                print(f"      ⚠️ {col}: KNN精填失败，改用中位数: {str(e)[:50]}")
                                train_refined.loc[train_mask & train_refined[col].isnull(), col] = cluster_median
                                if len(test_cluster) > 0:
                                    test_refined.loc[test_mask & test_refined[col].isnull(), col] = cluster_median
                        else:
                            # 没有足够的相关特征，用中位数
                            train_refined.loc[train_mask & train_refined[col].isnull(), col] = cluster_median
                            if len(test_cluster) > 0:
                                test_refined.loc[test_mask & test_refined[col].isnull(), col] = cluster_median
                    else:
                        train_refined.loc[train_mask & train_refined[col].isnull(), col] = cluster_median
                        if len(test_cluster) > 0:
                            test_refined.loc[test_mask & test_refined[col].isnull(), col] = cluster_median
        
        # ========================
        # 类别型特征精填
        # ========================
        categorical_missing = [col for col in missing_features 
                              if col in train_cluster.columns 
                              and train_cluster[col].dtype == 'object']
        
        if categorical_missing:
            print(f"  → 类别型特征精填 ({len(categorical_missing)}个)...")
            
            for col in categorical_missing:
                train_missing_count = train_cluster[col].isnull().sum()
                test_missing_count = test_cluster[col].isnull().sum() if len(test_cluster) > 0 else 0
                
                if train_missing_count == 0 and test_missing_count == 0:
                    continue
                
                # 策略：使用簇内众数填充
                cluster_mode = train_cluster[col].mode()
                
                if len(cluster_mode) > 0:
                    mode_value = cluster_mode[0]
                    
                    # 检查簇内缺失比例
                    train_missing_ratio = train_missing_count / len(train_cluster)
                    
                    if train_missing_ratio > 0.5:  # 缺失比例>50%，保留"未知"类别
                        unknown_value = 'Unknown'
                        train_refined.loc[train_mask & train_refined[col].isnull(), col] = unknown_value
                        if len(test_cluster) > 0:
                            test_refined.loc[test_mask & test_refined[col].isnull(), col] = unknown_value
                        print(f"    ✓ {col}: 缺失率高({train_missing_ratio:.1%})，填充为'Unknown'")
                    else:
                        # 使用众数填充
                        train_refined.loc[train_mask & train_refined[col].isnull(), col] = mode_value
                        if len(test_cluster) > 0:
                            test_refined.loc[test_mask & test_refined[col].isnull(), col] = mode_value
                        print(f"    ✓ {col}: 使用众数'{mode_value}'填充 (缺失率={train_missing_ratio:.1%})")
                else:
                    print(f"    ⚠️ {col}: 无法确定众数，跳过")
    
    return train_refined, test_refined

# ========================
# 4. 执行簇内精填
# ========================
print("\n【4】执行簇内精填修正...")

train_refined, test_refined = cluster_refined_imputation(
    train_df=train_df,
    test_df=test_df,
    cluster_col='cluster_label',
    missing_features=missing_features,
    k_neighbors=5
)

# ========================
# 5. 验证精填结果
# ========================
print("\n【5】验证精填结果...")

# 检查是否还有缺失值
train_missing_after = train_refined.isnull().sum()
test_missing_after = test_refined.isnull().sum()

train_missing_cols = train_missing_after[train_missing_after > 0]
test_missing_cols = test_missing_after[test_missing_after > 0]

print(f"\n训练集剩余缺失值统计:")
if len(train_missing_cols) > 0:
    for col, count in train_missing_cols.items():
        print(f"  {col}: {count}")
else:
    print("  ✅ 无缺失值")

print(f"\n测试集剩余缺失值统计:")
if len(test_missing_cols) > 0:
    for col, count in test_missing_cols.items():
        print(f"  {col}: {count}")
else:
    print("  ✅ 无缺失值")

# ========================
# 6. 恢复ID和目标变量
# ========================
print("\n【6】恢复ID和目标变量...")

if train_saleid is not None:
    train_refined['SaleID'] = train_saleid
if train_price is not None:
    train_refined['price'] = train_price
if test_saleid is not None:
    test_refined['SaleID'] = test_saleid

# ========================
# 7. 保存精填后的数据
# ========================
print("\n【7】保存精填后的数据...")

train_refined.to_csv('train_refined.csv', index=False)
test_refined.to_csv('test_refined.csv', index=False)

print("✅ 训练集已保存到: train_refined.csv")
print("✅ 测试集已保存到: test_refined.csv")

# ========================
# 8. 总结报告
# ========================
print("\n" + "=" * 80)
print("📊 簇内精填修正总结")
print("=" * 80)

print(f"""
✅ 完成的精填步骤:

1️⃣  识别缺失特征:
   - 需要精填的特征数: {len(missing_features)}

2️⃣  按簇分别处理:
   - 处理的簇数: {len(train_df['cluster_label'].unique())}
   - 每个簇独立精填，避免跨簇污染

3️⃣  数值型特征策略:
   - 优先: 簇内中位数填充（稳健，适配偏态分布）
   - 备选: 簇内KNN精填（缺失率>10%时）
   - 兜底: 完整样本不足时保留粗填结果

4️⃣  类别型特征策略:
   - 优先: 簇内众数填充
   - 备选: 缺失率>50%时填充为'Unknown'

5️⃣  兜底机制:
   - 簇内完整样本数 < k_neighbors时，不强行精填
   - 避免引入更大误差

📁 输出文件:
   - train_refined.csv: 精填后的训练集
   - test_refined.csv: 精填后的测试集

💡 对比:
   - 粗填: 全局KNN (k=7)，快速但粗糙
   - 精填: 簇内中位数/KNN，精准且稳健
""")

print("=" * 80)
print("🎉 簇内精填修正完成！")
print("=" * 80)
