import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔄 簇内精填修正 + XGBoost建模（k=5）")
print("=" * 80)

# ========================
# 第一步：加载带簇标签的数据
# ========================
print("\n" + "=" * 80)
print("【第一步】加载带簇标签的数据（k=5）")
print("=" * 80)

train_df = pd.read_csv('train_with_clusters.csv')
test_df = pd.read_csv('test_with_clusters.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 保存ID和目标变量
train_saleid = train_df['SaleID'].copy() if 'SaleID' in train_df.columns else None
test_saleid = test_df['SaleID'].copy() if 'SaleID' in test_df.columns else None
train_price = train_df['price'].copy() if 'price' in train_df.columns else None

# 获取簇标签
train_clusters = train_df['cluster_label'].copy()
test_clusters = test_df['cluster_label'].copy()

# 检查缺失值
print(f"\n粗填后缺失值:")
print(f"  训练集: {train_df.isnull().sum().sum()}")
print(f"  测试集: {test_df.isnull().sum().sum()}")

# ========================
# 第二步：识别需要精填的特征
# ========================
print("\n" + "=" * 80)
print("【第二步】识别需要精填的特征")
print("=" * 80)

# 从原始数据中检查哪些特征有缺失值
train_original = pd.read_csv('train_preprocessed.csv')
test_original = pd.read_csv('test_preprocessed.csv')

# 找出在原始数据中有缺失值的特征
missing_features = []
common_cols = [col for col in train_original.columns 
               if col in test_original.columns 
               and col not in ['SaleID', 'price', 'name']]

for col in common_cols:
    if train_original[col].isnull().sum() > 0 or test_original[col].isnull().sum() > 0:
        missing_features.append(col)

print(f"需要精填的特征数: {len(missing_features)}")
print("\n缺失特征列表（前15个）:")
for feat in missing_features[:15]:
    train_missing = train_original[feat].isnull().sum()
    test_missing = test_original[feat].isnull().sum()
    print(f"  {feat:30s}: 训练集={train_missing}, 测试集={test_missing}")

# ========================
# 第三步：簇内精填修正
# ========================
print("\n" + "=" * 80)
print("【第三步】簇内精填修正（k=5）")
print("=" * 80)

def cluster_refined_imputation(train_df, test_df, cluster_col, missing_features, k_neighbors=5):
    """
    簇内精填修正函数
    
    参数:
        train_df: 训练集DataFrame
        test_df: 测试集DataFrame
        cluster_col: 簇标签列名
        missing_features: 需要精填的特征列表
        k_neighbors: KNN近邻数
    
    返回:
        train_refined, test_refined: 精填后的训练集和测试集
    """
    
    train_refined = train_df.copy()
    test_refined = test_df.copy()
    
    # 获取所有簇
    clusters = sorted(train_df[cluster_col].unique())
    print(f"\n处理 {len(clusters)} 个簇的精填...")
    
    total_numerical_filled = 0
    total_categorical_filled = 0
    
    for cluster_id in clusters:
        print(f"\n{'='*60}")
        print(f"处理簇 {cluster_id}...")
        print(f"{'='*60}")
        
        # 获取当前簇的训练集和测试集索引
        train_mask = train_df[cluster_col] == cluster_id
        test_mask = test_df[cluster_col] == cluster_id
        
        train_cluster = train_df[train_mask].copy()
        test_cluster = test_df[test_mask].copy()
        
        print(f"  训练集样本数: {len(train_cluster):,}")
        print(f"  测试集样本数: {len(test_cluster):,}")
        
        # ---- 数值型特征精填 ----
        numerical_missing = [col for col in missing_features 
                            if col in train_cluster.columns 
                            and train_cluster[col].dtype in ['float64', 'int64']]
        
        if numerical_missing:
            print(f"\n  → 数值型特征精填 ({len(numerical_missing)}个)...")
            
            for col in numerical_missing:
                train_missing_count = train_cluster[col].isnull().sum()
                test_missing_count = test_cluster[col].isnull().sum() if len(test_cluster) > 0 else 0
                
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
                
                # 计算缺失比例
                train_missing_ratio = train_missing_count / len(train_cluster)
                
                if train_missing_ratio < 0.1:  # 缺失比例<10%，用中位数
                    filled_count = train_missing_count
                    train_refined.loc[train_mask & train_refined[col].isnull(), col] = cluster_median
                    
                    if len(test_cluster) > 0:
                        test_refined.loc[test_mask & test_refined[col].isnull(), col] = cluster_median
                        filled_count += test_missing_count
                    
                    total_numerical_filled += filled_count
                    print(f"    ✓ {col}: 使用中位数填充 (缺失率={train_missing_ratio:.1%}, 填充{filled_count}条)")
                
                else:
                    # 缺失比例较高，使用簇内KNN精填
                    print(f"    → {col}: 缺失率较高({train_missing_ratio:.1%})，使用簇内KNN精填...")
                    
                    # 选择用于KNN的相关特征
                    candidate_feats = [f for f in train_cluster.columns 
                                      if f not in ['SaleID', 'price', 'name', cluster_col]
                                      and train_cluster[f].dtype in ['float64', 'int64']
                                      and f != col]
                    
                    if len(candidate_feats) > 0:
                        # 计算相关性，选择Top 10相关特征
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
                                
                                filled_count = train_missing_count + test_missing_count
                                total_numerical_filled += filled_count
                                print(f"      ✓ {col}: 簇内KNN精填完成 (k={min(k_neighbors, len(train_complete))}, 填充{filled_count}条)")
                            
                            except Exception as e:
                                print(f"      ⚠️ {col}: KNN精填失败，改用中位数")
                                train_refined.loc[train_mask & train_refined[col].isnull(), col] = cluster_median
                                if len(test_cluster) > 0:
                                    test_refined.loc[test_mask & test_refined[col].isnull(), col] = cluster_median
                        else:
                            train_refined.loc[train_mask & train_refined[col].isnull(), col] = cluster_median
                            if len(test_cluster) > 0:
                                test_refined.loc[test_mask & test_refined[col].isnull(), col] = cluster_median
                    else:
                        train_refined.loc[train_mask & train_refined[col].isnull(), col] = cluster_median
                        if len(test_cluster) > 0:
                            test_refined.loc[test_mask & test_refined[col].isnull(), col] = cluster_median
        
        # ---- 类别型特征精填 ----
        categorical_missing = [col for col in missing_features 
                              if col in train_cluster.columns 
                              and train_cluster[col].dtype == 'object']
        
        if categorical_missing:
            print(f"\n  → 类别型特征精填 ({len(categorical_missing)}个)...")
            
            for col in categorical_missing:
                train_missing_count = train_cluster[col].isnull().sum()
                test_missing_count = test_cluster[col].isnull().sum() if len(test_cluster) > 0 else 0
                
                if train_missing_count == 0 and test_missing_count == 0:
                    continue
                
                # 计算簇内众数
                cluster_mode = train_cluster[col].mode()
                
                if len(cluster_mode) > 0:
                    mode_value = cluster_mode[0]
                    train_missing_ratio = train_missing_count / len(train_cluster)
                    
                    if train_missing_ratio > 0.5:
                        # 缺失率过高，填充为'Unknown'
                        unknown_value = 'Unknown'
                        train_refined.loc[train_mask & train_refined[col].isnull(), col] = unknown_value
                        if len(test_cluster) > 0:
                            test_refined.loc[test_mask & test_refined[col].isnull(), col] = unknown_value
                        
                        filled_count = train_missing_count + test_missing_count
                        total_categorical_filled += filled_count
                        print(f"    ✓ {col}: 缺失率高({train_missing_ratio:.1%})，填充为'Unknown' (填充{filled_count}条)")
                    else:
                        # 使用众数填充
                        train_refined.loc[train_mask & train_refined[col].isnull(), col] = mode_value
                        if len(test_cluster) > 0:
                            test_refined.loc[test_mask & test_refined[col].isnull(), col] = mode_value
                        
                        filled_count = train_missing_count + test_missing_count
                        total_categorical_filled += filled_count
                        print(f"    ✓ {col}: 使用众数'{mode_value}'填充 (缺失率={train_missing_ratio:.1%}, 填充{filled_count}条)")
    
    print(f"\n{'='*80}")
    print(f"✅ 簇内精填完成!")
    print(f"   数值型特征填充: {total_numerical_filled} 条")
    print(f"   类别型特征填充: {total_categorical_filled} 条")
    print(f"{'='*80}")
    
    return train_refined, test_refined

# 执行簇内精填
print("\n开始执行簇内精填修正...")
train_refined, test_refined = cluster_refined_imputation(
    train_df=train_df,
    test_df=test_df,
    cluster_col='cluster_label',
    missing_features=missing_features,
    k_neighbors=5
)

# 验证精填结果
print("\n验证精填结果...")
train_missing_after = train_refined.isnull().sum().sum()
test_missing_after = test_refined.isnull().sum().sum()

print(f"训练集剩余缺失值: {train_missing_after}")
print(f"测试集剩余缺失值: {test_missing_after}")

if train_missing_after == 0 and test_missing_after == 0:
    print("✅ 精填完成，无缺失值")
else:
    print(f"⚠️  仍有缺失值: 训练集={train_missing_after}, 测试集={test_missing_after}")

# 恢复ID和目标变量
if train_saleid is not None:
    train_refined['SaleID'] = train_saleid
if train_price is not None:
    train_refined['price'] = train_price
if test_saleid is not None:
    test_refined['SaleID'] = test_saleid

# 保存精填后的数据
print("\n保存精填后的数据...")
train_refined.to_csv('train_refined_k5.csv', index=False)
test_refined.to_csv('test_refined_k5.csv', index=False)

print("✅ 训练集已保存到: train_refined_k5.csv")
print("✅ 测试集已保存到: test_refined_k5.csv")

# ========================
# 第四步：XGBoost建模
# ========================
print("\n" + "=" * 80)
print("【第四步】XGBoost建模")
print("=" * 80)

# 准备特征和目标变量
print("\n准备特征和目标变量...")

# 确定要删除的列
drop_columns = ['SaleID', 'price', 'name']
drop_columns = [col for col in drop_columns if col in train_refined.columns]

# 特征列（所有数值型特征）
feature_cols = [col for col in train_refined.columns 
                if col not in drop_columns 
                and train_refined[col].dtype in ['int64', 'float64']]

print(f"特征数量: {len(feature_cols)}")
print(f"特征列表（前20个）: {feature_cols[:20]}")

# 分离特征和目标变量
X = train_refined[feature_cols]
y = train_refined['price']

# 测试集特征（确保特征一致）
common_features = [col for col in feature_cols if col in test_refined.columns]
X_test = test_refined[common_features]

print(f"训练特征矩阵形状: {X.shape}")
print(f"测试特征矩阵形状: {X_test.shape}")

# 检查特征一致性
if len(feature_cols) != len(common_features):
    missing_in_test = set(feature_cols) - set(common_features)
    print(f"⚠️  测试集缺少的特征: {missing_in_test}")
    # 只使用共同特征
    X = train_refined[common_features]
    feature_cols = common_features
    print(f"✅ 已调整为共同特征: {len(feature_cols)}个")

# 划分训练集和验证集
print("\n划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]:,} 样本")
print(f"验证集: {X_val.shape[0]:,} 样本")

# 训练XGBoost模型
print("\n训练XGBoost模型...")

# XGBoost参数配置
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'eval_metric': 'rmse',  # 评估指标
    'max_depth': 8,  # 最大深度
    'learning_rate': 0.05,  # 学习率
    'n_estimators': 1000,  # 树的数量
    'min_child_weight': 5,  # 最小子节点权重
    'subsample': 0.8,  # 样本采样比例
    'colsample_bytree': 0.8,  # 特征采样比例
    'gamma': 0.1,  # L2正则化参数
    'lambda': 1.0,  # L1正则化参数
    'random_state': 42,
    'n_jobs': -1  # 使用所有CPU核心
}

# 创建模型
model = xgb.XGBRegressor(**params)

# 训练模型（带早停）
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=50,  # 50轮无改善则停止
    verbose=100  # 每100轮打印一次日志
)

print("\n✅ 模型训练完成！")

# 模型评估
print("\n模型评估...")

# 验证集预测
y_val_pred = model.predict(X_val)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"验证集 RMSE: {rmse:.4f}")
print(f"验证集 MAE: {mae:.4f}")
print(f"验证集 R²: {r2:.4f}")

# 特征重要性
print("\n【Top 20 重要特征】")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

# 测试集预测
print("\n测试集预测...")
test_predictions = model.predict(X_test)

# 确保预测值为正
test_predictions = np.maximum(test_predictions, 0)

print(f"测试集预测完成: {len(test_predictions)} 个样本")
print(f"预测值统计:")
print(f"  均值: {test_predictions.mean():.2f}")
print(f"  标准差: {test_predictions.std():.2f}")
print(f"  最小值: {test_predictions.min():.2f}")
print(f"  最大值: {test_predictions.max():.2f}")

# 保存提交文件
submission = pd.DataFrame({
    'SaleID': test_refined['SaleID'],
    'price': test_predictions
})

submission.to_csv('submission_xgboost_k5_refined.csv', index=False)
print("\n✅ 提交文件已保存到: submission_xgboost_k5_refined.csv")

# 保存模型
try:
    model.save_model('xgboost_model_k5_refined.json')
    print("✅ 模型已保存到: xgboost_model_k5_refined.json")
except Exception as e:
    print(f"JSON保存失败: {e}，尝试pickle保存...")
    import pickle
    with open('xgboost_model_k5_refined.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 模型已保存到: xgboost_model_k5_refined.pkl")

# ========================
# 总结报告
# ========================
print("\n" + "=" * 80)
print("📊 完整流程总结报告")
print("=" * 80)

print(f"""
✅ 完成的完整流程:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣  数据准备
   - 输入: train_with_clusters.csv, test_with_clusters.csv
   - 簇数: k=5（已通过手肘法和轮廓系数确定）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2️⃣  簇内精填修正
   - 需要精填的特征: {len(missing_features)} 个
   - 数值型策略: 簇内中位数（缺失率<10%）或簇内KNN（缺失率≥10%）
   - 类别型策略: 簇内众数（缺失率≤50%）或'Unknown'（缺失率>50%）
   - 兜底机制: 完整样本不足时保留粗填结果
   - 精填后缺失值: 训练集={train_missing_after}, 测试集={test_missing_after}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3️⃣  XGBoost建模
   - 特征数量: {len(feature_cols)} 个
   - 训练集: {X_train.shape[0]:,} 样本
   - 验证集: {X_val.shape[0]:,} 样本
   - 测试集: {X_test.shape[0]:,} 样本
   
   模型参数:
   - max_depth: 8
   - learning_rate: 0.05
   - n_estimators: 1000（实际: {model.best_iteration}）
   - subsample: 0.8
   - colsample_bytree: 0.8
   
   模型性能:
   - 验证集 RMSE: {rmse:.4f}
   - 验证集 MAE: {mae:.4f}
   - 验证集 R²: {r2:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 输出文件清单:
   1. train_refined_k5.csv - 簇内精填后的训练集
   2. test_refined_k5.csv - 簇内精填后的测试集
   3. submission_xgboost_k5_refined.csv - XGBoost预测结果
   4. xgboost_model_k5_refined.json/pkl - 训练好的模型

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 关键优势:
   ✅ 基于聚类的精细化填充，提升数据质量
   ✅ 中位数填充适配偏态分布，更稳健
   ✅ 兜底机制避免引入更大误差
   ✅ XGBoost强学习能力，自动捕捉非线性关系
   ✅ 早停机制防止过拟合

""")

print("=" * 80)
print("🎉 簇内精填 + XGBoost建模完成！")
print("=" * 80)
