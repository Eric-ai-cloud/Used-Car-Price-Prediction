"""
方案A优化版 - 处理K5极小簇问题
在方案A基础上，将簇4（仅15样本）合并到最相似的簇

优化内容：
1. 分析簇4特征，找到最相似的簇进行合并
2. 移除4个冗余特征
3. 添加5个新特征（多项式+交互）
4. 使用CatBoost重新训练

预期效果：RMSE进一步降低
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 方案A优化版 - 处理K5极小簇 + CatBoost建模")
print("=" * 80)

# ========================
# 配置参数
# ========================
CLUSTER_VERSION = 'k5'
ITERATIONS = 8000
LEARNING_RATE = 0.02
MERGE_TINY_CLUSTER = True  # 是否合并极小簇

print(f"\n📋 配置:")
print(f"  • 聚类版本: {CLUSTER_VERSION.upper()}")
print(f"  • 迭代次数: {ITERATIONS}")
print(f"  • 学习率: {LEARNING_RATE}")
print(f"  • 合并极小簇: {MERGE_TINY_CLUSTER}")

# ========================
# 1. 数据加载
# ========================
print("\n【1】数据加载...")

if CLUSTER_VERSION == 'k5':
    train_file = 'train_final_k5.csv'
    test_file = 'test_final_k5.csv'
else:
    train_file = 'train_final_k6.csv'
    test_file = 'test_final_k6.csv'

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# ========================
# 2. 处理极小簇（簇4）
# ========================
if MERGE_TINY_CLUSTER and 'cluster_label' in train_df.columns:
    print("\n【2】处理极小簇...")
    
    # 检查各簇样本数
    cluster_counts = train_df['cluster_label'].value_counts().sort_index()
    print(f"\n当前簇分布:")
    for cluster_id, count in cluster_counts.items():
        pct = count / len(train_df) * 100
        status = "⚠️ 极小簇" if count < 100 else "✅"
        print(f"  簇{cluster_id}: {count:>6,} 样本 ({pct:5.2f}%) {status}")
    
    # 识别极小簇（样本数<100）
    tiny_clusters = cluster_counts[cluster_counts < 100].index.tolist()
    
    if tiny_clusters:
        print(f"\n⚠️  发现{len(tiny_clusters)}个极小簇: {tiny_clusters}")
        
        for tiny_cluster in tiny_clusters:
            print(f"\n  处理簇{tiny_cluster} ({cluster_counts[tiny_cluster]}样本)...")
            
            # 获取极小簇的特征均值
            tiny_data = train_df[train_df['cluster_label'] == tiny_cluster]
            tiny_means = tiny_data[['power', 'kilometer', 'carAge', 'price']].mean()
            
            print(f"    簇{tiny_cluster}特征均值:")
            print(f"      power: {tiny_means['power']:.2f}")
            print(f"      kilometer: {tiny_means['kilometer']:.2f}")
            print(f"      carAge: {tiny_means['carAge']:.2f}")
            print(f"      price: {tiny_means['price']:.2f}")
            
            # 计算与其他簇的欧氏距离（基于标准化特征）
            other_clusters = [c for c in cluster_counts.index if c != tiny_cluster and cluster_counts[c] >= 100]
            
            distances = {}
            for other_cluster in other_clusters:
                other_data = train_df[train_df['cluster_label'] == other_cluster]
                other_means = other_data[['power', 'kilometer', 'carAge', 'price']].mean()
                
                # 计算标准化欧氏距离
                features = ['power', 'kilometer', 'carAge', 'price']
                distance = 0
                for feat in features:
                    std = train_df[train_df['cluster_label'] == other_cluster][feat].std()
                    if std > 0:
                        diff = (tiny_means[feat] - other_means[feat]) / std
                        distance += diff ** 2
                
                distance = np.sqrt(distance)
                distances[other_cluster] = distance
            
            # 找到最相似的簇
            most_similar = min(distances, key=distances.get)
            print(f"\n    与其他簇的距离:")
            for cluster_id, dist in sorted(distances.items()):
                marker = " ← 最相似" if cluster_id == most_similar else ""
                print(f"      簇{cluster_id}: {dist:.4f}{marker}")
            
            print(f"\n    ✅ 将簇{tiny_cluster}合并到簇{most_similar}")
            
            # 执行合并
            train_df.loc[train_df['cluster_label'] == tiny_cluster, 'cluster_label'] = most_similar
            if 'cluster_label' in test_df.columns:
                test_df.loc[test_df['cluster_label'] == tiny_cluster, 'cluster_label'] = most_similar
            
            print(f"    合并完成！")
    
    # 验证合并后的分布
    print(f"\n合并后的簇分布:")
    new_cluster_counts = train_df['cluster_label'].value_counts().sort_index()
    for cluster_id, count in new_cluster_counts.items():
        pct = count / len(train_df) * 100
        print(f"  簇{cluster_id}: {count:>6,} 样本 ({pct:5.2f}%)")
    
    print(f"\n✅ 极小簇处理完成！")
else:
    print("\n【2】跳过极小簇处理")

# ========================
# 3. 方案A优化：移除冗余特征
# ========================
print("\n【3】方案A优化 - 移除冗余特征...")

features_to_drop = [
    'carAgeMonth',     # 与carAge高度相关 (r=0.9984)
    'regYear',         # 与carAge完全相关 (r=-1.0000)
    'isYearStart',     # 与creatQuarter高度相关 (r=0.9987)
    'creatYear',       # 方差极低，信息量少
]

print(f"准备移除的特征:")
for feat in features_to_drop:
    if feat in train_df.columns:
        print(f"  ✓ {feat}")
    else:
        print(f"  ✗ {feat} (不存在)")

# 只移除存在的特征
features_to_drop = [f for f in features_to_drop if f in train_df.columns]
print(f"\n实际移除: {len(features_to_drop)} 个特征")

# 从训练集和测试集中移除
train_df_optimized = train_df.drop(columns=features_to_drop)
test_df_optimized = test_df.drop(columns=[f for f in features_to_drop if f in test_df.columns])

print(f"优化后训练集形状: {train_df_optimized.shape}")
print(f"优化后测试集形状: {test_df_optimized.shape}")

# ========================
# 4. 方案A优化：添加多项式和交互特征
# ========================
print("\n【4】方案A优化 - 添加新特征...")

new_features_count = 0

# 4.1 多项式特征
print("\n  4.1 添加多项式特征:")

# power²
if 'power' in train_df_optimized.columns:
    train_df_optimized['power_squared'] = train_df_optimized['power'] ** 2
    test_df_optimized['power_squared'] = test_df_optimized['power'] ** 2
    print(f"    ✓ power_squared (power²)")
    new_features_count += 1

# carAge²
if 'carAge' in train_df_optimized.columns:
    train_df_optimized['carAge_squared'] = train_df_optimized['carAge'] ** 2
    test_df_optimized['carAge_squared'] = test_df_optimized['carAge'] ** 2
    print(f"    ✓ carAge_squared (carAge²)")
    new_features_count += 1

# kilometer²
if 'kilometer' in train_df_optimized.columns:
    train_df_optimized['kilometer_squared'] = train_df_optimized['kilometer'] ** 2
    test_df_optimized['kilometer_squared'] = test_df_optimized['kilometer'] ** 2
    print(f"    ✓ kilometer_squared (kilometer²)")
    new_features_count += 1

# 4.2 交互特征
print("\n  4.2 添加交互特征:")

# power × carAge
if 'power' in train_df_optimized.columns and 'carAge' in train_df_optimized.columns:
    train_df_optimized['power_x_carAge'] = train_df_optimized['power'] * train_df_optimized['carAge']
    test_df_optimized['power_x_carAge'] = test_df_optimized['power'] * test_df_optimized['carAge']
    print(f"    ✓ power_x_carAge (power × carAge)")
    new_features_count += 1

# power × kilometer
if 'power' in train_df_optimized.columns and 'kilometer' in train_df_optimized.columns:
    train_df_optimized['power_x_kilometer'] = train_df_optimized['power'] * train_df_optimized['kilometer']
    test_df_optimized['power_x_kilometer'] = test_df_optimized['power'] * test_df_optimized['kilometer']
    print(f"    ✓ power_x_kilometer (power × kilometer)")
    new_features_count += 1

print(f"\n  共添加 {new_features_count} 个新特征")
print(f"优化后训练集形状: {train_df_optimized.shape}")
print(f"优化后测试集形状: {test_df_optimized.shape}")

# ========================
# 5. 准备训练数据
# ========================
print("\n【5】准备训练数据...")

# 确定目标变量和ID列
target_col = 'price'
id_col = 'SaleID'

# 排除的列
exclude_cols = [target_col, id_col]

# 获取所有特征列（数值型）
feature_cols = [col for col in train_df_optimized.columns 
                if col not in exclude_cols 
                and train_df_optimized[col].dtype in ['int64', 'float64', 'int32', 'float32']]

print(f"特征数量: {len(feature_cols)}")
print(f"前10个特征: {feature_cols[:10]}")

# 分离特征和目标
X = train_df_optimized[feature_cols]
y = train_df_optimized[target_col]
X_test = test_df_optimized[feature_cols]

print(f"\n训练集特征矩阵: {X.shape}")
print(f"训练集目标变量: {y.shape}")
print(f"测试集特征矩阵: {X_test.shape}")

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n训练集: {X_train.shape}")
print(f"验证集: {X_val.shape}")

# ========================
# 6. CatBoost模型训练
# ========================
print("\n【6】CatBoost模型训练...")

# 定义类别特征（如果有）
cat_features = [col for col in feature_cols if train_df_optimized[col].dtype == 'object']
if cat_features:
    print(f"检测到{len(cat_features)}个类别特征: {cat_features[:5]}")
else:
    print("未检测到类别特征（全部为数值型）")

# 创建CatBoost池
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# 模型配置
model_params = {
    'iterations': ITERATIONS,
    'learning_rate': LEARNING_RATE,
    'depth': 8,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 200,
    'early_stopping_rounds': 100,
    'use_best_model': True,
}

print(f"\n模型配置:")
for key, value in model_params.items():
    print(f"  {key}: {value}")

# 训练模型
print(f"\n开始训练...")
model = CatBoostRegressor(**model_params)
model.fit(
    train_pool,
    eval_set=val_pool,
)

# ========================
# 7. 模型评估
# ========================
print("\n【7】模型评估...")

# 验证集预测
y_val_pred = model.predict(val_pool)

# 计算指标
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"\n验证集性能:")
print(f"  RMSE: {rmse_val:.2f}")
print(f"  MAE:  {mae_val:.2f}")
print(f"  R²:   {r2_val:.4f}")

# 全量训练集预测（用于对比）
full_train_pool = Pool(X, y, cat_features=cat_features)
y_train_pred = model.predict(full_train_pool)

rmse_train = np.sqrt(mean_squared_error(y, y_train_pred))
mae_train = mean_absolute_error(y, y_train_pred)
r2_train = r2_score(y, y_train_pred)

print(f"\n训练集性能:")
print(f"  RMSE: {rmse_train:.2f}")
print(f"  MAE:  {mae_train:.2f}")
print(f"  R²:   {r2_train:.4f}")

# ========================
# 8. 特征重要性分析
# ========================
print("\n【8】特征重要性分析...")

feature_importance = model.get_feature_importance()
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 20 重要特征:")
print(feature_importance_df.head(20).to_string(index=False))

# ========================
# 9. 生成预测结果
# ========================
print("\n【9】生成预测结果...")

# 测试集预测
test_pool = Pool(X_test, cat_features=cat_features)
y_test_pred = model.predict(test_pool)

# 创建提交文件
submission = pd.DataFrame({
    'SaleID': test_df_optimized[id_col],
    'price': y_test_pred
})

# 保存预测结果
output_file = f'submission_catboost_{CLUSTER_VERSION}_optimized_A_merged.csv'
submission.to_csv(output_file, index=False)
print(f"✅ 预测结果已保存: {output_file}")
print(f"预测样本数: {len(submission)}")
print(f"预测价格范围: [{y_test_pred.min():.2f}, {y_test_pred.max():.2f}]")

# 保存模型
model_file = f'catboost_model_{CLUSTER_VERSION}_optimized_A_merged.cbm'
model.save_model(model_file)
print(f"✅ 模型已保存: {model_file}")

# ========================
# 10. 总结报告
# ========================
print("\n" + "=" * 80)
print("📊 方案A优化版总结报告")
print("=" * 80)

print(f"\n🔧 优化内容:")
print(f"  1. 合并极小簇: {'是' if MERGE_TINY_CLUSTER else '否'}")
if MERGE_TINY_CLUSTER and tiny_clusters:
    for tc in tiny_clusters:
        print(f"     • 簇{tc} → 合并到最相似簇")
print(f"  2. 移除冗余特征: {len(features_to_drop)}个")
for feat in features_to_drop:
    print(f"     • {feat}")
print(f"  3. 添加新特征: {new_features_count}个")
print(f"     • 多项式特征: power_squared, carAge_squared, kilometer_squared")
print(f"     • 交互特征: power_x_carAge, power_x_kilometer")

print(f"\n📈 模型性能:")
print(f"  验证集 RMSE: {rmse_val:.2f}")
print(f"  验证集 MAE:  {mae_val:.2f}")
print(f"  验证集 R²:   {r2_val:.4f}")

print(f"\n💾 生成文件:")
print(f"  • {output_file} - 预测结果（推荐提交）")
print(f"  • {model_file} - 训练好的模型")

print(f"\n{'='*80}")
print("✅ 方案A优化版完成！")
print(f"{'='*80}")
