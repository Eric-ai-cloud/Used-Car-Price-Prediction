"""
方案A：保守优化 - 基于K5 Final数据
优化内容：
1. 移除4个冗余特征
2. 添加5个新特征（多项式+交互）
3. 使用CatBoost重新训练

预期效果：RMSE降低10-30点
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 方案A：保守优化 - CatBoost建模")
print("=" * 80)

# ========================
# 配置参数
# ========================
CLUSTER_VERSION = 'k5'
ITERATIONS = 8000
LEARNING_RATE = 0.02

print(f"\n📋 配置:")
print(f"  • 聚类版本: {CLUSTER_VERSION.upper()}")
print(f"  • 迭代次数: {ITERATIONS}")
print(f"  • 学习率: {LEARNING_RATE}")

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
# 2. 方案A优化：移除冗余特征
# ========================
print("\n【2】方案A优化 - 移除冗余特征...")

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
# 3. 方案A优化：添加多项式和交互特征
# ========================
print("\n【3】方案A优化 - 添加新特征...")

new_features_count = 0

# 3.1 多项式特征
print("\n  3.1 添加多项式特征:")

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

# 3.2 交互特征
print("\n  3.2 添加交互特征:")

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

print(f"\n✅ 共添加 {new_features_count} 个新特征")
print(f"最终训练集形状: {train_df_optimized.shape}")
print(f"最终测试集形状: {test_df_optimized.shape}")

# ========================
# 4. 准备特征和目标变量
# ========================
print("\n【4】准备特征和目标变量...")

drop_columns = ['SaleID', 'price', 'cluster_label']
drop_columns = [col for col in drop_columns if col in train_df_optimized.columns]

target = 'price'
y_train = train_df_optimized[target]
train_X = train_df_optimized.drop(drop_columns, axis=1)
test_X = test_df_optimized.drop([col for col in drop_columns if col in test_df_optimized.columns], axis=1)

# 对齐列
common_cols = list(set(train_X.columns) & set(test_X.columns))
train_X = train_X[common_cols]
test_X = test_X[common_cols]

print(f"特征数量: {train_X.shape[1]}")

# 识别类别特征
cat_features = []
for col in train_X.columns:
    if train_X[col].dtype == 'object':
        cat_features.append(col)
    elif train_X[col].nunique() < 10 and train_X[col].dtype in ['int64', 'float64']:
        train_X[col] = train_X[col].astype(int)
        test_X[col] = test_X[col].astype(int)
        cat_features.append(col)

print(f"类别特征数量: {len(cat_features)}")

# ========================
# 5. 划分验证集
# ========================
print("\n【5】划分训练集和验证集...")

X_train, X_val, y_train_split, y_val = train_test_split(
    train_X, y_train, 
    test_size=0.2, 
    random_state=42
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")

# ========================
# 6. CatBoost模型配置
# ========================
print("\n【6】配置CatBoost模型...")

train_pool = Pool(
    data=X_train,
    label=y_train_split,
    cat_features=cat_features
)

val_pool = Pool(
    data=X_val,
    label=y_val,
    cat_features=cat_features
)

params = {
    'iterations': ITERATIONS,
    'learning_rate': LEARNING_RATE,
    'depth': 8,
    'l2_leaf_reg': 3,
    'bagging_temperature': 0.5,
    'random_strength': 0.5,
    'border_count': 254,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 100,
    'verbose': 100,
    'random_seed': 42,
    'thread_count': -1,
    'use_best_model': True,
}

print("模型参数:")
for key, value in params.items():
    if key not in ['verbose']:
        print(f"  {key}: {value}")

# ========================
# 7. 模型训练
# ========================
print("\n【7】开始训练...")

model = CatBoostRegressor(**params)

model.fit(
    train_pool,
    eval_set=val_pool,
)

print(f"\n✅ 训练完成！")
print(f"最佳迭代轮数: {model.best_iteration_}")
print(f"最佳验证集RMSE: {model.best_score_['validation']['RMSE']:.2f}")

# ========================
# 8. 模型评估
# ========================
print("\n【8】模型评估...")

y_val_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"\n📊 验证集性能:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  R²:   {r2:.4f}")

# 对比之前模型
print(f"\n📈 性能对比:")
print(f"  CatBoost (原始):")
print(f"    RMSE: 1226.33")
print(f"    MAE:  520.57")
print(f"    R²:   0.9726")
print(f"  CatBoost (方案A优化):")
print(f"    RMSE: {rmse:.2f} ({'↑改善' if rmse < 1226.33 else '→持平' if abs(rmse - 1226.33) < 5 else '↓待优化'})")
print(f"    MAE:  {mae:.2f} ({'↑改善' if mae < 520.57 else '→持平' if abs(mae - 520.57) < 2 else '↓待优化'})")
print(f"    R²:   {r2:.4f} ({'↑改善' if r2 > 0.9726 else '→持平' if abs(r2 - 0.9726) < 0.001 else '↓待优化'})")

if rmse < 1226.33:
    improvement = 1226.33 - rmse
    pct = (improvement / 1226.33) * 100
    print(f"\n🎉 RMSE降低了 {improvement:.2f} 点 ({pct:.2f}%)")
elif rmse > 1226.33:
    decline = rmse - 1226.33
    print(f"\n⚠️ RMSE上升了 {decline:.2f} 点")
else:
    print(f"\n➡️ RMSE基本持平")

# ========================
# 9. 特征重要性分析
# ========================
print("\n【9】特征重要性分析...")

importance = model.get_feature_importance()
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print(f"\nTop 20 重要特征:")
print("-" * 60)
print(f"{'排名':<6} {'特征':<25} {'重要性':<10} {'占比'}")
print("-" * 60)

for i, row in importance_df.head(20).iterrows():
    pct = row['importance']
    bar = "█" * int(pct / 2)
    print(f"{i+1:<6} {row['feature']:<25} {row['importance']:<10.4f} {pct:>5.2f}% {bar}")

# 检查新添加的特征
print(f"\n🔍 新添加特征的重要性:")
new_feature_names = ['power_squared', 'carAge_squared', 'kilometer_squared', 
                     'power_x_carAge', 'power_x_kilometer']
for feat in new_feature_names:
    if feat in importance_df['feature'].values:
        imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
        rank = importance_df[importance_df['feature'] == feat].index[0] + 1
        print(f"  {feat:<25} 重要性={imp:.4f}, 排名={rank}")
    else:
        print(f"  {feat:<25} 未进入Top特征")

# ========================
# 10. 测试集预测
# ========================
print("\n【10】测试集预测...")

test_predictions = model.predict(test_X)
test_predictions = np.maximum(test_predictions, 0)

print(f"预测结果统计:")
print(f"  均值: {test_predictions.mean():.2f}")
print(f"  中位数: {np.median(test_predictions):.2f}")
print(f"  最小值: {test_predictions.min():.2f}")
print(f"  最大值: {test_predictions.max():.2f}")
print(f"  标准差: {test_predictions.std():.2f}")

# ========================
# 11. 保存结果
# ========================
print("\n【11】保存结果...")

submission = pd.DataFrame({
    'SaleID': test_df['SaleID'],
    'price': test_predictions
})

output_file = f'submission_catboost_k5_optimized_A.csv'
submission.to_csv(output_file, index=False)
print(f"✅ 提交文件已保存: {output_file}")
print(f"   记录数: {len(submission)}")
print(f"   表头: SaleID, price")

# 保存模型
try:
    model.save_model(f'catboost_model_k5_optimized_A.cbm')
    print(f"✅ 模型已保存: catboost_model_k5_optimized_A.cbm")
except Exception as e:
    print(f"⚠️ 模型保存失败: {e}")

# ========================
# 12. 总结报告
# ========================
print("\n" + "=" * 80)
print("📊 方案A优化总结报告")
print("=" * 80)

print(f"\n✅ 完成的优化:")
print(f"  1. 移除冗余特征: {len(features_to_drop)} 个")
for feat in features_to_drop:
    print(f"     • {feat}")
print(f"  2. 添加新特征: {new_features_count} 个")
print(f"     • power_squared (power²)")
print(f"     • carAge_squared (carAge²)")
print(f"     • kilometer_squared (kilometer²)")
print(f"     • power_x_carAge (power × carAge)")
print(f"     • power_x_kilometer (power × kilometer)")
print(f"  3. 特征总数变化: {train_df.shape[1]-2} → {train_X.shape[1]}")

print(f"\n📈 性能对比:")
print(f"  优化前 (CatBoost K5):")
print(f"    RMSE: 1226.33, MAE: 520.57, R²: 0.9726")
print(f"  优化后 (方案A):")
print(f"    RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

if rmse < 1226.33:
    improvement = 1226.33 - rmse
    print(f"\n🎉 成功! RMSE降低了 {improvement:.2f} 点")
    if improvement >= 10:
        print(f"✅ 达到预期目标（降低10-30点）")
    if rmse < 1200:
        print(f"🏆 突破1200大关！")
else:
    print(f"\n⚠️ 未达到预期，需要进一步调优")

print(f"\n💡 下一步建议:")
print(f"  1. 如果效果良好，尝试方案B（激进优化）")
print(f"  2. 如果效果不佳，检查新特征的质量")
print(f"  3. 尝试不同的超参数组合")
print(f"  4. 考虑对偏态特征进行log变换")

print("\n" + "=" * 80)
print("✅ 方案A优化完成！")
print("=" * 80)
