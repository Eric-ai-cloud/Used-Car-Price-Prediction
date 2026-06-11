"""
CatBoost建模 - K5 Final数据
特点：
- 使用动态学习率（略低）
- 5000次迭代
- 自动处理类别特征
- 强大的正则化能力
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 CatBoost 二手车价格预测建模 - K5 Final数据")
print("=" * 80)

# ========================
# 配置参数
# ========================
CLUSTER_VERSION = 'k5'  # 可选: 'k5' 或 'k6'
USE_CLUSTER_LABEL = True  # 是否使用聚类标签作为特征
ITERATIONS = 5000  # 迭代次数
LEARNING_RATE = 0.02 # 较低的学习率

print(f"\n📋 配置:")
print(f"  • 聚类版本: {CLUSTER_VERSION.upper()}")
print(f"  • 使用聚类标签: {'是' if USE_CLUSTER_LABEL else '否'}")
print(f"  • 迭代次数: {ITERATIONS}")
print(f"  • 学习率: {LEARNING_RATE} (动态调整)")

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
print(f"训练集缺失值总数: {train_df.isnull().sum().sum()}")
print(f"测试集缺失值总数: {test_df.isnull().sum().sum()}")

# ========================
# 2. 数据质量检查
# ========================
print("\n【2】数据质量检查...")

# 检查Power范围
print(f"Power范围: [{train_df['power'].min():.2f}, {train_df['power'].max():.2f}]")
if train_df['power'].max() <= 600 and train_df['power'].min() >= 0:
    print("✅ Power值在合理范围内 [0, 600]")
else:
    print("⚠️ 警告: Power值超出合理范围！")

# 检查衍生特征
derived_features = ['power_per_km', 'km_per_year', 'power_age_ratio']
for col in derived_features:
    if col in train_df.columns:
        has_negative = (train_df[col] < 0).any()
        has_nan = train_df[col].isnull().any()
        if has_negative or has_nan:
            print(f"⚠️ {col} 存在异常值")
        else:
            print(f"✅ {col} 正常")

# ========================
# 3. 准备特征和目标变量
# ========================
print("\n【3】准备特征和目标变量...")

# 确定要删除的列
drop_columns = ['SaleID', 'price']

# 如果不使用聚类标签，也删除它
if not USE_CLUSTER_LABEL:
    drop_columns.append('cluster_label')

# 过滤出存在的列
drop_columns = [col for col in drop_columns if col in train_df.columns]
print(f"删除的列: {drop_columns}")

# 分离特征和目标变量
target = 'price'
y_train = train_df[target]
train_X = train_df.drop(drop_columns, axis=1)

# 确保测试集有相同的列
test_X = test_df.drop([col for col in drop_columns if col in test_df.columns], axis=1)

# 对齐训练集和测试集的列
common_cols = list(set(train_X.columns) & set(test_X.columns))
train_X = train_X[common_cols]
test_X = test_X[common_cols]

print(f"特征数量: {train_X.shape[1]}")

# 识别类别特征并转换为合适的类型
cat_features = []
for col in train_X.columns:
    # 检查是否为类别特征（低基数或object类型）
    if train_X[col].dtype == 'object':
        cat_features.append(col)
    elif train_X[col].nunique() < 10 and train_X[col].dtype in ['int64', 'float64']:
        # 将低基数的数值型特征转换为整数
        train_X[col] = train_X[col].astype(int)
        test_X[col] = test_X[col].astype(int)
        cat_features.append(col)

print(f"类别特征数量: {len(cat_features)}")
if cat_features:
    print(f"类别特征列表: {cat_features[:10]}{'...' if len(cat_features) > 10 else ''}")
    
# 验证类别特征的类型
print(f"\n🔍 验证类别特征类型:")
for col in cat_features[:5]:
    print(f"  {col}: dtype={train_X[col].dtype}, unique={train_X[col].nunique()}")

# ========================
# 4. 划分验证集
# ========================
print("\n【4】划分训练集和验证集...")

X_train, X_val, y_train_split, y_val = train_test_split(
    train_X, y_train, 
    test_size=0.2, 
    random_state=42
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")

# ========================
# 5. CatBoost模型配置
# ========================
print("\n【5】配置CatBoost模型...")

# 创建CatBoost池
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

# 模型参数
params = {
    'iterations': ITERATIONS,              # 迭代次数
    'learning_rate': LEARNING_RATE,        # 较低的学习率
    'depth': 8,                            # 树深度
    'l2_leaf_reg': 3,                      # L2正则化
    'bagging_temperature': 0.5,            # Bagging温度
    'random_strength': 0.5,                # 随机强度
    'border_count': 254,                   # 分位数边界数
    'loss_function': 'RMSE',               # 损失函数
    'eval_metric': 'RMSE',                 # 评估指标
    'early_stopping_rounds': 100,          # 早停轮数
    'verbose': 100,                        # 每100轮打印一次
    'random_seed': 42,                     # 随机种子
    'thread_count': -1,                    # 使用所有CPU核心
    'use_best_model': True,                # 使用最佳模型
}

print("模型参数:")
for key, value in params.items():
    if key not in ['verbose']:
        print(f"  {key}: {value}")

# ========================
# 6. 模型训练
# ========================
print("\n【6】开始训练...")

model = CatBoostRegressor(**params)

model.fit(
    train_pool,
    eval_set=val_pool,
)

print(f"\n✅ 训练完成！")
print(f"最佳迭代轮数: {model.best_iteration_}")
print(f"最佳验证集RMSE: {model.best_score_['validation']['RMSE']:.2f}")

# ========================
# 7. 模型评估
# ========================
print("\n【7】模型评估...")

# 验证集预测
y_val_pred = model.predict(X_val)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"\n📊 验证集性能:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  R²:   {r2:.4f}")

# 与之前模型对比
print(f"\n📈 与之前模型对比:")
print(f"  单一模型 (K5 Final):")
print(f"    RMSE: 1267.30")
print(f"    MAE:  530.40")
print(f"    R²:   0.9707")
print(f"  分层建模 v2:")
print(f"    RMSE: 1250.58")
print(f"    MAE:  519.86")
print(f"    R²:   0.9205")
print(f"  当前CatBoost模型:")
print(f"    RMSE: {rmse:.2f} ({'↑改善' if rmse < 1250.58 else '↓待优化'})")
print(f"    MAE:  {mae:.2f} ({'↑改善' if mae < 519.86 else '↓待优化'})")
print(f"    R²:   {r2:.4f} ({'↑改善' if r2 > 0.9205 else '↓待优化'})")

# ========================
# 8. 特征重要性分析
# ========================
print("\n【8】特征重要性分析...")

# 获取特征重要性
importance = model.get_feature_importance()
feature_names = X_train.columns

# 创建DataFrame并排序
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

# 检查衍生特征的重要性
print(f"\n🔍 衍生特征重要性:")
for col in derived_features:
    if col in importance_df['feature'].values:
        imp = importance_df[importance_df['feature'] == col]['importance'].values[0]
        rank = importance_df[importance_df['feature'] == col].index[0] + 1
        print(f"  {col:<25} 重要性={imp:.4f}, 排名={rank}")

# ========================
# 9. 测试集预测
# ========================
print("\n【9】测试集预测...")

test_predictions = model.predict(test_X)

# 确保预测值非负
test_predictions = np.maximum(test_predictions, 0)

print(f"预测结果统计:")
print(f"  均值: {test_predictions.mean():.2f}")
print(f"  中位数: {np.median(test_predictions):.2f}")
print(f"  最小值: {test_predictions.min():.2f}")
print(f"  最大值: {test_predictions.max():.2f}")
print(f"  标准差: {test_predictions.std():.2f}")

# ========================
# 10. 保存结果
# ========================
print("\n【10】保存结果...")

# 创建提交文件
submission = pd.DataFrame({
    'SaleID': test_df['SaleID'],
    'price': test_predictions
})

# 保存CSV
output_file = f'submission_catboost_{CLUSTER_VERSION}.csv'
submission.to_csv(output_file, index=False)
print(f"✅ 提交文件已保存: {output_file}")
print(f"   记录数: {len(submission)}")
print(f"   表头: SaleID, price")

# 保存模型
try:
    model.save_model(f'catboost_model_{CLUSTER_VERSION}.cbm')
    print(f"✅ 模型已保存: catboost_model_{CLUSTER_VERSION}.cbm")
except Exception as e:
    print(f"⚠️ 模型保存失败: {e}")

# ========================
# 11. 总结报告
# ========================
print("\n" + "=" * 80)
print("📊 建模总结报告")
print("=" * 80)

print(f"\n✅ 完成的工作:")
print(f"  1. 使用Final数据（Power截断 + 衍生特征重算）")
print(f"  2. 聚类版本: {CLUSTER_VERSION.upper()}")
print(f"  3. 使用聚类标签: {'是' if USE_CLUSTER_LABEL else '否'}")
print(f"  4. 特征数量: {train_X.shape[1]}")
print(f"  5. 类别特征数量: {len(cat_features)}")
print(f"  6. 迭代次数: {ITERATIONS}")
print(f"  7. 学习率: {LEARNING_RATE} (动态调整)")
print(f"  8. 最佳迭代: {model.best_iteration_}")
print(f"  9. 验证集RMSE: {rmse:.2f}")
print(f"  10. 验证集R²: {r2:.4f}")

print(f"\n💡 CatBoost的优势:")
print(f"  • 自动处理类别特征（无需One-Hot编码）")
print(f"  • 强大的正则化能力（防止过拟合）")
print(f"  • 支持有序 boosting（减少预测偏移）")
print(f"  • 内置早停机制")
print(f"  • GPU加速支持")

print(f"\n📁 生成的文件:")
print(f"  • {output_file} - 测试集预测结果")
print(f"  • catboost_model_{CLUSTER_VERSION}.cbm - 训练好的模型")

print(f"\n🎯 下一步建议:")
print(f"  1. 尝试不同的学习率（0.01, 0.03, 0.05）")
print(f"  2. 调整树深度（6-10）")
print(f"  3. 尝试GPU加速（如果有GPU）")
print(f"  4. 与XGBoost模型融合")
print(f"  5. 实施K-Fold交叉验证")

print("\n" + "=" * 80)
print("✅ CatBoost建模完成！")
print("=" * 80)
