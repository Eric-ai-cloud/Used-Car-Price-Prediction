"""
XGBoost建模 - 基于Final数据（Power截断 + 衍生特征重算）
支持K5和K6两种聚类版本
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ========================
# 配置参数
# ========================
CLUSTER_VERSION = 'k6'  # 可选: 'k5' 或 'k6'
USE_CLUSTER_LABEL = True  # 是否使用聚类标签作为特征

print("=" * 80)
print(f"🚀 XGBoost 二手车价格预测建模 - Final数据 ({CLUSTER_VERSION.upper()})")
print("=" * 80)
print(f"\n📋 数据特点:")
print(f"  • Power已截断到[0, 600]范围")
print(f"  • 衍生特征已重新计算")
print(f"  • 无缺失值")
print(f"  • 包含聚类标签: {CLUSTER_VERSION}")
print(f"  • 使用聚类标签: {'是' if USE_CLUSTER_LABEL else '否'}")

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
print(f"特征列表: {list(train_X.columns)}")

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
# 5. XGBoost模型配置
# ========================
print("\n【5】配置XGBoost模型...")

params = {
    'objective': 'reg:squarederror',     # 回归任务
    'eval_metric': 'rmse',               # RMSE评估
    'max_depth': 8,                      # 树最大深度
    'learning_rate': 0.05,               # 学习率
    'n_estimators': 1000,                # 树的数量
    'min_child_weight': 5,               # 最小子节点权重
    'subsample': 0.8,                    # 样本采样比例
    'colsample_bytree': 0.8,             # 特征采样比例
    'gamma': 0.1,                        # L2正则化
    'lambda': 1.0,                       # L1正则化
    'random_state': 42,                  # 随机种子
    'n_jobs': -1                         # 使用所有CPU核心
}

print("模型参数:")
for key, value in params.items():
    print(f"  {key}: {value}")

# ========================
# 6. 模型训练
# ========================
print("\n【6】开始训练...")

model = xgb.XGBRegressor(**params)

model.fit(
    X_train, y_train_split,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=100
)

print(f"\n✅ 训练完成！")
print(f"实际迭代轮数: {model.best_iteration}")

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
print(f"  之前模型 (xgboost_modeling.py):")
print(f"    RMSE: 1131.22")
print(f"    MAE:  509.74")
print(f"    R²:   0.9699")
print(f"  当前模型:")
print(f"    RMSE: {rmse:.2f} ({'↑改善' if rmse < 1131.22 else '↓变差'})")
print(f"    MAE:  {mae:.2f} ({'↑改善' if mae < 509.74 else '↓变差'})")
print(f"    R²:   {r2:.4f} ({'↑改善' if r2 > 0.9699 else '↓变差'})")

# ========================
# 8. 特征重要性分析
# ========================
print("\n【8】特征重要性分析...")

# 获取特征重要性
importance = model.feature_importances_
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
    pct = row['importance'] * 100
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
output_file = f'submission_xgboost_final_{CLUSTER_VERSION}.csv'
submission.to_csv(output_file, index=False)
print(f"✅ 提交文件已保存: {output_file}")
print(f"   记录数: {len(submission)}")

# 保存模型
try:
    model.save_model(f'xgboost_model_final_{CLUSTER_VERSION}.json')
    print(f"✅ 模型已保存: xgboost_model_final_{CLUSTER_VERSION}.json")
except Exception as e:
    print(f"⚠️ JSON保存失败: {e}")
    try:
        import pickle
        with open(f'xgboost_model_final_{CLUSTER_VERSION}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ 模型已保存 (pickle): xgboost_model_final_{CLUSTER_VERSION}.pkl")
    except Exception as e2:
        print(f"❌ 模型保存失败: {e2}")

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
print(f"  5. 验证集RMSE: {rmse:.2f}")
print(f"  6. 验证集R²: {r2:.4f}")

print(f"\n💡 关键发现:")
top3_features = importance_df.head(3)['feature'].values
print(f"  • Top3重要特征: {', '.join(top3_features)}")

# 检查衍生特征是否在Top10
top10_features = importance_df.head(10)['feature'].values
derived_in_top10 = [f for f in derived_features if f in top10_features]
if derived_in_top10:
    print(f"  • 衍生特征进入Top10: {', '.join(derived_in_top10)}")
    print(f"  • 说明衍生特征重算带来了显著价值！")

print(f"\n📁 生成的文件:")
print(f"  • {output_file} - 测试集预测结果")
print(f"  • xgboost_model_final_{CLUSTER_VERSION}.json/pkl - 训练好的模型")

print(f"\n🎯 下一步建议:")
print(f"  1. 尝试K6版本，对比性能差异")
print(f"  2. 调整超参数（max_depth, learning_rate等）")
print(f"  3. 尝试LightGBM、CatBoost等其他算法")
print(f"  4. 实施K-Fold交叉验证")
print(f"  5. 模型融合（Stacking/Blending）")

print("\n" + "=" * 80)
print("✅ 建模完成！")
print("=" * 80)
