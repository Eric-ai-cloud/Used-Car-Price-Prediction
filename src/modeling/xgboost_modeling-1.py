import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 XGBoost 二手车价格预测建模（精填数据版）")
print("=" * 80)

# ========================
# 1. 数据加载
# ========================
print("\n【1】加载精填后的数据...")
train_df = pd.read_csv('train_refined.csv')
test_df = pd.read_csv('test_refined.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 验证缺失值
train_missing = train_df.isnull().sum().sum()
test_missing = test_df.isnull().sum().sum()
print(f"训练集缺失值: {train_missing}")
print(f"测试集缺失值: {test_missing}")

if train_missing > 0 or test_missing > 0:
    print("⚠️  警告：数据仍存在缺失值！")
else:
    print("✅ 数据完整，无缺失值")

# ========================
# 2. 准备特征和目标变量
# ========================
print("\n【2】准备特征和目标变量...")

# 确定要删除的列
drop_columns = ['SaleID', 'price', 'name']

# 过滤出存在的列
drop_columns = [col for col in drop_columns if col in train_df.columns]

# 特征列（所有数值型特征）
feature_cols = [col for col in train_df.columns 
                if col not in drop_columns 
                and train_df[col].dtype in ['int64', 'float64']]

print(f"特征数量: {len(feature_cols)}")
print(f"特征列表（前20个）: {feature_cols[:20]}")

# 分离特征和目标变量
X = train_df[feature_cols]
y = train_df['price']

# 测试集特征（确保特征一致）
common_features = [col for col in feature_cols if col in test_df.columns]
X_test = test_df[common_features]

print(f"训练特征矩阵形状: {X.shape}")
print(f"测试特征矩阵形状: {X_test.shape}")

# 检查特征一致性
if len(feature_cols) != len(common_features):
    missing_in_test = set(feature_cols) - set(common_features)
    print(f"⚠️  测试集缺少的特征: {missing_in_test}")
    # 只使用共同特征
    X = train_df[common_features]
    feature_cols = common_features
    print(f"✅ 已调整为共同特征: {len(feature_cols)}个")

# ========================
# 3. 划分验证集
# ========================
print("\n【3】划分训练集和验证集...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")

# ========================
# 4. 训练XGBoost模型
# ========================
print("\n【4】训练XGBoost模型...")

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

# ========================
# 5. 模型评估
# ========================
print("\n【5】模型评估...")

# 验证集预测
y_val_pred = model.predict(X_val)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
r2 = 1 - rmse**2 / np.var(y_val)

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

# ========================
# 6. 测试集预测
# ========================
print("\n【6】测试集预测...")

# 对测试集进行预测
test_predictions = model.predict(X_test)

# 确保预测值为正
test_predictions = np.maximum(test_predictions, 0)

print(f"测试集预测完成！")
print(f"预测价格统计:")
print(f"  均值: {test_predictions.mean():.2f}")
print(f"  中位数: {np.median(test_predictions):.2f}")
print(f"  最小值: {test_predictions.min():.2f}")
print(f"  最大值: {test_predictions.max():.2f}")
print(f"  标准差: {test_predictions.std():.2f}")

# ========================
# 7. 保存预测结果
# ========================
print("\n【7】保存预测结果...")

# 创建提交文件
submission = pd.DataFrame({
    'SaleID': test_df['SaleID'],
    'price': test_predictions
})

# 保存为CSV
submission.to_csv('submission_xgboost_refined.csv', index=False)

print(f"✅ 预测结果已保存到: submission_xgboost_refined.csv")
print(f"提交文件形状: {submission.shape}")
print(f"\n前5行预测结果:")
print(submission.head())

# ========================
# 8. 保存模型
# ========================
print("\n【8】保存模型...")

try:
    model.save_model('xgboost_price_model_refined.json')
    print("✅ 模型已保存到: xgboost_price_model_refined.json")
except Exception as e:
    print(f"⚠️ 模型保存失败: {e}")
    print("💡 使用备用方法保存...")
    # 使用pickle保存
    import pickle
    with open('xgboost_model_refined.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 模型已使用pickle保存到: xgboost_model_refined.pkl")

# ========================
# 9. 总结
# ========================
print("\n" + "=" * 80)
print("📊 建模总结（精填数据版）")
print("=" * 80)
print(f"""
✅ 数据处理:
   - 数据来源: train_refined.csv / test_refined.csv（簇内精填后）
   - 训练集: {len(train_df)} 条记录
   - 测试集: {len(test_df)} 条记录
   - 缺失值: 训练集={train_missing}, 测试集={test_missing}

✅ 特征工程:
   - 包含车龄、聚类标签等衍生特征
   - 总特征数: {len(feature_cols)}
   - 包含cluster_label作为强特征

✅ 模型性能:
   - 验证集 RMSE: {rmse:.4f}
   - 验证集 MAE: {mae:.4f}
   - 验证集 R²: {r2:.4f}

✅ 输出文件:
   - 预测结果: submission_xgboost_refined.csv
   - 模型文件: xgboost_price_model_refined.json

💡 优势:
   - 使用精填后的高质量数据
   - 包含聚类标签特征
   - 缺失值处理更精准
""")

print("=" * 80)
print("🎉 建模完成！可以查看 submission_xgboost_refined.csv 文件了")
print("=" * 80)
