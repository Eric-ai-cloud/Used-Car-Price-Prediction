import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 XGBoost 二手车价格预测建模")
print("=" * 80)

# ========================
# 1. 数据加载
# ========================
print("\n【1】数据加载...")
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# ========================
# 2. 特征工程
# ========================
print("\n【2】特征工程...")

def feature_engineering(df):
    """特征工程函数"""
    
    # 处理日期变量 - 提取年月日信息
    df['regDate'] = df['regDate'].astype(str)
    df['creatDate'] = df['creatDate'].astype(str)
    
    # 提取注册日期信息
    df['regYear'] = df['regDate'].str[:4].astype(int)
    df['regMonth'] = df['regDate'].str[4:6].astype(int)
    df['regDay'] = df['regDate'].str[6:8].astype(int)
    
    # 提取交易日期信息
    df['creatYear'] = df['creatDate'].str[:4].astype(int)
    df['creatMonth'] = df['creatDate'].str[4:6].astype(int)
    df['creatDay'] = df['creatDate'].str[6:8].astype(int)
    
    # 计算车龄（年）
    df['carAge'] = df['creatYear'] - df['regYear']
    
    # 计算车龄（月）- 更精确
    df['carAgeMonth'] = (df['creatYear'] - df['regYear']) * 12 + (df['creatMonth'] - df['regMonth'])
    
    # 删除原始日期列
    df.drop(['regDate', 'creatDate'], axis=1, inplace=True)
    
    return df

# 对训练集和测试集进行特征工程
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

print(f"特征工程完成！新增特征: regYear, regMonth, regDay, creatYear, creatMonth, creatDay, carAge, carAgeMonth")

# ========================
# 3. 数据清洗
# ========================
print("\n【3】数据清洗...")

# 处理异常值
print(f"删除前训练集大小: {train_df.shape}")

# 删除power异常值（>300马力）
train_df = train_df[train_df['power'] <= 300]

# 删除price异常值（<100元）
train_df = train_df[train_df['price'] >= 100]

print(f"删除后训练集大小: {train_df.shape}")
print(f"删除了 {150000 - len(train_df)} 条异常记录")

# ========================
# 4. 缺失值处理
# ========================
print("\n【4】缺失值处理...")

# 填充分类特征的缺失值为 -1（作为一个新类别）
categorical_cols = ['bodyType', 'fuelType', 'gearbox', 'model']
for col in categorical_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(-1)
        test_df[col] = test_df[col].fillna(-1)
        print(f"{col}: 训练集缺失值数={train_df[col].isnull().sum()}, 测试集缺失值数={test_df[col].isnull().sum()}")

# ========================
# 5. 准备特征和目标变量
# ========================
print("\n【5】准备特征和目标变量...")

# 确定要删除的列
drop_columns = ['SaleID', 'price', 'name', 'seller', 'offerType', 'notRepairedDamage']

# 过滤出存在的列
drop_columns = [col for col in drop_columns if col in train_df.columns]

# 特征列
feature_cols = [col for col in train_df.columns if col not in drop_columns]

print(f"特征数量: {len(feature_cols)}")
print(f"特征列表: {feature_cols}")

# 分离特征和目标变量
X = train_df[feature_cols]
y = train_df['price']

# 测试集特征
X_test = test_df[[col for col in feature_cols if col in test_df.columns]]

print(f"训练特征矩阵形状: {X.shape}")
print(f"测试特征矩阵形状: {X_test.shape}")

# ========================
# 6. 划分验证集
# ========================
print("\n【6】划分训练集和验证集...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")

# ========================
# 7. 训练XGBoost模型
# ========================
print("\n【7】训练XGBoost模型...")

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
# 8. 模型评估
# ========================
print("\n【8】模型评估...")

# 验证集预测
y_val_pred = model.predict(X_val)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)

print(f"验证集 RMSE: {rmse:.4f}")
print(f"验证集 MAE: {mae:.4f}")
print(f"验证集 R²: {1 - rmse**2 / np.var(y_val):.4f}")

# 特征重要性
print("\n【Top 20 重要特征】")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

# ========================
# 9. 测试集预测
# ========================
print("\n【9】测试集预测...")

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

# ========================
# 10. 保存预测结果
# ========================
print("\n【10】保存预测结果...")

# 创建提交文件
submission = pd.DataFrame({
    'SaleID': test_df['SaleID'],
    'price': test_predictions
})

# 保存为CSV
submission.to_csv('submission_xgboost.csv', index=False)

print(f"✅ 预测结果已保存到: submission_xgboost.csv")
print(f"提交文件形状: {submission.shape}")
print(f"\n前5行预测结果:")
print(submission.head())

# ========================
# 11. 保存模型
# ========================
print("\n【11】保存模型...")

try:
    model.save_model('xgboost_price_model.json')
    print("✅ 模型已保存到: xgboost_price_model.json")
except Exception as e:
    print(f"⚠️ 模型保存失败: {e}")
    print("💡 使用备用方法保存...")
    # 使用pickle保存
    import pickle
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 模型已使用pickle保存到: xgboost_model.pkl")

# ========================
# 总结
# ========================
print("\n" + "=" * 80)
print("📊 建模总结")
print("=" * 80)
print(f"""
✅ 数据处理:
   - 训练集: {len(train_df)} 条记录
   - 测试集: {len(test_df)} 条记录
   - 删除异常值: {150000 - len(train_df)} 条

✅ 特征工程:
   - 从日期提取: regYear, regMonth, regDay, creatYear, creatMonth, creatDay
   - 计算车龄: carAge (年), carAgeMonth (月)
   - 总特征数: {len(feature_cols)}

✅ 模型性能:
   - 验证集 RMSE: {rmse:.4f}
   - 验证集 MAE: {mae:.4f}

✅ 输出文件:
   - 预测结果: submission_xgboost.csv
   - 模型文件: xgboost_price_model.json
""")

print("=" * 80)
print("🎉 建模完成！可以查看 submission_xgboost.csv 文件了")
print("=" * 80)
