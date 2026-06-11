"""
CatBoost超参数调优 - 使用Optuna搜索最优参数
在方案A优化版基础上，通过Optuna自动搜索最佳超参数组合

优化目标: 最小化验证集RMSE
搜索空间: learning_rate, depth, l2_leaf_reg, random_strength, bagging_temperature等
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔬 CatBoost超参数调优 - Optuna自动搜索")
print("=" * 80)

# ========================
# 配置参数
# ========================
CLUSTER_VERSION = 'k5'
N_TRIALS = 50  # Optuna试验次数（建议30-100）
RANDOM_SEED = 42

print(f"\n📋 配置:")
print(f"  • 聚类版本: {CLUSTER_VERSION.upper()}")
print(f"  • Optuna试验次数: {N_TRIALS}")
print(f"  • 随机种子: {RANDOM_SEED}")

# ========================
# 1. 数据加载与预处理
# ========================
print("\n【1】数据加载与预处理...")

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

# 1.1 合并极小簇
if 'cluster_label' in train_df.columns:
    print("\n  1.1 合并极小簇...")
    cluster_counts = train_df['cluster_label'].value_counts()
    tiny_clusters = cluster_counts[cluster_counts < 100].index.tolist()
    
    for tiny_cluster in tiny_clusters:
        # 计算与其他簇的距离
        tiny_data = train_df[train_df['cluster_label'] == tiny_cluster]
        tiny_means = tiny_data[['power', 'kilometer', 'carAge', 'price']].mean()
        
        other_clusters = [c for c in cluster_counts.index if c != tiny_cluster and cluster_counts[c] >= 100]
        distances = {}
        
        for other_cluster in other_clusters:
            other_data = train_df[train_df['cluster_label'] == other_cluster]
            other_means = other_data[['power', 'kilometer', 'carAge', 'price']].mean()
            
            features = ['power', 'kilometer', 'carAge', 'price']
            distance = 0
            for feat in features:
                std = train_df[train_df['cluster_label'] == other_cluster][feat].std()
                if std > 0:
                    diff = (tiny_means[feat] - other_means[feat]) / std
                    distance += diff ** 2
            
            distances[other_cluster] = np.sqrt(distance)
        
        most_similar = min(distances, key=distances.get)
        train_df.loc[train_df['cluster_label'] == tiny_cluster, 'cluster_label'] = most_similar
        if 'cluster_label' in test_df.columns:
            test_df.loc[test_df['cluster_label'] == tiny_cluster, 'cluster_label'] = most_similar
    
    print(f"    ✅ 已合并{len(tiny_clusters)}个极小簇")

# 1.2 移除冗余特征
print("\n  1.2 移除冗余特征...")
features_to_drop = ['carAgeMonth', 'regYear', 'isYearStart', 'creatYear']
features_to_drop = [f for f in features_to_drop if f in train_df.columns]
train_df = train_df.drop(columns=features_to_drop)
test_df = test_df.drop(columns=[f for f in features_to_drop if f in test_df.columns])
print(f"    ✅ 已移除{len(features_to_drop)}个冗余特征")

# 1.3 添加新特征
print("\n  1.3 添加新特征...")

# 多项式特征
if 'power' in train_df.columns:
    train_df['power_squared'] = train_df['power'] ** 2
    test_df['power_squared'] = test_df['power'] ** 2

if 'carAge' in train_df.columns:
    train_df['carAge_squared'] = train_df['carAge'] ** 2
    test_df['carAge_squared'] = test_df['carAge'] ** 2

if 'kilometer' in train_df.columns:
    train_df['kilometer_squared'] = train_df['kilometer'] ** 2
    test_df['kilometer_squared'] = test_df['kilometer'] ** 2

# 交互特征
if 'power' in train_df.columns and 'carAge' in train_df.columns:
    train_df['power_x_carAge'] = train_df['power'] * train_df['carAge']
    test_df['power_x_carAge'] = test_df['power'] * test_df['carAge']

if 'power' in train_df.columns and 'kilometer' in train_df.columns:
    train_df['power_x_kilometer'] = train_df['power'] * train_df['kilometer']
    test_df['power_x_kilometer'] = test_df['power'] * test_df['kilometer']

print(f"    ✅ 已添加5个新特征")

# 1.4 添加v_14异常值标识特征 ⭐ 新增
print("\n  1.4 添加v_14异常值标识特征...")
Q1 = train_df['v_14'].quantile(0.25)
Q3 = train_df['v_14'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train_df['is_v14_outlier'] = ((train_df['v_14'] < lower_bound) | (train_df['v_14'] > upper_bound)).astype(int)
test_df['is_v14_outlier'] = ((test_df['v_14'] < lower_bound) | (test_df['v_14'] > upper_bound)).astype(int)

outlier_count = train_df['is_v14_outlier'].sum()
outlier_pct = outlier_count / len(train_df) * 100
print(f"    ✅ 已创建 is_v14_outlier 特征")
print(f"       异常值数量: {outlier_count:,} ({outlier_pct:.2f}%)")

# 1.5 准备训练数据
print("\n  1.5 准备训练数据...")
target_col = 'price'
id_col = 'SaleID'
exclude_cols = [target_col, id_col]

feature_cols = [col for col in train_df.columns 
                if col not in exclude_cols 
                and train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']]

X = train_df[feature_cols]
y = train_df[target_col]
X_test = test_df[feature_cols]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

print(f"    特征数量: {len(feature_cols)}")
print(f"    训练集: {X_train.shape}")
print(f"    验证集: {X_val.shape}")

# 识别类别特征
cat_features = [col for col in feature_cols if train_df[col].dtype == 'object']
print(f"    类别特征: {len(cat_features)}个")

# ========================
# 2. Optuna超参数搜索
# ========================
print("\n【2】Optuna超参数搜索...")

def objective(trial):
    """Optuna优化目标函数"""
    
    # 定义超参数搜索空间
    params = {
        # 基础参数
        'iterations': trial.suggest_int('iterations', 3000, 8000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        
        # 正则化参数
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-2, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        
        # 采样参数
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        
        # 其他参数
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 1, 4),
    }
    
    # 创建模型
    model = CatBoostRegressor(
        **params,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=RANDOM_SEED,
        verbose=0,  # 静默模式
        early_stopping_rounds=100,
        use_best_model=True,
    )
    
    # 创建数据池
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # 训练模型
    model.fit(
        train_pool,
        eval_set=val_pool,
    )
    
    # 预测并计算RMSE
    y_val_pred = model.predict(val_pool)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    return rmse

# 创建Optuna研究
print(f"\n  开始Optuna搜索（{N_TRIALS}次试验）...")
print(f"  搜索空间维度: 10个超参数")
print(f"  目标: 最小化验证集RMSE\n")

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),  # TPE采样器
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)  # 剪枝策略
)

# 执行搜索
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ========================
# 3. 搜索结果分析
# ========================
print("\n【3】搜索结果分析...")

print(f"\n🏆 最优超参数配置:")
print(f"  最优RMSE: {study.best_value:.2f}")
print(f"\n  超参数详情:")
for param_name, param_value in study.best_params.items():
    print(f"    • {param_name}: {param_value}")

# 显示Top 10试验
print(f"\n📊 Top 10 试验结果:")
df_trials = study.trials_dataframe().sort_values('value')
top_trials = df_trials.head(10)

print(f"{'排名':<6} {'RMSE':<12} {'iterations':<12} {'learning_rate':<15} {'depth':<8} {'l2_leaf_reg':<12}")
print("-" * 80)
for idx, (_, row) in enumerate(top_trials.iterrows(), 1):
    print(f"{idx:<6} {row['value']:<12.2f} "
          f"{int(row['params_iterations']):<12} "
          f"{row['params_learning_rate']:<15.4f} "
          f"{int(row['params_depth']):<8} "
          f"{row['params_l2_leaf_reg']:<12.4f}")

# 可视化优化过程
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：优化历史
    ax1 = axes[0]
    ax1.plot([t.value for t in study.trials], marker='o', markersize=3, linewidth=1)
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Optimization History', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.2f}')
    ax1.legend()
    
    # 右图：参数重要性
    ax2 = axes[1]
    param_importance = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.close()  # 关闭自动显示的图
    
    # 手动绘制参数重要性
    importance_dict = {}
    for param_name in study.best_params.keys():
        if param_name != 'iterations':  # 排除iterations
            values = [t.params.get(param_name) for t in study.trials if t.params.get(param_name) is not None]
            scores = [t.value for t in study.trials if t.params.get(param_name) is not None]
            if len(values) > 1:
                # 简单的相关性作为重要性指标
                corr = abs(np.corrcoef(values, scores)[0, 1]) if len(set(values)) > 1 else 0
                importance_dict[param_name] = corr
    
    if importance_dict:
        sorted_params = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        param_names = [p[0] for p in sorted_params[:8]]
        param_values = [p[1] for p in sorted_params[:8]]
        
        ax2.barh(range(len(param_names)), param_values, color='steelblue', alpha=0.8)
        ax2.set_yticks(range(len(param_names)))
        ax2.set_yticklabels(param_names)
        ax2.set_xlabel('Importance (Correlation)', fontsize=12)
        ax2.set_title('Parameter Importances', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('optuna_optimization_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 优化过程图表已保存: optuna_optimization_results.png")
except Exception as e:
    print(f"\n⚠️  可视化失败: {e}")

# ========================
# 4. 使用最优参数重新训练最终模型
# ========================
print("\n【4】使用最优参数训练最终模型...")

best_params = study.best_params.copy()

# 固定一些参数
final_params = {
    'iterations': best_params.get('iterations', 5000),
    'learning_rate': best_params.get('learning_rate', 0.02),
    'depth': best_params.get('depth', 8),
    'l2_leaf_reg': best_params.get('l2_leaf_reg', 3.0),
    'random_strength': best_params.get('random_strength', 1.0),
    'bagging_temperature': best_params.get('bagging_temperature', 1.0),
    'subsample': best_params.get('subsample', 0.8),
    'colsample_bylevel': best_params.get('colsample_bylevel', 0.8),
    'min_data_in_leaf': best_params.get('min_data_in_leaf', 10),
    'max_ctr_complexity': best_params.get('max_ctr_complexity', 2),
    
    # 固定参数
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': RANDOM_SEED,
    'verbose': 200,
    'early_stopping_rounds': 100,
    'use_best_model': True,
}

print(f"\n最终模型配置:")
for key, value in final_params.items():
    print(f"  {key}: {value}")

# 在全量训练集上训练
full_train_pool = Pool(X, y, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

print(f"\n开始训练最终模型...")
final_model = CatBoostRegressor(**final_params)
final_model.fit(
    full_train_pool,
    eval_set=val_pool,
)

# ========================
# 5. 模型评估
# ========================
print("\n【5】模型评估...")

# 验证集性能
y_val_pred = final_model.predict(val_pool)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"\n验证集性能:")
print(f"  RMSE: {rmse_val:.2f}")
print(f"  MAE:  {mae_val:.2f}")
print(f"  R²:   {r2_val:.4f}")

# 训练集性能
y_train_pred = final_model.predict(full_train_pool)
rmse_train = np.sqrt(mean_squared_error(y, y_train_pred))
mae_train = mean_absolute_error(y, y_train_pred)
r2_train = r2_score(y, y_train_pred)

print(f"\n训练集性能:")
print(f"  RMSE: {rmse_train:.2f}")
print(f"  MAE:  {mae_train:.2f}")
print(f"  R²:   {r2_train:.4f}")

# ========================
# 6. 特征重要性分析
# ========================
print("\n【6】特征重要性分析...")

feature_importance = final_model.get_feature_importance()
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 20 重要特征:")
print(feature_importance_df.head(20).to_string(index=False))

# 检查is_v14_outlier的重要性
if 'is_v14_outlier' in feature_cols:
    v14_outlier_importance = feature_importance_df[feature_importance_df['feature'] == 'is_v14_outlier']['importance'].values[0]
    v14_outlier_rank = feature_importance_df[feature_importance_df['feature'] == 'is_v14_outlier'].index[0] + 1
    print(f"\n⭐ is_v14_outlier 特征表现:")
    print(f"   重要性: {v14_outlier_importance:.2f}%")
    print(f"   排名: #{v14_outlier_rank}")

# ========================
# 7. 生成预测结果
# ========================
print("\n【7】生成预测结果...")

test_pool = Pool(X_test, cat_features=cat_features)
y_test_pred = final_model.predict(test_pool)

# 创建提交文件
submission = pd.DataFrame({
    'SaleID': test_df[id_col],
    'price': y_test_pred
})

output_file = f'submission_catboost_{CLUSTER_VERSION}_optuna_tuned.csv'
submission.to_csv(output_file, index=False)
print(f"✅ 预测结果已保存: {output_file}")
print(f"预测样本数: {len(submission)}")
print(f"预测价格范围: [{y_test_pred.min():.2f}, {y_test_pred.max():.2f}]")

# 保存模型
model_file = f'catboost_model_{CLUSTER_VERSION}_optuna_tuned.cbm'
final_model.save_model(model_file)
print(f"✅ 模型已保存: {model_file}")

# 保存最优超参数
import json
optuna_params_file = f'optuna_best_params_{CLUSTER_VERSION}.json'
with open(optuna_params_file, 'w', encoding='utf-8') as f:
    json.dump({
        'best_params': study.best_params,
        'best_rmse': study.best_value,
        'n_trials': N_TRIALS,
        'feature_cols': feature_cols,
    }, f, indent=2, ensure_ascii=False)
print(f"✅ 最优超参数已保存: {optuna_params_file}")

# ========================
# 8. 总结报告
# ========================
print("\n" + "=" * 80)
print("📊 Optuna超参数调优总结报告")
print("=" * 80)

print(f"\n🔧 优化内容:")
print(f"  1. 合并极小簇: 是")
print(f"  2. 移除冗余特征: 4个")
print(f"  3. 添加新特征: 5个（多项式+交互）")
print(f"  4. 添加v_14异常值标识: is_v14_outlier ⭐")
print(f"  5. Optuna超参数搜索: {N_TRIALS}次试验")

print(f"\n📈 模型性能:")
print(f"  验证集 RMSE: {rmse_val:.2f}")
print(f"  验证集 MAE:  {mae_val:.2f}")
print(f"  验证集 R²:   {r2_val:.4f}")

# 与之前模型对比
baseline_rmse = 1215.51  # 方案A优化版
improvement = baseline_rmse - rmse_val
improvement_pct = improvement / baseline_rmse * 100

print(f"\n💡 性能对比（vs 方案A优化版）:")
print(f"  RMSE变化: {improvement:+.2f}点 ({improvement_pct:+.2f}%)")
if improvement > 0:
    print(f"  ✅ 性能提升！RMSE降低{improvement:.2f}点")
else:
    print(f"  ⚠️  性能略有下降，但超参数已优化，可进一步调整")

print(f"\n💾 生成文件:")
print(f"  • {output_file} - 预测结果")
print(f"  • {model_file} - 训练好的模型")
print(f"  • {optuna_params_file} - 最优超参数配置")
print(f"  • optuna_optimization_results.png - 优化过程可视化")

print(f"\n{'='*80}")
print("✅ Optuna超参数调优完成！")
print(f"{'='*80}")
