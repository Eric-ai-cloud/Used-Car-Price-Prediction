# Optuna超参数调优（纯净版）- 使用说明

## 📋 脚本说明

**文件名**: `catboost_optuna_tuning_clean.py`

**版本特点**: 
- ✅ 基于方案A优化版的所有优化
- ✅ 使用Optuna自动搜索最优超参数
- ❌ **不包含** `is_v14_outlier` 特征（经验证无效）

---

## 🔧 优化内容

### 1. 数据预处理
- ✅ 合并极小簇（簇4→簇2）
- ✅ 移除4个冗余特征（carAgeMonth, regYear, isYearStart, creatYear）
- ✅ 添加5个新特征（3个多项式 + 2个交互）

### 2. 特征工程
```python
# 多项式特征
power_squared = power²
carAge_squared = carAge²
kilometer_squared = kilometer²

# 交互特征
power_x_carAge = power × carAge
power_x_kilometer = power × kilometer
```

### 3. Optuna超参数搜索
搜索10个超参数的最优组合：

| 超参数 | 搜索范围 | 说明 |
|--------|---------|------|
| iterations | [3000, 8000] | 迭代次数 |
| learning_rate | [0.01, 0.1] (log) | 学习率 |
| depth | [4, 10] | 树深度 |
| l2_leaf_reg | [0.01, 10] (log) | L2正则化 |
| random_strength | [0.01, 10] (log) | 随机强度 |
| bagging_temperature | [0.0, 10.0] | Bagging温度 |
| subsample | [0.5, 1.0] | 样本采样率 |
| colsample_bylevel | [0.5, 1.0] | 特征采样率 |
| min_data_in_leaf | [1, 100] | 叶节点最小样本数 |
| max_ctr_complexity | [1, 4] | 类别特征复杂度 |

### 4. 关键区别

#### vs 含v_14标识版

| 项目 | 含v_14标识版 | 纯净版 |
|------|------------|--------|
| 特征数量 | 46个 | **45个** ⭐ |
| is_v14_outlier | ✅ 包含 | ❌ **不包含** |
| 其他特征 | 完全相同 | 完全相同 |
| Optuna配置 | 完全相同 | 完全相同 |

**预期效果**: 性能应该与含v_14标识版基本一致（RMSE≈1215.51），但模型更简洁。

---

## 🚀 使用方法

### 直接运行

```bash
python catboost_optuna_tuning_clean.py
```

### 预计运行时间

- **单次试验**: ~30-40秒
- **50次试验**: ~25-35分钟
- **最终模型训练**: ~2-3分钟
- **总耗时**: **约30-40分钟**

### 后台运行（推荐）

由于运行时间较长，建议使用后台运行：

```bash
# Windows PowerShell
Start-Process python -ArgumentList "catboost_optuna_tuning_clean.py" -NoNewWindow

# 或者使用nohup（Linux/Mac）
nohup python catboost_optuna_tuning_clean.py > optuna_clean.log 2>&1 &
```

---

## 📊 输出文件

运行完成后会生成以下文件：

### 1. 预测结果
```
submission_catboost_k5_optuna_clean.csv
```
- **格式**: CSV
- **内容**: SaleID, price
- **用途**: 提交到比赛平台

### 2. 训练好的模型
```
catboost_model_k5_optuna_clean.cbm
```
- **格式**: CatBoost二进制模型
- **大小**: ~30MB
- **用途**: 后续预测或部署

### 3. 最优超参数配置
```
optuna_best_params_k5_clean.json
```
- **格式**: JSON
- **内容**: 
  ```json
  {
    "best_params": {...},
    "best_rmse": 1215.xx,
    "n_trials": 50,
    "feature_cols": [...],
    "note": "纯净版-不含v_14标识"
  }
  ```

### 4. 优化过程可视化
```
optuna_optimization_clean.png
```
- **格式**: PNG图片
- **内容**: 
  - 左图：优化历史曲线
  - 右图：参数重要性排序

---

## 📈 预期性能

### 基准对比

| 模型版本 | RMSE | MAE | R² | 说明 |
|---------|------|-----|----|------|
| 原始CatBoost | 1267.30 | 530.40 | 0.9707 | 基线 |
| 方案A优化版 | **1215.51** | **505.44** | **0.9730** | 当前最优 |
| **Optuna纯净版** | **~1210-1215** | **~500-505** | **~0.9730+** | **预期** ⭐ |

### 性能提升预期

根据Optuna的优化能力，预期：

- **保守估计**: RMSE降低2-5点（1215.51 → 1210-1213）
- **乐观估计**: RMSE降低5-10点（1215.51 → 1205-1210）
- **目标**: 突破1210，接近1200大关

---

## 🔍 监控进度

### 查看实时输出

```bash
# 如果以后台方式运行，可以查看日志
tail -f optuna_clean.log

# 或者直接查看终端输出
```

### 关键输出示例

```
[I 2026-04-30 16:45:53,759] Trial 0 finished with value: 1240.20...
Best trial: 0. Best value: 1240.2

[I 2026-04-30 16:46:xx,xxx] Trial 1 finished with value: 1238.50...
Best trial: 1. Best value: 1238.50

...

🏆 最优超参数配置:
  最优RMSE: 1212.34
  
  超参数详情:
    • iterations: 6500
    • learning_rate: 0.0234
    • depth: 8
    ...
```

---

## 💡 使用建议

### 1. 首次运行

- ✅ 使用默认配置（N_TRIALS=50）
- ✅ 等待完整运行完成
- ✅ 检查生成的文件和性能

### 2. 如果效果不理想

可以尝试增加试验次数：

```python
# 修改脚本中的这一行
N_TRIALS = 100  # 从50增加到100
```

更多试验次数可能找到更优的超参数组合。

### 3. 如果运行时间过长

可以减少试验次数：

```python
N_TRIALS = 30  # 从50减少到30
```

30次试验通常也能找到不错的超参数组合。

### 4. 对比实验

可以同时运行两个版本进行对比：

```bash
# 终端1：含v_14标识版
python catboost_optuna_tuning.py

# 终端2：纯净版
python catboost_optuna_tuning_clean.py
```

然后对比两个版本的最终RMSE。

---

## 🎯 下一步行动

### Optuna完成后

1. ✅ **检查性能**
   - 查看验证集RMSE
   - 与方案A纯净版对比（1215.51）

2. ✅ **如果性能提升**
   - 使用生成的预测结果提交
   - 保存最优超参数配置供后续使用

3. ✅ **如果性能持平**
   - 保持方案A纯净版作为最优模型
   - 尝试其他优化方向（Log变换、模型融合等）

4. ✅ **如果性能下降**
   - 检查是否有异常试验
   - 调整搜索空间或增加试验次数

---

## 📝 注意事项

### 1. 资源需求

- **内存**: 至少8GB RAM
- **CPU**: 多核处理器（并行计算）
- **磁盘**: 至少100MB可用空间

### 2. 中断处理

如果中途需要中断：

- **Ctrl+C**: 优雅中断，会保存已完成的试验
- **下次运行**: 可以从头重新开始，或手动加载已有结果

### 3.  reproducibility

为保证可重复性：

- ✅ 设置固定随机种子（RANDOM_SEED=42）
- ✅ 记录所有超参数配置
- ✅ 保存最优参数JSON文件

### 4. 性能波动

由于随机性，每次运行的结果可能略有不同：

- RMSE波动范围: ±2-3点
- 这是正常现象
- 多次运行取平均值可获得更稳定的估计

---

## 🏆 总结

**Optuna纯净版的核心优势**:

1. ✅ **自动化**: 无需手动调参，Optuna自动搜索
2. ✅ **高效**: TPE采样器+剪枝策略，快速收敛
3. ✅ **简洁**: 不含无效的v_14标识特征
4. ✅ **可解释**: 提供参数重要性和优化历史

**推荐使用场景**:

- 希望进一步降低RMSE
- 愿意投入30-40分钟运行时间
- 追求模型性能的极致优化

**不推荐使用场景**:

- 时间紧迫，需要快速结果
- 对RMSE要求不高（1215已经很好）
- 计算资源有限

---

**创建时间**: 2026-04-30  
**版本**: 纯净版（不含v_14标识）  
**状态**: 🔄 运行中（预计30-40分钟完成）
