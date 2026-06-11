# 🚗 二手车价格预测竞赛项目

这是一个基于机器学习的二手车价格预测项目，旨在通过数据分析和建模预测二手车的市场交易价格。

## 📋 项目概述

本项目是针对二手车价格预测竞赛的解决方案，通过完整的机器学习流水线实现了数据预处理、特征工程、聚类分析和模型优化，最终取得了优异的预测性能。

## 🎯 项目目标

- 准确预测二手车的交易价格（price字段）
- 探索数据特征与价格之间的关系
- 构建稳健的机器学习模型
- 优化模型性能（降低RMSE，提高R²）

## 📊 数据集

原始数据集包含两个CSV文件：
- `used_car_train_20200313.csv`：训练集（150,000条记录）
- `used_car_testB_20200421.csv`：测试集（50,000条记录）

数据集包含以下类型的特征：
- **车辆基本信息**：brand, model, fuelType, gearbox等
- **车辆技术参数**：power, kilometer, notRepairedDamage等
- **时间信息**：regDate（注册日期）, creatDate（交易日期）
- **匿名特征**：v_0到v_14（经过脱敏处理的匿名特征）

## 🏗️ 项目架构

```
项目结构：
├── src/                         # 源代码目录
│   ├── data_analysis/           # 数据分析脚本（EDA、数据质量检查）
│   ├── preprocessing/           # 数据预处理脚本（缺失值填充、异常值处理）
│   ├── feature_engineering/     # 特征工程脚本（特征创建、选择、优化）
│   ├── modeling/                # 建模脚本（XGBoost/CatBoost训练、Optuna调优）
│   ├── clustering/              # 聚类分析脚本（K-Means聚类、分簇建模）
│   └── evaluation/              # 模型评估脚本（性能对比、K5/K6方案对比）
├── reports/                     # 分析报告（EDA、建模、优化等报告文档）
├── configs/                     # 配置文件（Optuna最优参数等）
├── requirements.txt             # Python依赖包列表
├── .gitignore                   # Git忽略规则
└── README.md                    # 项目说明文档
```

## 🔧 技术栈

- **编程语言**：Python 3.8+
- **数据处理**：Pandas, NumPy
- **可视化**：Matplotlib, Seaborn
- **机器学习**：Scikit-learn, XGBoost, CatBoost
- **优化**：Optuna
- **开发环境**：Jupyter Notebook, VS Code

## 📈 主要工作流程

### 1. 探索性数据分析（EDA）
- 目标变量分布分析
- 数值型特征与价格相关性分析
- 分类特征分布分析
- 缺失值处理策略制定

### 2. 数据预处理
- 缺失值填充（基于业务逻辑和统计方法）
- 异常值检测与处理
- 数据类型转换
- 日期特征提取（车龄计算等）

### 3. 特征工程
- 创建衍生特征（车龄、季度等）
- 多项式特征和交互特征
- 特征选择与优化
- 聚类特征生成

### 4. 建模策略
- **基线模型**：XGBoost回归
- **优化模型**：CatBoost回归
- **分层建模**：基于K-means聚类结果的分簇建模
- **模型融合**：多模型集成

### 5. 模型优化
- 超参数调优（Optuna）
- 特征重要性分析
- 模型性能对比
- 过拟合预防（正则化、早停）

## 🏆 模型性能

当前最优模型性能（基于CatBoost优化版）：

| 指标 | 值 | 说明 |
|------|-----|------|
| RMSE | 1215.51 | 均方根误差 |
| MAE | 505.44 | 平均绝对误差 |
| R² | 0.9730 | 决定系数 |

## 🚀 快速开始

### 环境准备
```bash
# 克隆项目
git clone <repository-url>

# 安装依赖
pip install -r requirements.txt
```

### 运行完整流程
```bash
# 1. 数据预处理
python src/preprocessing/preprocessing.py

# 2. 特征工程
python src/feature_engineering/feature_engineering.py

# 3. 聚类分析
python src/clustering/clustering_optimized.py

# 4. 模型训练
python src/modeling/xgboost_modeling_final.py
```

### 运行特定模块
```bash
# 探索性数据分析
python src/data_analysis/eda_analysis.py

# 聚类分析
python src/clustering/clustering_optimized.py

# 模型优化
python src/modeling/catboost_optimization_plan_A_merged.py
```

## 📁 文件说明

### 源代码 (`src/`)

**数据分析** (`src/data_analysis/`)
- `analyze_data.py` - 基础数据分析
- `eda_analysis.py` - 探索性数据分析
- `check_data_status.py` - 数据状态检查

**数据预处理** (`src/preprocessing/`)
- `read_data.py` - 数据读取
- `preprocessing.py` - 主预处理流程
- `verify_preprocessing.py` - 预处理验证
- `refined_imputation.py` - 精细化缺失值填充
- `refined_imputation_k5_xgboost.py` - K5方案XGBoost精填
- `refined_imputation_k6.py` - K6方案精填
- `recalculate_derived_features.py` - 衍生特征重算

**特征工程** (`src/feature_engineering/`)
- `feature_engineering.py` - 特征工程主流程
- `clip_power_outliers.py` - power异常值裁剪
- `analyze_v14_outliers.py` - v_14异常值分析
- `check_outliers_k5_k6.py` - K5/K6异常值检查
- `compare_derived_features.py` - 衍生特征对比
- `analyze_feature_optimization.py` - 特征优化分析
- `compare_power_clipping.py` - power裁剪方案对比

**建模** (`src/modeling/`)
- `xgboost_modeling.py` - XGBoost基线建模
- `xgboost_modeling_final.py` - XGBoost最终版建模
- `catboost_modeling_k5.py` - CatBoost K5方案建模
- `catboost_optimization_plan_A.py` - CatBoost方案A优化
- `catboost_optimization_plan_A_merged.py` - CatBoost方案A合并版
- `catboost_optuna_tuning.py` - Optuna超参数调优
- `catboost_optuna_tuning_clean.py` - Optuna纯净版调优
- `full_pipeline_k6.py` - K6方案完整流水线

**聚类分析** (`src/clustering/`)
- `clustering_analysis.py` - 聚类分析
- `clustering_fast.py` - 快速聚类
- `clustering_k6.py` - K6聚类方案
- `clustering_optimized.py` - 优化聚类分析

**模型评估** (`src/evaluation/`)
- `compare_model_performance.py` - 模型性能对比
- `compare_k5_k6.py` - K5/K6方案对比
- `compare_k5_k6_features.py` - K5/K6特征对比
- `compare_k5_k6_results.py` - K5/K6结果对比

### 报告文档 (`reports/`)
- `EDA分析报告.md` - 探索性数据分析报告
- `数据预处理报告.md` - 预处理流程报告
- `数据字段说明.md` - 数据字段含义说明
- `聚类分析报告.md` - 聚类分析报告
- `XGBoost建模报告.md` - XGBoost建模报告
- `XGBoost建模报告-Final数据.md` - XGBoost最终数据报告
- `CatBoost建模报告-K5.md` - CatBoost建模报告
- `特征优化分析报告.md` - 特征优化分析报告
- `簇内精填报告.md` - 簇内精细化填充报告
- `方案A优化报告.md` - 方案A优化报告
- `方案A优化版报告-合并极小簇.md` - 方案A合并极小簇报告
- `方案A纯净版vs含v14标识版对比报告.md` - 方案A版本对比报告
- `超参数调优与v14标识报告.md` - 超参数调优报告
- `k6聚类精填报告.md` - K6聚类精填报告
- `Price_Log变换验证报告.md` - Price Log变换验证
- `项目技术报告.md` - 项目技术总结报告
- `README-Optuna纯净版.md` - Optuna纯净版说明
- `README-方案A优化版.md` - 方案A优化版说明

### 配置文件 (`configs/`)
- `optuna_best_params_k5.json` - Optuna K5方案最优参数
- `optuna_best_params_k5_clean.json` - Optuna K5纯净版最优参数

## 🎓 关键发现与优化策略

### 1. 聚类分层建模
- 使用K-means聚类（K=5）将数据分成5个簇
- 每个簇独立训练模型，提高预测精度
- 合并极小簇（样本量少的簇）减少噪声影响

### 2. 特征优化
- 移除高度相关的冗余特征
- 添加多项式特征（carAge², power²等）
- 添加交互特征（power×carAge等）
- 基于特征重要性筛选最佳特征组合

### 3. 模型优化
- 使用Optuna进行超参数调优
- 早停策略防止过拟合
- 学习率调度优化收敛速度

### 4. 异常值处理
- 识别并处理功率（power）异常值
- v_14特征异常值检测与处理
- 基于统计方法的离群点修正

## 📝 后续优化方向

1. **特征工程深化**
   - 目标编码高基数分类特征
   - 基于业务知识的特征构造
   - 时间序列特征挖掘

2. **模型创新**
   - 深度学习模型尝试
   - 模型融合与堆叠
   - 概率预测模型

3. **工程优化**
   - 流水线自动化
   - 分布式训练
   - 模型部署与API化

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进本项目。主要贡献方向：
- 新特征的想法与实现
- 模型性能优化
- 代码重构与文档完善
- 可视化改进

## 📄 许可证

本项目采用MIT许可证。

## 📞 联系与支持

如有问题或建议，请通过GitHub Issues提交。

---

**最后更新**：2026-06-11  
**当前最优模型**：CatBoost优化方案A（合并极小簇）  
**推荐提交文件**：`submission_catboost_k5_optimized_A_merged.csv`

**项目状态**：🚀 进行中 - 持续优化