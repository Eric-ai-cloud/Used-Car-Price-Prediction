"""
数据状态检查脚本
用于快速查看当前数据处理的状态和统计信息
"""
import pandas as pd
import numpy as np
from pathlib import Path

def check_data_status():
    """检查所有数据文件的状态"""
    
    print("=" * 80)
    print("📊 二手车价格预测 - 数据处理状态检查")
    print("=" * 80)
    
    # 定义要检查的文件
    files_to_check = {
        '原始训练集': 'used_car_train_20200313.csv',
        '原始测试集': 'used_car_testB_20200421.csv',
        '特征工程后(训练)': 'train_featured.csv',
        '特征工程后(测试)': 'test_featured.csv',
        '预处理后(训练)': 'train_preprocessed.csv',
        '预处理后(测试)': 'test_preprocessed.csv',
        'K5聚类(训练)': 'train_with_clusters.csv',
        'K5聚类(测试)': 'test_with_clusters.csv',
        'K6聚类(训练)': 'train_with_clusters_k6.csv',
        'K6聚类(测试)': 'test_with_clusters_k6.csv',
        'K5精填(训练)': 'train_refined_k5.csv',
        'K5精填(测试)': 'test_refined_k5.csv',
        'K6精填(训练)': 'train_refined_k6.csv',
        'K6精填(测试)': 'test_refined_k6.csv',
        '全局精填(训练)': 'train_refined.csv',
        '全局精填(测试)': 'test_refined.csv',
    }
    
    print("\n📁 文件存在性检查:")
    print("-" * 80)
    
    for name, filepath in files_to_check.items():
        path = Path(filepath)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {name:20s} | {filepath:35s} | {size_mb:8.2f} MB")
        else:
            print(f"❌ {name:20s} | {filepath:35s} | 不存在")
    
    # 详细检查关键文件
    print("\n\n🔍 关键数据文件详细分析:")
    print("=" * 80)
    
    key_files = [
        ('最新精填数据(推荐)', 'train_refined.csv', 'test_refined.csv'),
        ('K5精填数据', 'train_refined_k5.csv', 'test_refined_k5.csv'),
        ('K6精填数据', 'train_refined_k6.csv', 'test_refined_k6.csv'),
        ('预处理数据', 'train_preprocessed.csv', 'test_preprocessed.csv'),
    ]
    
    for label, train_file, test_file in key_files:
        print(f"\n{'='*80}")
        print(f"📌 {label}")
        print(f"{'='*80}")
        
        try:
            # 检查训练集
            if Path(train_file).exists():
                train_df = pd.read_csv(train_file, nrows=5)  # 只读前5行获取列名
                train_full = pd.read_csv(train_file)
                
                print(f"\n训练集 ({train_file}):")
                print(f"  形状: {train_full.shape}")
                print(f"  列数: {train_full.shape[1]}")
                print(f"  缺失值总数: {train_full.isnull().sum().sum()}")
                print(f"  数据类型分布:")
                dtype_counts = train_full.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    print(f"    {dtype}: {count} 列")
                
                # 检查是否有簇标签
                if 'cluster_label' in train_full.columns:
                    print(f"  ✅ 包含簇标签 (cluster_label)")
                    cluster_dist = train_full['cluster_label'].value_counts().sort_index()
                    print(f"  簇分布:")
                    for cluster_id, count in cluster_dist.items():
                        pct = count / len(train_full) * 100
                        print(f"    簇{cluster_id}: {count:,} 样本 ({pct:.2f}%)")
                
                # 显示部分列名
                print(f"  前10列: {', '.join(train_full.columns[:10])}")
                
            else:
                print(f"\n❌ 训练集文件不存在: {train_file}")
            
            # 检查测试集
            if Path(test_file).exists():
                test_full = pd.read_csv(test_file)
                
                print(f"\n测试集 ({test_file}):")
                print(f"  形状: {test_full.shape}")
                print(f"  列数: {test_full.shape[1]}")
                print(f"  缺失值总数: {test_full.isnull().sum().sum()}")
                
                # 检查是否有簇标签
                if 'cluster_label' in test_full.columns:
                    print(f"  ✅ 包含簇标签 (cluster_label)")
                    cluster_dist = test_full['cluster_label'].value_counts().sort_index()
                    print(f"  簇分布:")
                    for cluster_id, count in cluster_dist.items():
                        pct = count / len(test_full) * 100
                        print(f"    簇{cluster_id}: {count:,} 样本 ({pct:.2f}%)")
            else:
                print(f"\n❌ 测试集文件不存在: {test_file}")
                
        except Exception as e:
            print(f"\n⚠️ 读取文件时出错: {e}")
    
    # 数据处理流程总结
    print("\n\n" + "=" * 80)
    print("📋 数据处理流程总结")
    print("=" * 80)
    
    flowchart = """
    原始数据 (used_car_train/test)
         ↓
    【步骤1】特征工程 (feature_engineering.py)
         ├─ 提取时间特征 (regYear, regMonth, carAge等8个)
         ├─ 删除无用特征 (SaleID, name, seller, offerType)
         └─ 输出: train_featured.csv, test_featured.csv
         ↓
    【步骤2】数据预处理 (preprocessing.py)
         ├─ 类别特征编码 (频率编码 + 标签编码)
         ├─ KNN智能填补 (k=7, 距离加权)
         ├─ Z-score标准化 (StandardScaler)
         └─ 输出: train_preprocessed.csv, test_preprocessed.csv
         ↓
    【步骤3】聚类分析 (clustering_fast.py)
         ├─ 特征筛选 (57→38个核心特征)
         ├─ 最优簇数确定 (k=5或k=6)
         ├─ K-Means聚类
         └─ 输出: train_with_clusters.csv, test_with_clusters.csv
         ↓
    【步骤4】簇内精填 (refined_imputation.py)
         ├─ 按簇分别处理 (避免跨簇污染)
         ├─ 数值型: 簇内中位数 或 KNN
         ├─ 类别型: 簇内众数 或 Unknown
         ├─ 兜底机制 (样本不足时保留粗填)
         └─ 输出: train_refined.csv, test_refined.csv
              train_refined_k5.csv, test_refined_k5.csv
              train_refined_k6.csv, test_refined_k6.csv
         ↓
    【步骤5】XGBoost建模 (xgboost_modeling.py)
         ├─ 数据清洗 (删除异常值)
         ├─ 模型训练 (R²=0.97, RMSE=1131)
         └─ 输出: submission_xgboost.csv
    """
    
    print(flowchart)
    
    # 推荐使用的数据
    print("\n" + "=" * 80)
    print("💡 推荐使用的数据版本")
    print("=" * 80)
    
    recommendations = [
        ("最佳选择", "train_refined_k5.csv / test_refined_k5.csv", 
         "K5聚类 + 簇内精填，质量最高"),
        ("备选方案", "train_refined_k6.csv / test_refined_k6.csv", 
         "K6聚类 + 簇内精填，更多细分群体"),
        ("快速开始", "train_preprocessed.csv / test_preprocessed.csv", 
         "已完成编码、KNN填补、标准化，可直接建模"),
    ]
    
    for rank, files, reason in recommendations:
        print(f"\n{rank}:")
        print(f"  文件: {files}")
        print(f"  理由: {reason}")
    
    print("\n" + "=" * 80)
    print("✅ 检查完成!")
    print("=" * 80)


if __name__ == "__main__":
    check_data_status()
