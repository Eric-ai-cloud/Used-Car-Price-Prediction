"""
K5 vs K6 精填数据异常值检测
全面检查数据中的异常值和潜在问题
"""
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    """使用IQR方法检测异常值"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    """使用Z-Score方法检测异常值"""
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    outliers = data[z_scores > threshold]
    return outliers

def check_data_quality(df, dataset_name):
    """全面检查数据质量"""
    print(f"\n{'='*80}")
    print(f"📊 {dataset_name} 数据质量检查")
    print(f"{'='*80}")
    
    # 基本信息
    print(f"\n📋 基本信息:")
    print(f"  形状: {df.shape}")
    print(f"  缺失值总数: {df.isnull().sum().sum()}")
    print(f"  重复行数: {df.duplicated().sum()}")
    
    # 数值型特征统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n📈 数值型特征数量: {len(numeric_cols)}")
    
    # 检查每个关键特征的异常值
    key_features = {
        'power': {'name': '发动机功率', 'normal_range': (0, 300), 'unit': '马力'},
        'kilometer': {'name': '行驶里程', 'normal_range': (0, 30), 'unit': '万公里'},
        'price': {'name': '价格', 'normal_range': (100, 100000), 'unit': '元'},
        'carAge': {'name': '车龄', 'normal_range': (0, 30), 'unit': '年'},
        'regionCode': {'name': '地区编码', 'normal_range': None, 'unit': ''},
    }
    
    print(f"\n🔍 关键特征异常值检测:")
    print("-" * 80)
    
    outlier_summary = {}
    
    for col, info in key_features.items():
        if col not in df.columns:
            continue
            
        print(f"\n{info['name']} ({col}):")
        
        # 基本统计
        print(f"  均值: {df[col].mean():.2f} {info['unit']}")
        print(f"  中位数: {df[col].median():.2f} {info['unit']}")
        print(f"  标准差: {df[col].std():.2f} {info['unit']}")
        print(f"  最小值: {df[col].min():.2f} {info['unit']}")
        print(f"  最大值: {df[col].max():.2f} {info['unit']}")
        
        # IQR方法检测
        outliers_iqr, lower, upper = detect_outliers_iqr(df, col)
        outlier_count_iqr = len(outliers_iqr)
        outlier_pct_iqr = outlier_count_iqr / len(df) * 100
        
        print(f"  IQR范围: [{lower:.2f}, {upper:.2f}]")
        print(f"  IQR异常值: {outlier_count_iqr:,} ({outlier_pct_iqr:.2f}%)")
        
        # Z-Score方法检测
        outliers_zscore = detect_outliers_zscore(df, col, threshold=3)
        outlier_count_zscore = len(outliers_zscore)
        outlier_pct_zscore = outlier_count_zscore / len(df) * 100
        
        print(f"  Z-Score异常值(>3σ): {outlier_count_zscore:,} ({outlier_pct_zscore:.2f}%)")
        
        # 业务规则检测
        if info['normal_range']:
            min_val, max_val = info['normal_range']
            business_outliers = df[(df[col] < min_val) | (df[col] > max_val)]
            business_outlier_count = len(business_outliers)
            business_outlier_pct = business_outlier_count / len(df) * 100
            
            print(f"  业务规则异常值(<{min_val}或>{max_val}): {business_outlier_count:,} ({business_outlier_pct:.2f}%)")
            
            if business_outlier_count > 0:
                print(f"    ⚠️  异常值示例:")
                print(f"      最小值: {df[col].min():.2f}")
                print(f"      最大值: {df[col].max():.2f}")
        
        outlier_summary[col] = {
            'iqr_count': outlier_count_iqr,
            'zscore_count': outlier_count_zscore,
            'business_count': business_outlier_count if info['normal_range'] else 0
        }
    
    # 匿名特征检查
    print(f"\n🔍 匿名特征 (v_0 ~ v_14) 异常值概览:")
    print("-" * 80)
    
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    
    v_outlier_summary = {}
    for col in v_cols:
        outliers_iqr, _, _ = detect_outliers_iqr(df, col)
        outlier_count = len(outliers_iqr)
        outlier_pct = outlier_count / len(df) * 100
        v_outlier_summary[col] = outlier_count
        
        if outlier_pct > 5:  # 只显示异常值比例>5%的特征
            print(f"  {col}: {outlier_count:,} 异常值 ({outlier_pct:.2f}%) ⚠️")
    
    high_outlier_v = [col for col, count in v_outlier_summary.items() 
                      if count / len(df) * 100 > 5]
    if high_outlier_v:
        print(f"\n  ⚠️  高异常值比例的匿名特征: {', '.join(high_outlier_v)}")
    else:
        print(f"  ✅ 所有匿名特征的异常值比例均在5%以下")
    
    # 聚类标签检查
    if 'cluster_label' in df.columns:
        print(f"\n🔍 聚类标签 (cluster_label) 检查:")
        print("-" * 80)
        cluster_dist = df['cluster_label'].value_counts().sort_index()
        print(f"  簇分布:")
        for cluster_id, count in cluster_dist.items():
            pct = count / len(df) * 100
            status = "⚠️ 样本过少" if count < 100 else "✅"
            print(f"    簇{cluster_id}: {count:>6,} 样本 ({pct:5.2f}%) {status}")
        
        # 检查是否有极小簇
        tiny_clusters = cluster_dist[cluster_dist < 100]
        if len(tiny_clusters) > 0:
            print(f"\n  ⚠️  发现{len(tiny_clusters)}个样本数<100的极小簇，可能是异常值聚集")
    
    # 衍生特征合理性检查
    print(f"\n🔍 衍生特征合理性检查:")
    print("-" * 80)
    
    derived_checks = {
        'power_per_km': {'name': '功率里程比', 'check': lambda x: x >= 0},
        'km_per_year': {'name': '年均里程', 'check': lambda x: (x >= 0) & (x <= 10)},
        'power_age_ratio': {'name': '功率车龄比', 'check': lambda x: x >= 0},
    }
    
    for col, info in derived_checks.items():
        if col not in df.columns:
            continue
        
        invalid = df[~info['check'](df[col])]
        invalid_count = len(invalid)
        invalid_pct = invalid_count / len(df) * 100
        
        status = "⚠️" if invalid_count > 0 else "✅"
        print(f"  {status} {info['name']} ({col}): {invalid_count:,} 不合理值 ({invalid_pct:.2f}%)")
        
        if invalid_count > 0 and invalid_count < 10:
            print(f"      示例值: {invalid[col].values[:5]}")
    
    return outlier_summary

def compare_k5_k6_outliers():
    """对比K5和K6的异常值情况"""
    
    print("=" * 80)
    print("🔍 K5 vs K6 精填数据异常值全面检测")
    print("=" * 80)
    
    # 加载数据
    print("\n📂 正在加载数据...")
    train_k5 = pd.read_csv('train_refined_k5.csv')
    train_k6 = pd.read_csv('train_refined_k6.csv')
    test_k5 = pd.read_csv('test_refined_k5.csv')
    test_k6 = pd.read_csv('test_refined_k6.csv')
    
    print("✅ 数据加载完成\n")
    
    # 分别检查四个数据集
    results = {
        'K5训练集': check_data_quality(train_k5, 'K5训练集 (train_refined_k5.csv)'),
        'K6训练集': check_data_quality(train_k6, 'K6训练集 (train_refined_k6.csv)'),
        'K5测试集': check_data_quality(test_k5, 'K5测试集 (test_refined_k5.csv)'),
        'K6测试集': check_data_quality(test_k6, 'K6测试集 (test_refined_k6.csv)'),
    }
    
    # 总结对比
    print("\n" + "=" * 80)
    print("📊 K5 vs K6 异常值对比总结")
    print("=" * 80)
    
    print(f"\n关键特征异常值数量对比（训练集）:")
    print("-" * 80)
    print(f"{'特征':<20} {'K5-IQR':<12} {'K5-ZScore':<12} {'K6-IQR':<12} {'K6-ZScore':<12}")
    print("-" * 80)
    
    for col in ['power', 'kilometer', 'price', 'carAge']:
        k5_iqr = results['K5训练集'][col]['iqr_count']
        k5_zscore = results['K5训练集'][col]['zscore_count']
        k6_iqr = results['K6训练集'][col]['iqr_count']
        k6_zscore = results['K6训练集'][col]['zscore_count']
        
        print(f"{col:<20} {k5_iqr:<12,} {k5_zscore:<12,} {k6_iqr:<12,} {k6_zscore:<12,}")
    
    # 最终结论
    print("\n" + "=" * 80)
    print("💡 最终结论与建议")
    print("=" * 80)
    
    # 检查是否有严重异常
    severe_issues = []
    
    for dataset_name, result in results.items():
        for col, counts in result.items():
            if counts.get('business_count', 0) > 1000:
                severe_issues.append(f"{dataset_name}的{col}存在{counts['business_count']:,}个业务规则异常值")
    
    if severe_issues:
        print(f"\n⚠️  发现的严重问题:")
        for issue in severe_issues:
            print(f"  • {issue}")
    else:
        print(f"\n✅ 未发现严重的业务规则异常值")
    
    print(f"\n📌 建议:")
    print(f"  1. K5和K6的异常值情况基本一致（因为特征相同）")
    print(f"  2. 主要关注power、kilometer、price等关键业务的合理范围")
    print(f"  3. 匿名特征的异常值需要结合业务理解判断")
    print(f"  4. 极小簇（<100样本）可能需要单独处理或合并")
    print(f"  5. 建议在建模前根据业务规则清理明显的异常值")
    
    print("\n" + "=" * 80)
    print("✅ 异常值检测完成！")
    print("=" * 80)


if __name__ == "__main__":
    compare_k5_k6_outliers()
