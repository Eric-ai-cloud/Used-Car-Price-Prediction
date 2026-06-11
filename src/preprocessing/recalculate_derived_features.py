"""
重新计算截断后的衍生特征
确保衍生特征与截断后的Power值保持一致
"""
import pandas as pd
import numpy as np

def recalculate_derived_features(df, dataset_name):
    """
    重新计算基于Power的衍生特征
    
    参数:
        df: 输入DataFrame（已截断Power）
        dataset_name: 数据集名称
    
    返回:
        更新后的DataFrame
    """
    print(f"\n{'='*80}")
    print(f"🔄 重新计算 {dataset_name} 的衍生特征")
    print(f"{'='*80}")
    
    df_updated = df.copy()
    
    # 记录原始值用于对比
    if 'power_per_km' in df.columns:
        original_power_per_km = df['power_per_km'].mean()
    if 'power_age_ratio' in df.columns:
        original_power_age_ratio = df['power_age_ratio'].mean()
    if 'km_per_year' in df.columns:
        original_km_per_year = df['km_per_year'].mean()
    
    print(f"\n📊 重新计算衍生特征...")
    
    # 1. power_per_km: 功率里程比
    # 避免除以0
    df_updated['power_per_km'] = df_updated['power'] / df_updated['kilometer'].replace(0, np.nan)
    df_updated['power_per_km'] = df_updated['power_per_km'].fillna(0)
    
    # 2. km_per_year: 年均里程 = kilometer / carAge
    # 避免除以0
    df_updated['km_per_year'] = df_updated['kilometer'] / df_updated['carAge'].replace(0, np.nan)
    df_updated['km_per_year'] = df_updated['km_per_year'].fillna(0)
    
    # 3. power_age_ratio: 功率车龄比
    # 避免除以0
    df_updated['power_age_ratio'] = df_updated['power'] / df_updated['carAge'].replace(0, np.nan)
    df_updated['power_age_ratio'] = df_updated['power_age_ratio'].fillna(0)
    
    print(f"\n✅ 衍生特征重新计算完成！")
    
    # 对比变化
    print(f"\n📈 衍生特征变化对比:")
    print(f"{'特征':<25} {'截断前均值':<15} {'截断后均值':<15} {'变化'}")
    print("-" * 70)
    
    if 'power_per_km' in df.columns:
        new_power_per_km = df_updated['power_per_km'].mean()
        change_ppk = new_power_per_km - original_power_per_km
        change_pct_ppk = (change_ppk / original_power_per_km * 100) if original_power_per_km != 0 else 0
        print(f"{'power_per_km':<25} {original_power_per_km:<15.4f} {new_power_per_km:<15.4f} {change_pct_ppk:+.2f}%")
    
    if 'km_per_year' in df.columns:
        new_km_per_year = df_updated['km_per_year'].mean()
        change_kpy = new_km_per_year - original_km_per_year
        change_pct_kpy = (change_kpy / original_km_per_year * 100) if original_km_per_year != 0 else 0
        print(f"{'km_per_year':<25} {original_km_per_year:<15.4f} {new_km_per_year:<15.4f} {change_pct_kpy:+.2f}%")
    
    if 'power_age_ratio' in df.columns:
        new_power_age_ratio = df_updated['power_age_ratio'].mean()
        change_par = new_power_age_ratio - original_power_age_ratio
        change_pct_par = (change_par / original_power_age_ratio * 100) if original_power_age_ratio != 0 else 0
        print(f"{'power_age_ratio':<25} {original_power_age_ratio:<15.4f} {new_power_age_ratio:<15.4f} {change_pct_par:+.2f}%")
    
    # 验证计算结果
    print(f"\n🔍 验证计算结果:")
    print(f"  power_per_km 范围: [{df_updated['power_per_km'].min():.4f}, {df_updated['power_per_km'].max():.4f}]")
    print(f"  km_per_year 范围: [{df_updated['km_per_year'].min():.4f}, {df_updated['km_per_year'].max():.4f}]")
    print(f"  power_age_ratio 范围: [{df_updated['power_age_ratio'].min():.4f}, {df_updated['power_age_ratio'].max():.4f}]")
    
    # 检查是否有异常值
    invalid_ppk = (df_updated['power_per_km'] < 0).sum()
    invalid_kpy = (df_updated['km_per_year'] < 0).sum()
    invalid_par = (df_updated['power_age_ratio'] < 0).sum()
    
    if invalid_ppk > 0 or invalid_kpy > 0 or invalid_par > 0:
        print(f"\n  ⚠️  发现负值:")
        if invalid_ppk > 0:
            print(f"    power_per_km: {invalid_ppk} 个负值")
        if invalid_kpy > 0:
            print(f"    km_per_year: {invalid_kpy} 个负值")
        if invalid_par > 0:
            print(f"    power_age_ratio: {invalid_par} 个负值")
    else:
        print(f"  ✅ 所有衍生特征均为非负值")
    
    return df_updated

def process_all_clipped_datasets():
    """处理所有截断后的数据集，重新计算衍生特征"""
    
    print("=" * 80)
    print("🚀 重新计算截断后数据的衍生特征")
    print("=" * 80)
    print("\n📋 处理策略:")
    print("  1. 基于截断后的Power值重新计算衍生特征")
    print("  2. 确保衍生特征与Power保持一致")
    print("  3. 处理除零情况（用NaN填充为0）")
    
    # 定义要处理的数据集
    datasets = {
        'train_clipped_k5.csv': 'train_final_k5.csv',
        'test_clipped_k5.csv': 'test_final_k5.csv',
        'train_clipped_k6.csv': 'train_final_k6.csv',
        'test_clipped_k6.csv': 'test_final_k6.csv',
    }
    
    results = {}
    
    for input_file, output_file in datasets.items():
        print(f"\n\n{'#'*80}")
        print(f"# 处理文件: {input_file} → {output_file}")
        print(f"{'#'*80}")
        
        try:
            # 读取数据
            print(f"\n📂 正在读取 {input_file}...")
            df = pd.read_csv(input_file)
            print(f"✅ 读取完成: {df.shape[0]:,} 行 × {df.shape[1]} 列")
            
            # 重新计算衍生特征
            df_updated = recalculate_derived_features(df, input_file)
            
            # 保存结果
            print(f"\n💾 正在保存到 {output_file}...")
            df_updated.to_csv(output_file, index=False)
            print(f"✅ 保存完成！")
            
            # 验证保存的文件
            df_verify = pd.read_csv(output_file)
            
            print(f"\n🔍 验证结果:")
            print(f"  文件形状: {df_verify.shape}")
            print(f"  Power范围: [{df_verify['power'].min():.2f}, {df_verify['power'].max():.2f}]")
            print(f"  缺失值总数: {df_verify.isnull().sum().sum()}")
            
            # 验证衍生特征
            derived_cols = ['power_per_km', 'km_per_year', 'power_age_ratio']
            all_valid = True
            for col in derived_cols:
                if col in df_verify.columns:
                    has_negative = (df_verify[col] < 0).any()
                    has_nan = df_verify[col].isnull().any()
                    if has_negative or has_nan:
                        print(f"  ⚠️  {col}: 存在异常值")
                        all_valid = False
            
            if all_valid:
                print(f"  ✅ 所有衍生特征计算正确")
            
            results[input_file] = {
                'status': 'success',
                'records_processed': len(df),
                'output_file': output_file
            }
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            results[input_file] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # 总结报告
    print("\n\n" + "=" * 80)
    print("📊 处理总结报告")
    print("=" * 80)
    
    print(f"\n{'输入文件':<30} {'状态':<10} {'处理记录数':<15} {'输出文件'}")
    print("-" * 80)
    
    for input_file, result in results.items():
        if result['status'] == 'success':
            status_icon = "✅"
            records = f"{result['records_processed']:,}"
            output = result['output_file']
        else:
            status_icon = "❌"
            records = "-"
            output = f"错误: {result['error']}"
        
        print(f"{input_file:<30} {status_icon:<10} {records:<15} {output}")
    
    print(f"\n📌 生成的最终文件:")
    print(f"  ✅ train_final_k5.csv - K5训练集（Power截断 + 衍生特征重算）")
    print(f"  ✅ test_final_k5.csv - K5测试集（Power截断 + 衍生特征重算）")
    print(f"  ✅ train_final_k6.csv - K6训练集（Power截断 + 衍生特征重算）")
    print(f"  ✅ test_final_k6.csv - K6测试集（Power截断 + 衍生特征重算）")
    
    print(f"\n🎯 关键改进:")
    print(f"  1. Power值已截断到[0, 600]范围")
    print(f"  2. 衍生特征基于截断后的Power重新计算")
    print(f"  3. 确保了数据的一致性和准确性")
    print(f"  4. 处理了除零等边界情况")
    
    print(f"\n💡 下一步建议:")
    print(f"  1. 使用final版本的数据进行模型训练")
    print(f"  2. 对比重算前后衍生特征对模型的影响")
    print(f"  3. 检查衍生特征的重要性排名")
    
    print("\n" + "=" * 80)
    print("✅ 衍生特征重新计算完成！")
    print("=" * 80)


if __name__ == "__main__":
    process_all_clipped_datasets()
