"""
Power异常值截断处理
根据比赛规则：Power范围为[0, 600]，超出范围的值截断到边界
"""
import pandas as pd
import numpy as np

def clip_power_outliers(df, dataset_name):
    """
    对DataFrame中的power列进行截断处理
    
    参数:
        df: 输入DataFrame
        dataset_name: 数据集名称（用于打印信息）
    
    返回:
        处理后的DataFrame
    """
    print(f"\n{'='*80}")
    print(f"🔧 处理 {dataset_name} 的Power异常值")
    print(f"{'='*80}")
    
    # 统计截断前的情况
    original_min = df['power'].min()
    original_max = df['power'].max()
    original_mean = df['power'].mean()
    original_median = df['power'].median()
    
    print(f"\n📊 截断前统计:")
    print(f"  最小值: {original_min:.2f} 马力")
    print(f"  最大值: {original_max:.2f} 马力")
    print(f"  均值: {original_mean:.2f} 马力")
    print(f"  中位数: {original_median:.2f} 马力")
    
    # 统计需要截断的记录数
    below_zero = (df['power'] < 0).sum()
    above_600 = (df['power'] > 600).sum()
    total_clipped = below_zero + above_600
    total_records = len(df)
    clip_pct = total_clipped / total_records * 100
    
    print(f"\n⚠️  异常值统计:")
    print(f"  Power < 0: {below_zero:,} 条 ({below_zero/total_records*100:.2f}%)")
    print(f"  Power > 600: {above_600:,} 条 ({above_600/total_records*100:.2f}%)")
    print(f"  需要截断总数: {total_clipped:,} 条 ({clip_pct:.2f}%)")
    
    if total_clipped > 0:
        # 显示极端值示例
        extreme_high = df[df['power'] > 600]['power'].describe()
        if len(extreme_high) > 0:
            print(f"\n  🔴 Power > 600的极端值示例:")
            print(f"      最大值: {df['power'].max():.2f}")
            print(f"      次大值: {df['power'].nlargest(2).iloc[-1]:.2f}")
            print(f"      第三大: {df['power'].nlargest(3).iloc[-1]:.2f}")
    
    # 执行截断
    df_clipped = df.copy()
    df_clipped['power'] = df_clipped['power'].clip(lower=0, upper=600)
    
    # 统计截断后的情况
    clipped_min = df_clipped['power'].min()
    clipped_max = df_clipped['power'].max()
    clipped_mean = df_clipped['power'].mean()
    clipped_median = df_clipped['power'].median()
    
    print(f"\n✅ 截断后统计:")
    print(f"  最小值: {clipped_min:.2f} 马力")
    print(f"  最大值: {clipped_max:.2f} 马力")
    print(f"  均值: {clipped_mean:.2f} 马力")
    print(f"  中位数: {clipped_median:.2f} 马力")
    
    # 对比变化
    print(f"\n📈 变化对比:")
    print(f"  最小值: {original_min:.2f} → {clipped_min:.2f} ({'↑' if clipped_min > original_min else '→'})")
    print(f"  最大值: {original_max:.2f} → {clipped_max:.2f} (↓)")
    print(f"  均值: {original_mean:.2f} → {clipped_mean:.2f} ({'↓' if clipped_mean < original_mean else '→'})")
    print(f"  中位数: {original_median:.2f} → {clipped_median:.2f} (→)")
    
    print(f"\n✨ 截断完成！共处理 {total_clipped:,} 条异常记录")
    
    return df_clipped

def process_all_datasets():
    """处理所有K5和K6数据集"""
    
    print("=" * 80)
    print("🚀 Power异常值截断处理 - K5 & K6 精填数据")
    print("=" * 80)
    print("\n📋 比赛规则: Power发动机功率范围为 [0, 600] 马力")
    print("📋 处理策略: 超出范围的值截断到边界（<0→0, >600→600）")
    
    # 定义要处理的数据集
    datasets = {
        'train_refined_k5.csv': 'train_clipped_k5.csv',
        'test_refined_k5.csv': 'test_clipped_k5.csv',
        'train_refined_k6.csv': 'train_clipped_k6.csv',
        'test_refined_k6.csv': 'test_clipped_k6.csv',
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
            
            # 截断处理
            df_clipped = clip_power_outliers(df, input_file)
            
            # 保存结果
            print(f"\n💾 正在保存到 {output_file}...")
            df_clipped.to_csv(output_file, index=False)
            print(f"✅ 保存完成！")
            
            # 验证保存的文件
            df_verify = pd.read_csv(output_file)
            verify_min = df_verify['power'].min()
            verify_max = df_verify['power'].max()
            
            print(f"\n🔍 验证结果:")
            print(f"  文件形状: {df_verify.shape}")
            print(f"  Power范围: [{verify_min:.2f}, {verify_max:.2f}]")
            print(f"  缺失值: {df_verify.isnull().sum().sum()}")
            
            if verify_min >= 0 and verify_max <= 600:
                print(f"  ✅ Power值全部在[0, 600]范围内")
            else:
                print(f"  ❌ 警告: Power值仍有超出范围的情况！")
            
            results[input_file] = {
                'status': 'success',
                'records_processed': len(df),
                'records_clipped': ((df['power'] < 0) | (df['power'] > 600)).sum(),
                'output_file': output_file
            }
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            results[input_file] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # 总结报告
    print("\n\n" + "=" * 80)
    print("📊 处理总结报告")
    print("=" * 80)
    
    print(f"\n{'数据集':<30} {'状态':<10} {'处理记录数':<15} {'截断数量':<15} {'输出文件'}")
    print("-" * 80)
    
    for input_file, result in results.items():
        if result['status'] == 'success':
            status_icon = "✅"
            records = f"{result['records_processed']:,}"
            clipped = f"{result['records_clipped']:,}"
            output = result['output_file']
        else:
            status_icon = "❌"
            records = "-"
            clipped = "-"
            output = f"错误: {result['error']}"
        
        print(f"{input_file:<30} {status_icon:<10} {records:<15} {clipped:<15} {output}")
    
    # 关键发现
    print(f"\n💡 关键发现:")
    total_clipped = sum(r['records_clipped'] for r in results.values() if r['status'] == 'success')
    total_processed = sum(r['records_processed'] for r in results.values() if r['status'] == 'success')
    
    print(f"  • 总共处理了 {total_processed:,} 条记录")
    print(f"  • 共截断了 {total_clipped:,} 条Power异常记录")
    print(f"  • 截断比例: {total_clipped/total_processed*100:.2f}%")
    
    print(f"\n📌 生成的文件:")
    print(f"  ✅ train_clipped_k5.csv - K5训练集（Power已截断）")
    print(f"  ✅ test_clipped_k5.csv - K5测试集（Power已截断）")
    print(f"  ✅ train_clipped_k6.csv - K6训练集（Power已截断）")
    print(f"  ✅ test_clipped_k6.csv - K6测试集（Power已截断）")
    
    print(f"\n🎯 下一步建议:")
    print(f"  1. 使用截断后的数据进行模型训练")
    print(f"  2. 对比截断前后的模型性能差异")
    print(f"  3. 检查Power截断对特征重要性的影响")
    
    print("\n" + "=" * 80)
    print("✅ Power异常值截断处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    process_all_datasets()
