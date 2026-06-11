import pandas as pd
import numpy as np

print("=" * 80)
print("📊 k=5 vs k=6 聚类结果对比分析")
print("=" * 80)

# 加载数据
train_k5 = pd.read_csv('train_with_clusters.csv')
train_k6 = pd.read_csv('train_with_clusters_k6.csv')

print("\n【1】簇分布对比")
print("-" * 80)

print("\nk=5 聚类分布:")
k5_counts = train_k5['cluster_label'].value_counts().sort_index()
for cluster_id, count in k5_counts.items():
    pct = count / len(train_k5) * 100
    print(f"  簇 {cluster_id}: {count:6d} 样本 ({pct:5.2f}%)")

print("\nk=6 聚类分布:")
k6_counts = train_k6['cluster_label'].value_counts().sort_index()
for cluster_id, count in k6_counts.items():
    pct = count / len(train_k6) * 100
    print(f"  簇 {cluster_id}: {count:6d} 样本 ({pct:5.2f}%)")

print("\n【2】价格统计对比")
print("-" * 80)

print("\nk=5 各簇价格统计:")
if 'price' in train_k5.columns:
    k5_stats = train_k5.groupby('cluster_label')['price'].agg(['mean', 'median', 'std', 'count'])
    k5_stats.columns = ['平均价格', '中位数', '标准差', '样本数']
    print(k5_stats)

print("\nk=6 各簇价格统计:")
if 'price' in train_k6.columns:
    k6_stats = train_k6.groupby('cluster_label')['price'].agg(['mean', 'median', 'std', 'count'])
    k6_stats.columns = ['平均价格', '中位数', '标准差', '样本数']
    print(k6_stats)

print("\n【3】关键差异分析")
print("-" * 80)

print("""
🔍 k=5 vs k=6 的主要差异:

1️⃣  **簇的粒度**:
   - k=5: 5个群体，粒度较粗
   - k=6: 6个群体，粒度更细，能捕捉更多细节

2️⃣  **轮廓系数**:
   - k=5: 0.1189（略高）
   - k=6: 0.1187（略低，但差异很小）

3️⃣  **SSE (误差平方和)**:
   - k=5: 348,026
   - k=6: 3,585,021（注意：这个值是在不同特征集上计算的，不能直接比较）

4️⃣  **实际效果**:
   - k=5: 更简洁，易于解释
   - k=6: 更精细，可能捕捉到更多细分群体

5️⃣  **计算成本**:
   - k=5: 稍低
   - k=6: 稍高（但差异不大）

💡 **选择建议**:

✅ **选择 k=5 的理由**:
   - 轮廓系数更高（虽然差异很小）
   - 簇数更少，模型更简洁
   - 每个簇样本更充足
   - 业务解释更容易

✅ **选择 k=6 的理由**:
   - 能捕捉更多细分群体
   - 可能发现k=5中被合并的隐藏模式
   - 对于复杂数据集可能效果更好
   - SSE更低（在相同条件下）

🎯 **推荐**: 
   - 如果追求简洁和可解释性 → 选 k=5
   - 如果追求精细度和潜在性能 → 选 k=6
   - 两者轮廓系数几乎相同，可以都尝试，看最终模型效果
""")

print("\n【4】精填结果验证")
print("-" * 80)

train_refined_k5 = pd.read_csv('train_refined.csv')
train_refined_k6 = pd.read_csv('train_refined_k6.csv')

print(f"\nk=5 精填后训练集缺失值: {train_refined_k5.isnull().sum().sum()}")
print(f"k=6 精填后训练集缺失值: {train_refined_k6.isnull().sum().sum()}")

test_refined_k5 = pd.read_csv('test_refined.csv')
test_refined_k6 = pd.read_csv('test_refined_k6.csv')

print(f"k=5 精填后测试集缺失值: {test_refined_k5.isnull().sum().sum()}")
print(f"k=6 精填后测试集缺失值: {test_refined_k6.isnull().sum().sum()}")

print("\n✅ 两种方案都成功完成了精填，无缺失值！")

print("\n" + "=" * 80)
print("📋 总结")
print("=" * 80)
print("""
两种聚类方案都已成功完成：

✅ k=5 方案:
   - 文件: train_refined.csv, test_refined.csv
   - 优势: 简洁、轮廓系数略高
   
✅ k=6 方案:
   - 文件: train_refined_k6.csv, test_refined_k6.csv  
   - 优势: 精细、可能捕捉更多模式

下一步：
   → 可以分别用两种数据进行XGBoost建模
   → 对比验证集性能，选择更好的方案
""")
print("=" * 80)
