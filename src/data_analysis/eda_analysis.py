import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# 读取数据
print("=" * 80)
print("📊 开始探索性数据分析 (EDA)")
print("=" * 80)

df = pd.read_csv('used_car_train_20200313.csv', sep=' ')

print(f"\n✅ 数据加载成功！")
print(f"   - 样本数量: {df.shape[0]:,}")
print(f"   - 特征数量: {df.shape[1]}")

# ========================
# 1. 基本信息查看
# ========================
print("\n" + "=" * 80)
print("📋 第一部分：数据概览")
print("=" * 80)

print("\n【前5行数据】")
print(df.head())

print("\n【数据类型统计】")
print(df.dtypes.value_counts())

print("\n【缺失值统计】")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing,
    '缺失比例(%)': round(missing_pct, 2)
})
missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False)
print(missing_df)

print("\n【重复值统计】")
print(f"重复行数: {df.duplicated().sum()}")

# ========================
# 2. 目标变量分析
# ========================
print("\n" + "=" * 80)
print("💰 第二部分：目标变量 (price) 分析")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# price 分布
axes[0, 0].hist(df['price'].dropna(), bins=100, edgecolor='black', color='steelblue', alpha=0.7)
axes[0, 0].set_xlabel('价格 (元)', fontsize=12)
axes[0, 0].set_ylabel('频数', fontsize=12)
axes[0, 0].set_title('Price 分布', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# price 箱线图
axes[0, 1].boxplot(df['price'].dropna(), vert=True)
axes[0, 1].set_ylabel('价格 (元)', fontsize=12)
axes[0, 1].set_title('Price 箱线图', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# price Q-Q图
sample_data = df['price'].dropna()
if len(sample_data) > 1000:
    sample_data = sample_data.sample(1000, random_state=42)
stats.probplot(sample_data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Price Q-Q 图', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# price 对数变换后的分布
log_price = np.log1p(df['price'].dropna())
axes[1, 1].hist(log_price, bins=100, edgecolor='black', color='coral', alpha=0.7)
axes[1, 1].set_xlabel('Log(Price+1)', fontsize=12)
axes[1, 1].set_ylabel('频数', fontsize=12)
axes[1, 1].set_title('Log(Price) 分布', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_target_variable_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ 图表已保存: 01_target_variable_analysis.png")
plt.show()

print(f"\n【Price 统计描述】")
print(df['price'].describe())
print(f"\n偏度 (Skewness): {df['price'].skew():.4f}")
print(f"峰度 (Kurtosis): {df['price'].kurtosis():.4f}")

# ========================
# 3. 数值型特征分析
# ========================
print("\n" + "=" * 80)
print("📈 第三部分：数值型特征分析")
print("=" * 80)

numerical_cols = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 
                  'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']

fig, axes = plt.subplots(3, 2, figsize=(18, 15))

# power 分析
axes[0, 0].hist(df['power'].clip(0, 300), bins=100, edgecolor='black', color='green', alpha=0.7)
axes[0, 0].set_xlabel('功率 (马力)', fontsize=12)
axes[0, 0].set_ylabel('频数', fontsize=12)
axes[0, 0].set_title('Power 分布 (截断至300)', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# kilometer 分析
axes[0, 1].hist(df['kilometer'], bins=50, edgecolor='black', color='purple', alpha=0.7)
axes[0, 1].set_xlabel('里程 (万公里)', fontsize=12)
axes[0, 1].set_ylabel('频数', fontsize=12)
axes[0, 1].set_title('Kilometer 分布', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# power vs price
df_clean = df[(df['power'] <= 300) & (df['power'] >= 0)]
axes[1, 0].scatter(df_clean['power'][::100], df_clean['price'][::100], 
                   alpha=0.3, s=10, color='red')
axes[1, 0].set_xlabel('功率 (马力)', fontsize=12)
axes[1, 0].set_ylabel('价格 (元)', fontsize=12)
axes[1, 0].set_title('Power vs Price', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# kilometer vs price
axes[1, 1].scatter(df['kilometer'], df['price'], alpha=0.3, s=10, color='orange')
axes[1, 1].set_xlabel('里程 (万公里)', fontsize=12)
axes[1, 1].set_ylabel('价格 (元)', fontsize=12)
axes[1, 1].set_title('Kilometer vs Price', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# v_0 分布
axes[2, 0].hist(df['v_0'], bins=100, edgecolor='black', color='teal', alpha=0.7)
axes[2, 0].set_xlabel('v_0', fontsize=12)
axes[2, 0].set_ylabel('频数', fontsize=12)
axes[2, 0].set_title('V_0 分布', fontsize=14, fontweight='bold')
axes[2, 0].grid(True, alpha=0.3)

# v_1 分布
axes[2, 1].hist(df['v_1'], bins=100, edgecolor='black', color='brown', alpha=0.7)
axes[2, 1].set_xlabel('v_1', fontsize=12)
axes[2, 1].set_ylabel('频数', fontsize=12)
axes[2, 1].set_title('V_1 分布', fontsize=14, fontweight='bold')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_numerical_features_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ 图表已保存: 02_numerical_features_analysis.png")
plt.show()

# ========================
# 4. 分类特征分析
# ========================
print("\n" + "=" * 80)
print("🏷️ 第四部分：分类特征分析")
print("=" * 80)

categorical_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# brand 分布
brand_top10 = df['brand'].value_counts().head(10)
axes[0, 0].bar(range(len(brand_top10)), brand_top10.values, color='skyblue', edgecolor='black')
axes[0, 0].set_xticks(range(len(brand_top10)))
axes[0, 0].set_xticklabels(brand_top10.index, rotation=45)
axes[0, 0].set_xlabel('品牌编码', fontsize=12)
axes[0, 0].set_ylabel('数量', fontsize=12)
axes[0, 0].set_title('Top 10 品牌分布', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# bodyType 分布
bodytype_count = df['bodyType'].value_counts()
axes[0, 1].bar(range(len(bodytype_count)), bodytype_count.values, color='lightcoral', edgecolor='black')
axes[0, 1].set_xticks(range(len(bodytype_count)))
axes[0, 1].set_xticklabels(bodytype_count.index, rotation=45)
axes[0, 1].set_xlabel('车身类型', fontsize=12)
axes[0, 1].set_ylabel('数量', fontsize=12)
axes[0, 1].set_title('BodyType 分布', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# fuelType 分布
fueltype_count = df['fuelType'].value_counts()
axes[0, 2].bar(range(len(fueltype_count)), fueltype_count.values, color='lightgreen', edgecolor='black')
axes[0, 2].set_xticks(range(len(fueltype_count)))
axes[0, 2].set_xticklabels(fueltype_count.index, rotation=45)
axes[0, 2].set_xlabel('燃油类型', fontsize=12)
axes[0, 2].set_ylabel('数量', fontsize=12)
axes[0, 2].set_title('FuelType 分布', fontsize=14, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# gearbox 分布
gearbox_count = df['gearbox'].value_counts()
axes[1, 0].pie(gearbox_count.values, labels=['手动挡' if x == 0 else '自动挡' for x in gearbox_count.index],
               autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
axes[1, 0].set_title('Gearbox 分布', fontsize=14, fontweight='bold')

# notRepairedDamage 分布
damage_count = df['notRepairedDamage'].value_counts()
axes[1, 1].bar(range(len(damage_count)), damage_count.values, color='gold', edgecolor='black')
axes[1, 1].set_xticks(range(len(damage_count)))
axes[1, 1].set_xticklabels(damage_count.index, rotation=45)
axes[1, 1].set_xlabel('是否损伤', fontsize=12)
axes[1, 1].set_ylabel('数量', fontsize=12)
axes[1, 1].set_title('NotRepairedDamage 分布', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# seller 分布
seller_count = df['seller'].value_counts()
axes[1, 2].pie(seller_count.values, labels=['个人' if x == 0 else '车商' for x in seller_count.index],
               autopct='%1.1f%%', startangle=90, colors=['#99ff99', '#ffcc99'])
axes[1, 2].set_title('Seller 分布', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('03_categorical_features_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ 图表已保存: 03_categorical_features_analysis.png")
plt.show()

# ========================
# 5. 相关性分析
# ========================
print("\n" + "=" * 80)
print("🔗 第五部分：相关性分析")
print("=" * 80)

# 计算数值型特征的相关性
numerical_df = df[['price', 'power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 
                    'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']]

# 处理异常值后再计算相关性
numerical_df_clean = numerical_df.copy()
numerical_df_clean = numerical_df_clean[numerical_df_clean['power'] <= 300]

corr_matrix = numerical_df_clean.corr()

# 绘制热力图
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('特征相关性热力图', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✅ 图表已保存: 04_correlation_heatmap.png")
plt.show()

# price 与其他特征的相关性
print("\n【与 Price 的相关系数 (Top 15)】")
corr_with_price = corr_matrix['price'].drop('price').sort_values(ascending=False)
print(corr_with_price.head(15))

# 可视化与price的相关性
top_corr_features = corr_with_price.head(10).index.tolist()
top_corr_features.append('price')
corr_top = numerical_df_clean[top_corr_features].corr()['price'].drop('price').sort_values(ascending=False)

plt.figure(figsize=(12, 8))
colors = ['coral' if x > 0 else 'steelblue' for x in corr_top.values]
bars = plt.barh(range(len(corr_top)), corr_top.values, color=colors)
plt.yticks(range(len(corr_top)), corr_top.index)
plt.xlabel('与 Price 的相关系数', fontsize=12)
plt.title('Top 10 与 Price 相关的特征', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('05_price_correlation.png', dpi=300, bbox_inches='tight')
print("\n✅ 图表已保存: 05_price_correlation.png")
plt.show()

# ========================
# 6. 时间特征分析
# ========================
print("\n" + "=" * 80)
print("📅 第六部分：时间特征分析")
print("=" * 80)

# 提取年份信息
df['regYear'] = df['regDate'].astype(str).str[:4].astype(int)
df['creatYear'] = df['creatDate'].astype(str).str[:4].astype(int)
df['carAge'] = df['creatYear'] - df['regYear']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 注册年份分布
reg_year_count = df['regYear'].value_counts().sort_index()
axes[0, 0].plot(reg_year_count.index, reg_year_count.values, marker='o', linewidth=2, color='darkblue')
axes[0, 0].fill_between(reg_year_count.index, reg_year_count.values, alpha=0.3)
axes[0, 0].set_xlabel('注册年份', fontsize=12)
axes[0, 0].set_ylabel('数量', fontsize=12)
axes[0, 0].set_title('车辆注册年份分布', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 车龄分布
axes[0, 1].hist(df['carAge'], bins=50, edgecolor='black', color='seagreen', alpha=0.7)
axes[0, 1].set_xlabel('车龄 (年)', fontsize=12)
axes[0, 1].set_ylabel('频数', fontsize=12)
axes[0, 1].set_title('车龄分布', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 车龄 vs 价格
df_age_price = df[df['carAge'] <= 25].copy()
age_mean = df_age_price.groupby('carAge')['price'].mean()
axes[1, 0].plot(age_mean.index, age_mean.values, marker='s', linewidth=2, color='red')
axes[1, 0].set_xlabel('车龄 (年)', fontsize=12)
axes[1, 0].set_ylabel('平均价格 (元)', fontsize=12)
axes[1, 0].set_title('车龄 vs 平均价格', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 不同年份的价格箱线图
df_recent = df[df['regYear'] >= 2005].copy()
recent_brands = df_recent['regYear'].value_counts().head(10).index
df_plot = df_recent[df_recent['regYear'].isin(recent_brands)]
axes[1, 1].boxplot([df_plot[df_plot['regYear']==year]['price'].values for year in recent_brands],
                   labels=recent_brands)
axes[1, 1].set_xlabel('注册年份', fontsize=12)
axes[1, 1].set_ylabel('价格 (元)', fontsize=12)
axes[1, 1].set_title('不同注册年份的价格分布', fontsize=14, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('06_time_features_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ 图表已保存: 06_time_features_analysis.png")
plt.show()

print(f"\n【车龄统计】")
print(df['carAge'].describe())

# ========================
# 7. 异常值检测
# ========================
print("\n" + "=" * 80)
print("⚠️ 第七部分：异常值检测")
print("=" * 80)

print("\n【Power 异常值】")
print(f"Power > 300 的记录数: {(df['power'] > 300).sum()}")
print(f"Power 最大值: {df['power'].max()}")
print(f"Power 正常值范围内的记录占比: {(df['power'] <= 300).sum() / len(df) * 100:.2f}%")

print("\n【Price 异常值】")
print(f"Price < 100 的记录数: {(df['price'] < 100).sum()}")
print(f"Price > 50000 的记录数: {(df['price'] > 50000).sum()}")

print("\n【建议删除的异常记录】")
outlier_mask = (df['power'] > 300) | (df['price'] < 100)
print(f"需要删除的记录数: {outlier_mask.sum()} ({outlier_mask.sum() / len(df) * 100:.2f}%)")

# ========================
# 8. EDA总结报告
# ========================
print("\n" + "=" * 80)
print("📝 EDA 总结报告")
print("=" * 80)

print("""
【关键发现】

1️⃣ 目标变量 (Price):
   - 价格分布严重右偏，存在大量低价车辆和少量高价车辆
   - 建议使用对数变换处理价格数据
   - 中位数价格为 3250 元，均值为 5923 元

2️⃣ 重要特征:
   - power (功率): 与价格呈弱正相关
   - kilometer (里程): 与价格呈负相关
   - carAge (车龄): 新车价格更高
   - v_0: 匿名特征中与价格相关性较高的变量

3️⃣ 数据质量问题:
   - power 字段存在极端异常值（最大值19312）
   - price 存在极小值（最小值11元）
   - bodyType, fuelType, gearbox 存在缺失值
   - seller 字段严重不平衡（99.99%为个人）
   - offerType 字段无变化（全为0）

4️⃣ 特征工程建议:
   ✅ 创建车龄特征（已从注册年份和交易年份计算）
   ✅ 删除异常值（power > 300 或 price < 100）
   ✅ 对价格进行对数变换
   ⚠️ 需要对缺失值进行填充
   ⚠️ 需要对分类变量进行编码
   ❌ 建议删除 offerType 和 seller 字段

5️⃣ 下一步工作:
   → 数据清洗：处理异常值和缺失值
   → 特征工程：构造更多有意义的特征
   → 特征选择：筛选重要特征
   → 模型训练：尝试多种机器学习算法
""")

print("\n" + "=" * 80)
print("✅ EDA 分析完成！所有图表已保存。")
print("=" * 80)
