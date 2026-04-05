# ============================================================
# 第二课：多特征回归 & 随机森林
# 学习目标：
#   1. 理解"特征工程"——用更多输入提升预测效果
#   2. 学会随机森林——比线性回归强的非线性模型
#   3. sklearn 流程复用：同样的6步，换模型就行
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 55)
print("多特征回归 & 随机森林 — 用更多特征提升预测")
print("=" * 55)


# ─────────────────────────────────────────────
# 【1. 准备数据 — 用 landslide_data.csv】
# ─────────────────────────────────────────────
print("\n【1. 加载数据】")
print("-" * 40)

df = pd.read_csv('D:/python-lstm-learning/02_data_processing/landslide_data.csv')
print(f"数据形状: {df.shape}")
print(df.head())

# 取出特征和目标
# X = 输入特征（降雨量 + 水位）
# y = 预测目标（位移）
X = df[['rainfall', 'water_level']].values
y = df['displacement'].values

print(f"\n特征 X 形状: {X.shape}  ← {X.shape[0]}个样本, {X.shape[1]}个特征")
print(f"目标 y 形状: {y.shape}")


# ─────────────────────────────────────────────
# 【2. 划分训练集和测试集】
# ─────────────────────────────────────────────
print("\n\n【2. 划分训练集 / 测试集】")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]} 个样本（用来学习）")
print(f"测试集: {X_test.shape[0]} 个样本（用来考试）")
print(f"\n训练集就是平时作业，测试集就是期末考试")
print(f"模型在训练集上学规律，在测试集上考真本事")


# ─────────────────────────────────────────────
# 【3. 模型一：线性回归（2个特征）】
# ─────────────────────────────────────────────
print("\n\n【3. 线性回归（2个特征）】")
print("-" * 40)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # 训练
lr_pred = lr_model.predict(X_test)  # 预测

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print(f"RMSE: {lr_rmse:.2f} mm  ← 平均误差")
print(f"MAE:  {lr_mae:.2f} mm   ← 绝对误差")
print(f"R²:   {lr_r2:.4f}      ← 拟合优度")
print(f"\n系数: 降雨量={lr_model.coef_[0]:.4f}, 水位={lr_model.coef_[1]:.4f}")


# ─────────────────────────────────────────────
# 【4. 模型二：随机森林（非线性模型）】
# ─────────────────────────────────────────────
print("\n\n【4. 随机森林回归】")
print("-" * 40)

print("\n💡 随机森林是什么？")
print("   线性回归 = 画一条直线拟合数据")
print("   随机森林 = 画很多条弯弯曲曲的线，取平均")
print("   弯曲的线能更好地捕捉复杂的非线性关系")
print()
print("   原理（简单版）：")
print("   ┌─────────────────────────────────────┐")
print("   │  树1（用部分数据）→ 预测值1        │")
print("   │  树2（用部分数据）→ 预测值2        │")
print("   │  树3（用部分数据）→ 预测值3        │")
print("   │  ...（默认100棵树）                │")
print("   │  最终预测 = 所有树预测值的平均       │")
print("   └─────────────────────────────────────┘")

# 创建随机森林模型
# n_estimators=100 → 100棵决策树
# random_state=42  → 固定随机种子，结果可复现
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # 训练
rf_pred = rf_model.predict(X_test)  # 预测

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"\n结果:")
print(f"RMSE: {rf_rmse:.2f} mm")
print(f"MAE:  {rf_mae:.2f} mm")
print(f"R²:   {rf_r2:.4f}")


# ─────────────────────────────────────────────
# 【5. 特征重要性 — 随机森林独有的优势】
# ─────────────────────────────────────────────
print("\n\n【5. 特征重要性】")
print("-" * 40)

importances = rf_model.feature_importances_
features = ['rainfall', 'water_level']

print("随机森林能告诉你：哪个特征更重要")
for feat, imp in zip(features, importances):
    bar = '█' * int(imp * 40)
    print(f"  {feat}: {imp:.3f} {bar}")


# ─────────────────────────────────────────────
# 【6. 两个模型对比】
# ─────────────────────────────────────────────
print("\n\n【6. 模型对比】")
print("-" * 40)
print(f"{'指标':<10} {'线性回归':<15} {'随机森林':<15}")
print(f"{'─' * 40}")
print(f"{'RMSE':<10} {lr_rmse:<15.2f} {rf_rmse:<15.2f}")
print(f"{'MAE':<10} {lr_mae:<15.2f} {rf_mae:<15.2f}")
print(f"{'R²':<10} {lr_r2:<15.4f} {rf_r2:<15.4f}")

better = '随机森林' if rf_r2 > lr_r2 else '线性回归'
print(f"\n🏆 整体表现更好: {better}")


# ─────────────────────────────────────────────
# 【7. 画图对比 — 答辩用】
# ─────────────────────────────────────────────
print("\n\n【7. 画图】")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('机器学习模型对比 — 白水河滑坡位移预测', fontsize=16, fontweight='bold')

# 排序（按真实值从小到大），方便画对比线
sort_idx = np.argsort(y_test)

# ① 左上：线性回归 - 预测 vs 真实
axes[0, 0].scatter(y_test, lr_pred, alpha=0.6, color='steelblue', s=50)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', linewidth=2, label='完美预测线')
axes[0, 0].set_xlabel('真实值 (mm)')
axes[0, 0].set_ylabel('预测值 (mm)')
axes[0, 0].set_title(f'线性回归: R2={lr_r2:.4f}')
axes[0, 0].legend()
axes[0, 0].set_aspect('equal', adjustable='box')

# ② 右上：随机森林 - 预测 vs 真实
axes[0, 1].scatter(y_test, rf_pred, alpha=0.6, color='darkgreen', s=50)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', linewidth=2, label='完美预测线')
axes[0, 1].set_xlabel('真实值 (mm)')
axes[0, 1].set_ylabel('预测值 (mm)')
axes[0, 1].set_title(f'随机森林: R2={rf_r2:.4f}')
axes[0, 1].legend()
axes[0, 1].set_aspect('equal', adjustable='box')

# ③ 左下：预测对比折线图
axes[1, 0].plot(range(len(y_test)), y_test[sort_idx], 'k-o', markersize=4,
                label='真实值', linewidth=1.5)
axes[1, 0].plot(range(len(y_test)), lr_pred[sort_idx], 'b--s', markersize=4,
                label='线性回归', linewidth=1.5)
axes[1, 0].plot(range(len(y_test)), rf_pred[sort_idx], 'g--^', markersize=4,
                label='随机森林', linewidth=1.5)
axes[1, 0].set_xlabel('测试样本（按位移从小到大排序）')
axes[1, 0].set_ylabel('位移 (mm)')
axes[1, 0].set_title('预测对比折线图')
axes[1, 0].legend()

# ④ 右下：特征重要性柱状图
colors = ['#4ECDC4', '#FF6B6B']
axes[1, 1].barh(features, importances, color=colors, edgecolor='black', height=0.5)
axes[1, 1].set_xlabel('重要性')
axes[1, 1].set_title('随机森林 — 特征重要性')
for i, v in enumerate(importances):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/04_machine_learning/02_model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ 图片已保存: 02_model_comparison.png")


# ─────────────────────────────────────────────
# 【8. 小结】
# ─────────────────────────────────────────────
print("\n\n" + "=" * 55)
print("📖 本课小结")
print("=" * 55)
print("""
1. 多特征输入
   X = [[降雨, 水位], [降雨, 水位], ...]
   用两个特征比用一个特征效果好

2. 随机森林 vs 线性回归
   线性回归 → 只能画直线
   随机森林 → 能画弯曲线，捕捉非线性关系
   sklearn 流程一样，只是换了个模型类名

3. 特征重要性
   rf_model.feature_importances_ 告诉你哪个特征更关键
   这是随机森林独有的优势

4. 评估指标
   RMSE → 平均误差（越小越好）
   MAE  → 绝对误差（越小越好）
   R²   → 拟合优度（越接近1越好）

5. 关键认知
   sklearn 的6步流程是通用的：
   准备数据 → train_test_split → 创建模型 → fit → predict → 评估
   不管换成什么模型，这6步都不变
""")
