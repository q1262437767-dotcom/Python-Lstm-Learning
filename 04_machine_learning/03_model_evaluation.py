# ============================================================
# 第三课：模型评估与调参入门
# 学习目标：
#   1. 学会交叉验证——比单次划分更可靠的评估方法
#   2. 学会网格搜索——自动找最优参数
#   3. 完整的 sklearn 工作流：评估 + 调参 + 对比
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 55)
print("模型评估与调参 — 交叉验证 + 网格搜索")
print("=" * 55)


# ─────────────────────────────────────────────
# 【1. 加载数据】
# ─────────────────────────────────────────────
print("\n【1. 加载数据】")
print("-" * 40)

df = pd.read_csv('D:/python-lstm-learning/02_data_processing/landslide_data.csv')
X = df[['rainfall', 'water_level']].values
y = df['displacement'].values

print(f"数据量: {X.shape[0]} 个样本")


# ─────────────────────────────────────────────
# 【2. 为什么需要交叉验证？】
# ─────────────────────────────────────────────
print("\n\n【2. 为什么需要交叉验证？】")
print("-" * 40)

print("""
之前用的 train_test_split 只拆一次：

  数据: [1][2][3][4][5][6][7][8][9][10]
              训练(80%)          测试(20%)
  [1][2][3][4][5][6][7][8]   [9][10]

  问题：如果碰巧测试集都是简单样本 → 评分虚高
       如果碰巧测试集都是难样本 → 评分虚低
       只测一次，结果有偶然性

交叉验证 = 多次拆分，取平均：

  第1次: [训练][训练][训练][训练][训练][测试][训练][测试][训练][训练]
  第2次: [训练][训练][测试][训练][训练][训练][训练][训练][训练][测试]
  第3次: [训练][测试][训练][训练][测试][训练][训练][训练][训练][训练]
  第4次: [测试][训练][训练][训练][训练][训练][训练][测试][训练][训练]
  第5次: [训练][训练][训练][测试][训练][训练][测试][训练][训练][训练]

  最终得分 = 5次的平均 → 更稳定、更可信
""")

print("K折交叉验证（K-Fold Cross Validation）:")
print(f"  cv=5  → 把数据分成5份，轮流做4份训练+1份测试，共5次")
print(f"  cv=10 → 分10份，共10次")
print(f"  最终得分 = 所有次得分的平均值和标准差")


# ─────────────────────────────────────────────
# 【3. 交叉验证实战】
# ─────────────────────────────────────────────
print("\n\n【3. 交叉验证实战】")
print("-" * 40)

model = RandomForestRegressor(n_estimators=100, random_state=42)

# 交叉验证（返回的是 R2 分数）
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"5折交叉验证结果（R2）:")
for i, score in enumerate(cv_scores):
    print(f"  第{i+1}折: {score:.4f}")
print(f"\n  平均 R2: {cv_scores.mean():.4f}")
print(f"  标准差:  {cv_scores.std():.4f}")
print(f"\n  标准差小说明 → 模型表现稳定，不依赖某一次运气好")
print(f"  标准差大说明 → 模型表现波动大，结果不太可靠")

# 用 RMSE 做交叉验证（需要负号，因为 sklearn 约定越小越好的指标取负值）
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"\n5折交叉验证结果（RMSE）:")
print(f"  平均 RMSE: {-cv_rmse.mean():.2f} mm")


# ─────────────────────────────────────────────
# 【4. 网格搜索 — 自动找最优参数】
# ─────────────────────────────────────────────
print("\n\n【4. 网格搜索（GridSearchCV）】")
print("-" * 40)

print("""
模型有很多参数可以调，手动一个个试太慢。
网格搜索 = 自动帮你穷举所有组合，找出最优的。

随机森林常见参数:
  n_estimators     → 树的数量（10/50/100/200...）
  max_depth        → 树的最大深度（3/5/10/None...）
  min_samples_split → 节点分裂最少需要几个样本（2/5/10...）
  min_samples_leaf  → 叶子节点最少需要几个样本（1/2/4...）
""")

# 定义参数网格：每个参数给几个候选值
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5],
}

print("参数组合数量:")
combinations = 3 * 4 * 2
print(f"  n_estimators: [50, 100, 200]     → 3个")
print(f"  max_depth: [3, 5, 10, None]      → 4个")
print(f"  min_samples_split: [2, 5]        → 2个")
print(f"  总共: 3 x 4 x 2 = {combinations} 种组合")
print(f"  每种组合做 5 折交叉验证")
print(f"  总共训练: {combinations} x 5 = {combinations * 5} 次模型")

print("\n正在搜索最优参数...")

# 创建网格搜索对象
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,                    # 5折交叉验证
    scoring='r2',            # 用R2评分
    n_jobs=-1                # 用所有CPU核心加速
)

# 执行搜索
grid_search.fit(X, y)

print(f"\n搜索完成!")
print(f"  最优参数: {grid_search.best_params_}")
print(f"  最优 R2:  {grid_search.best_score_:.4f}")


# ─────────────────────────────────────────────
# 【5. 用最优模型重新训练和预测】
# ─────────────────────────────────────────────
print("\n\n【5. 最优模型预测】")
print("-" * 40)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 默认模型（没调参）
default_model = RandomForestRegressor(n_estimators=100, random_state=42)
default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)
default_r2 = r2_score(y_test, default_pred)
default_rmse = np.sqrt(mean_squared_error(y_test, default_pred))

# 最优模型（调参后）
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)
best_r2 = r2_score(y_test, best_pred)
best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))

print(f"{'指标':<10} {'默认模型':<15} {'调参后模型':<15}")
print(f"{'─' * 40}")
print(f"{'R2':<10} {default_r2:<15.4f} {best_r2:<15.4f}")
print(f"{'RMSE':<10} {default_rmse:<15.2f} {best_rmse:<15.2f}")

improvement = (best_r2 - default_r2) / default_r2 * 100
if improvement > 0:
    print(f"\n调参后 R2 提升了 {improvement:.1f}%")
else:
    print(f"\n这个数据集上调参提升不大，说明默认参数已经够用了")
    print("（模拟数据量太少，调参效果不明显，真实大数据会有明显提升）")


# ─────────────────────────────────────────────
# 【6. 画图】
# ─────────────────────────────────────────────
print("\n\n【6. 画图】")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('模型评估与调参 — 交叉验证 & 网格搜索', fontsize=16, fontweight='bold')

# ① 左上：交叉验证各折得分
axes[0, 0].bar(range(1, 6), cv_scores, color='steelblue', edgecolor='navy', alpha=0.8)
axes[0, 0].axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'平均: {cv_scores.mean():.4f}')
axes[0, 0].set_xlabel('折数')
axes[0, 0].set_ylabel('R2')
axes[0, 0].set_title('5折交叉验证结果')
axes[0, 0].set_xticks(range(1, 6))
axes[0, 0].legend()

# ② 右上：默认模型 vs 最优模型 预测对比
sort_idx = np.argsort(y_test)
axes[0, 1].plot(range(len(y_test)), y_test[sort_idx], 'k-o', markersize=4,
                label='真实值', linewidth=1.5)
axes[0, 1].plot(range(len(y_test)), default_pred[sort_idx], 'b--s', markersize=4,
                label=f'默认模型 (R2={default_r2:.4f})', linewidth=1.5)
axes[0, 1].plot(range(len(y_test)), best_pred[sort_idx], 'g--^', markersize=4,
                label=f'调参模型 (R2={best_r2:.4f})', linewidth=1.5)
axes[0, 1].set_xlabel('测试样本')
axes[0, 1].set_ylabel('位移 (mm)')
axes[0, 1].set_title('默认 vs 调参 模型对比')
axes[0, 1].legend()

# ③ 左下：预测误差分布
errors_default = default_pred - y_test
errors_best = best_pred - y_test
axes[1, 0].hist(errors_default, bins=8, alpha=0.5, label='默认模型', color='steelblue')
axes[1, 0].hist(errors_best, bins=8, alpha=0.5, label='调参模型', color='darkgreen')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('预测误差 (mm)')
axes[1, 0].set_ylabel('频次')
axes[1, 0].set_title('预测误差分布')
axes[1, 0].legend()

# ④ 右下：特征重要性（用最优模型）
importances = best_model.feature_importances_
features = ['rainfall', 'water_level']
colors = ['#4ECDC4', '#FF6B6B']
axes[1, 1].barh(features, importances, color=colors, edgecolor='black', height=0.5)
axes[1, 1].set_xlabel('重要性')
axes[1, 1].set_title('最优模型 — 特征重要性')
for i, v in enumerate(importances):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/04_machine_learning/03_evaluation_tuning.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("图片已保存: 03_evaluation_tuning.png")


# ─────────────────────────────────────────────
# 【7. 小结】
# ─────────────────────────────────────────────
print("\n\n" + "=" * 55)
print("本课小结")
print("=" * 55)
print("""
1. 交叉验证（cross_val_score）
   比单次划分更可靠，多次训练取平均
   cv=5 是最常用的设置

2. 网格搜索（GridSearchCV）
   自动穷举参数组合 + 交叉验证 = 找最优参数
   参数越多组合越多，搜索越慢

3. 评估指标回顾
   R2    → 拟合优度（0~1，越接近1越好）
   RMSE  → 平均误差（越小越好）
   MAE   → 绝对误差（越小越好）

4. sklearn 完整工作流
   数据准备 → 交叉验证评估 → 网格搜索调参 → 最优模型预测 → 画图对比

5. 重要认知
   - 调参不是万能的，数据质量 > 模型参数
   - 小数据上调参效果可能不明显
   - 交叉验证的标准差比平均值更重要（稳定性）
""")
