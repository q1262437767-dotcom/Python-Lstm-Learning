"""
第四课（第一天）：机器学习基础概念 + 线性回归
==============================================
在学 LSTM（深度学习）之前，先学 sklearn 的传统机器学习，
理解"训练/预测/评估"这套流程，后面学 LSTM 会轻松很多。

今天内容：
1. 什么是机器学习（5分钟概念）
2. sklearn 的工作流程
3. 线性回归：用降雨量预测位移（最简单的入门模型）
4. 模型评估：看预测准不准
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# 通用设置
# ─────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════
# 【1. 机器学习 vs 传统编程】
# ═══════════════════════════════════════════════
print("=" * 50)
print("【1. 什么是机器学习？】")
print("=" * 50)

print("""
传统编程:
  你写规则 → 电脑按规则算结果
  例: if 降雨量 > 100: 位移 += 5

机器学习:
  你给数据 → 电脑自己学出规则
  例: 给一堆 (降雨量, 位移) 数据 → 模型自己找出规律

              传统编程                    机器学习
          ┌──────────┐              ┌──────────┐
 输入 ──→ │  规则代码  │ ──→ 输出      │   模型    │ ──→ 输出
          └──────────┘              └──────────┘
          你来写规则                 数据训练出规则

sklearn = Python 最常用的机器学习库
  提供各种现成的模型，几行代码就能训练
  你后面学 LSTM 用的 TensorFlow 也类似，只是模型更复杂
""")


# ═══════════════════════════════════════════════
# 【2. sklearn 通用工作流程】
# ═══════════════════════════════════════════════
print("=" * 50)
print("【2. sklearn 工作流程（所有模型都一样）】")
print("=" * 50)

print("""
不管是线性回归、随机森林还是 LSTM，流程都一样：

  ① 准备数据     X（特征）和 y（目标）
  ② 划分数据     训练集 / 测试集（通常 80% / 20%）
  ③ 创建模型     model = LinearRegression()
  ④ 训练模型     model.fit(X_train, y_train)
  ⑤ 预测         model.predict(X_test)
  ⑥ 评估         算 RMSE、R² 等指标

就这么6步！唯一的区别就是第③步换不同的模型。
后面学 LSTM，流程一模一样，只是第③步换成 LSTM 模型。
""")


# ═══════════════════════════════════════════════
# 【3. 实战：线性回归预测位移】
# ═══════════════════════════════════════════════
print("=" * 50)
print("【3. 线性回归：用降雨量预测位移】")
print("=" * 50)

# 读取数据
df = pd.read_csv('D:/python-lstm-learning/02_data_processing/landslide_data.csv')
print(f"\n数据: {len(df)} 个月")
print(df.head())

# ── ① 准备数据 ──
# X = 特征（输入）, y = 目标（要预测的）
X = df[['rainfall']].values     # 二维数组 (72, 1) — sklearn 要求X必须是二维
y = df['displacement'].values   # 一维数组 (72,)

print(f"\nX 形状: {X.shape}  ← 特征，必须是二维")
print(f"y 形状: {y.shape}  ← 目标，一维就行")
print("X[前3个]:", X[:3].flatten())
print("y[前3个]:", y[:3])

# ── ② 划分训练集/测试集 ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 测试集占 20%
    random_state=42       # 固定随机种子，保证每次结果一样
)

print(f"\n训练集: {len(X_train)} 个样本（用来学习）")
print(f"测试集: {len(X_test)} 个样本（用来考试）")

# ── ③ 创建模型 ──
model = LinearRegression()
print("\n模型已创建: LinearRegression()")
print("  线性回归 = 找一条直线 y = a*x + b")
print("  a = 斜率（降雨每多1mm，位移变化多少）")
print("  b = 截距（降雨为0时的基础位移）")

# ── ④ 训练模型 ──
model.fit(X_train, y_train)
print(f"\n✅ 训练完成！")
print(f"  斜率 a = {model.coef_[0]:.4f}  （降雨每多1mm，位移增加约 {model.coef_[0]:.4f}mm）")
print(f"  截距 b = {model.intercept_:.4f}")
print(f"  位移 = {model.coef_[0]:.4f} × 降雨量 + {model.intercept_:.4f}")

# ── ⑤ 预测 ──
y_pred = model.predict(X_test)
print(f"\n预测结果（前5个）:")
for i in range(5):
    print(f"  降雨量: {X_test[i][0]:.1f}mm → 预测位移: {y_pred[i]:.2f}mm | 真实位移: {y_test[i]:.2f}mm | 误差: {abs(y_pred[i]-y_test[i]):.2f}mm")

# ── ⑥ 评估 ──
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'─'*40}")
print(f"评估指标:")
print(f"  RMSE = {rmse:.2f} mm  （均方根误差）")
print(f"  MAE  = {mae:.2f} mm  （平均绝对误差）")
print(f"  R²   = {r2:.4f}      （拟合优度，越接近1越好）")
print(f"{'─'*40}")


# ═══════════════════════════════════════════════
# 【4. 可视化：回归直线 + 预测效果】
# ═══════════════════════════════════════════════
print("\n【4. 可视化结果】")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── 左图：回归直线 ──
ax1 = axes[0]
ax1.scatter(X, y, color='steelblue', alpha=0.6, s=40, label='真实数据')

# 画回归直线
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
ax1.plot(x_line, y_line, color='crimson', linewidth=2,
         label=f'y = {model.coef_[0]:.3f}x + {model.intercept_:.2f}')

# 标出测试集点
ax1.scatter(X_test, y_test, color='orange', s=60, marker='D', label='测试集', zorder=5)

ax1.set_xlabel('降雨量 (mm)', fontsize=12)
ax1.set_ylabel('累积位移 (mm)', fontsize=12)
ax1.set_title(f'线性回归: 降雨量 → 位移 (R²={r2:.4f})', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ── 右图：真实值 vs 预测值 ──
ax2 = axes[1]
ax2.scatter(y_test, y_pred, color='steelblue', s=60, alpha=0.8)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='完美预测线')

ax2.set_xlabel('真实位移 (mm)', fontsize=12)
ax2.set_ylabel('预测位移 (mm)', fontsize=12)
ax2.set_title('真实值 vs 预测值（越接近红线越好）', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/04_machine_learning/01_linear_regression.png', dpi=150)
plt.show()
print("图已保存: 01_linear_regression.png")


# ═══════════════════════════════════════════════
# 【知识点总结】
# ═══════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════╗
║        第一天 知识点总结                           ║
╠══════════════════════════════════════════════════╣
║                                                   ║
║  🤖 机器学习 = 用数据自动学出规则                  ║
║                                                   ║
║  📋 sklearn 流程（所有模型通用）:                  ║
║     1. train_test_split()  → 划分训练/测试集       ║
║     2. model.fit()         → 训练                  ║
║     3. model.predict()     → 预测                  ║
║     4. 算指标(RMSE/R²)     → 评估                  ║
║                                                   ║
║  📐 线性回归:                                      ║
║     找一条直线 y = ax + b                          ║
║     model.coef_     → 斜率 a                      ║
║     model.intercept_ → 截距 b                     ║
║                                                   ║
║  📊 评估指标:                                      ║
║     RMSE → 误差大小（越小越好）                    ║
║     R²   → 拟合优度（0~1，越接近1越好）            ║
║                                                   ║
║  💡 线性回归太简单了，只能找直线关系               ║
║     → 明天学更强大的模型                            ║
║                                                   ║
╚══════════════════════════════════════════════════╝
""")
