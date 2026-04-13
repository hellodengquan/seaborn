"""分析 KDE log_scale 问题的根本原因"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# 生成对数正态分布数据
np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

# 手动计算 KDE 来理解问题
log_vals = np.log10(vals)

# 在 log 空间计算 KDE
kde_log = gaussian_kde(log_vals)

# 创建 support 网格（在 log 空间）
log_support = np.linspace(log_vals.min(), log_vals.max(), 200)

# 在 log 空间评估 KDE
density_log = kde_log(log_support)

# 转换到线性空间
linear_support = np.power(10, log_support)

# 问题：density_log 是在 log 空间的密度
# 要得到线性空间的密度，需要乘以 Jacobian: d(log10(x))/dx = 1/(x*ln(10))
jacobian = 1 / (linear_support * np.log(10))
density_linear_correct = density_log * jacobian

print("=== 理论分析 ===")
print(f"Log space KDE integral (should be ~1): {np.trapz(density_log, log_support):.4f}")
print(f"Linear space KDE (no Jacobian) integral: {np.trapz(density_log, linear_support):.4f}")
print(f"Linear space KDE (with Jacobian) integral: {np.trapz(density_linear_correct, linear_support):.4f}")

# 计算直方图
hist, bin_edges = np.histogram(vals, bins=20)
bin_widths = np.diff(bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 直方图归一化到面积为1
hist_normalized = hist / (hist * bin_widths).sum()

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: log 空间的 KDE
ax = axes[0, 0]
ax.plot(log_support, density_log, 'b-', label='KDE in log space')
ax.hist(log_vals, bins=20, density=True, alpha=0.5, label='Histogram (density=True)')
ax.set_xlabel('log10(x)')
ax.set_title('Log Space: KDE matches histogram')
ax.legend()

# 右上: 线性空间，无 Jacobian 修正
ax = axes[0, 1]
ax.plot(linear_support, density_log, 'r-', label='KDE (no Jacobian)')
ax.bar(bin_centers, hist_normalized, width=bin_widths, alpha=0.5, label='Histogram (normalized)')
ax.set_xlabel('x (linear)')
ax.set_title('Linear Space: KDE WITHOUT Jacobian (WRONG)')
ax.set_xlim(0, 10)
ax.legend()

# 左下: 线性空间，有 Jacobian 修正
ax = axes[1, 0]
ax.plot(linear_support, density_linear_correct, 'g-', label='KDE (with Jacobian)')
ax.bar(bin_centers, hist_normalized, width=bin_widths, alpha=0.5, label='Histogram (normalized)')
ax.set_xlabel('x (linear)')
ax.set_title('Linear Space: KDE WITH Jacobian (CORRECT)')
ax.set_xlim(0, 10)
ax.legend()

# 右下: seaborn 的 histplot
ax = axes[1, 1]
sns.histplot(x=vals, log_scale=True, kde=True, bins=20, ax=ax)
ax.set_title('Seaborn histplot (log_scale=True)')
ax.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('/Users/dengquan/Downloads/job/bytedance/code/prod/dogfooding-3-425/kimi/kde_log_analysis.png', dpi=150)
print("\n分析图已保存到 kde_log_analysis.png")

# 计算 seaborn 的 KDE 峰值与柱子高度的比例
fig2, ax = sns.histplot(x=vals, log_scale=True, kde=True, bins=20)
patches = ax.patches
heights = [p.get_height() for p in patches]
max_bar = max(heights)
kde_line = ax.lines[0]
max_kde = max(kde_line.get_ydata())
plt.close(fig2)

print(f"\n=== Seaborn 实际结果 ===")
print(f"Max bar height: {max_bar:.4f}")
print(f"Max KDE: {max_kde:.4f}")
print(f"Ratio: {max_kde/max_bar:.4f}")
