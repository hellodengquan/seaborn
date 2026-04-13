"""复现 histplot log_scale=True 时 KDE 偏高的问题"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成对数正态分布数据
np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

# 创建对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图: log_scale=False
ax1 = axes[0]
sns.histplot(x=vals, log_scale=False, kde=True, bins=20, ax=ax1)
ax1.set_title('log_scale=False (线性轴)')

# 获取柱子高度和KDE峰值
patches = ax1.patches
heights = [p.get_height() for p in patches]
max_bar_height = max(heights)

kde_line = ax1.lines[0]
kde_y = kde_line.get_ydata()
max_kde = max(kde_y)

ax1.set_ylim(0, max(max_bar_height, max_kde) * 1.1)
ax1.axhline(y=max_bar_height, color='r', linestyle='--', label=f'Max bar height: {max_bar_height:.2f}')
ax1.axhline(y=max_kde, color='g', linestyle='--', label=f'Max KDE: {max_kde:.2f}')
ax1.legend()

print(f"=== log_scale=False ===")
print(f"Max bar height: {max_bar_height:.4f}")
print(f"Max KDE: {max_kde:.4f}")
print(f"Ratio (KDE/bar): {max_kde/max_bar_height:.4f}")

# 右图: log_scale=True
ax2 = axes[1]
sns.histplot(x=vals, log_scale=True, kde=True, bins=20, ax=ax2)
ax2.set_title('log_scale=True (对数轴)')

# 获取柱子高度和KDE峰值
patches = ax2.patches
heights = [p.get_height() for p in patches]
max_bar_height_log = max(heights)

kde_line = ax2.lines[0]
kde_y = kde_line.get_ydata()
max_kde_log = max(kde_y)

ax2.set_ylim(0, max(max_bar_height_log, max_kde_log) * 1.1)
ax2.axhline(y=max_bar_height_log, color='r', linestyle='--', label=f'Max bar height: {max_bar_height_log:.2f}')
ax2.axhline(y=max_kde_log, color='g', linestyle='--', label=f'Max KDE: {max_kde_log:.2f}')
ax2.legend()

print(f"\n=== log_scale=True ===")
print(f"Max bar height: {max_bar_height_log:.4f}")
print(f"Max KDE: {max_kde_log:.4f}")
print(f"Ratio (KDE/bar): {max_kde_log/max_bar_height_log:.4f}")

plt.tight_layout()
plt.savefig('/Users/dengquan/Downloads/job/bytedance/code/prod/dogfooding-3-425/kimi/kde_log_issue.png', dpi=150)
print(f"\n图片已保存到 kde_log_issue.png")

# 检查比例是否接近1
if max_kde_log/max_bar_height_log > 1.5:
    print("\n⚠️  问题确认: log_scale=True 时 KDE 峰值明显高于柱子高度!")
else:
    print("\n✓ KDE 和柱子高度比例正常")
