"""验证当前代码状态"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

# 测试 log_scale=True
fig, ax = plt.subplots()
sns.histplot(x=vals, log_scale=True, kde=True, bins=20, ax=ax)

patches = ax.patches
heights = [p.get_height() for p in patches]
max_bar = max(heights)

kde_line = ax.lines[0]
kde_y = kde_line.get_ydata()
max_kde = max(kde_y)

print(f"=== Current code state (log_scale=True) ===")
print(f"Max bar: {max_bar:.4f}")
print(f"Max KDE: {max_kde:.4f}")
print(f"Ratio: {max_kde/max_bar:.4f}")

plt.close()

# 测试 log_scale=False 作为参考
fig2, ax2 = plt.subplots()
sns.histplot(x=vals, log_scale=False, kde=True, bins=20, ax=ax2)

patches2 = ax2.patches
heights2 = [p.get_height() for p in patches2]
max_bar2 = max(heights2)

kde_line2 = ax2.lines[0]
kde_y2 = kde_line2.get_ydata()
max_kde2 = max(kde_y2)

print(f"\n=== Reference (log_scale=False) ===")
print(f"Max bar: {max_bar2:.4f}")
print(f"Max KDE: {max_kde2:.4f}")
print(f"Ratio: {max_kde2/max_bar2:.4f}")
