"""调试 log_base 传递"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成对数正态分布数据
np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

# 创建 figure 和 axes
fig, ax = plt.subplots()

# 调用 histplot
result = sns.histplot(x=vals, log_scale=True, kde=True, bins=20, ax=ax)

# 检查 axes 的 log scale
print(f"Axes x scale: {ax.get_xscale()}")
print(f"X axis scale: {ax.xaxis.get_scale()}")

# 获取 scale 对象
scale = ax.xaxis._scale
print(f"Scale type: {type(scale)}")
print(f"Scale base: {scale.base if hasattr(scale, 'base') else 'N/A'}")

# 打印 KDE 数据
kde_line = ax.lines[0]
kde_x = kde_line.get_xdata()
kde_y = kde_line.get_ydata()
print(f"\nKDE x range: {kde_x.min():.4f} to {kde_x.max():.4f}")
print(f"KDE y range: {kde_y.min():.4f} to {kde_y.max():.4f}")

# 获取直方图柱子
patches = ax.patches
heights = [p.get_height() for p in patches]
print(f"\nMax bar height: {max(heights):.4f}")
print(f"Max KDE: {max(kde_y):.4f}")
print(f"Ratio: {max(kde_y)/max(heights):.4f}")

plt.close()
