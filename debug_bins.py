"""分析直方图 bin 的问题"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

fig, ax = plt.subplots()
sns.histplot(x=vals, log_scale=True, kde=True, bins=20, ax=ax)

# 分析 bin 宽度
patches = ax.patches
print("=== Bin Analysis (log_scale=True) ===")
for i, patch in enumerate(patches):
    x = patch.get_x()
    w = patch.get_width()
    h = patch.get_height()
    print(f"Bin {i}: x={x:.4f}, width={w:.4f}, height={h:.0f}, area={h*w:.2f}")

print(f"\nTotal area: {sum(p.get_height() * p.get_width() for p in patches):.4f}")

# 检查 log 空间的 bin 宽度
log_vals = np.log10(vals)
log_bins = np.logspace(0, 1, 21)  # 10^0 到 10^1
log_bin_widths = np.diff(log_bins)
print(f"\n=== Log space bin analysis ===")
print(f"Log bins: {log_bins[:5]}...")
print(f"Log bin widths: {log_bin_widths[:5]}...")

# 对于密度估计，直方图应该归一化为密度
# height * width = count
# density = count / (total_count * width)
# 所以 density * width = count / total_count

# 验证直方图密度
total_count = len(vals)
densities = [p.get_height() / (p.get_width() * total_count) for p in patches]
print(f"\nDensity at bin center:")
for i, patch in enumerate(patches[:5]):
    center = patch.get_x() + patch.get_width()/2
    print(f"  x={center:.4f}, density={densities[i]:.6f}")

# KDE 峰值位置
kde_line = ax.lines[0]
kde_x = kde_line.get_xdata()
kde_y = kde_line.get_ydata()
peak_idx = np.argmax(kde_y)
print(f"\nKDE peak at x={kde_x[peak_idx]:.4f}, y={kde_y[peak_idx]:.6f}")

plt.close()

# 对比 log_scale=False 的情况
fig2, ax2 = plt.subplots()
sns.histplot(x=vals, log_scale=False, kde=True, bins=20, ax=ax2)

patches2 = ax2.patches
print("\n=== Bin Analysis (log_scale=False) ===")
for i, patch in enumerate(patches2[:5]):
    x = patch.get_x()
    w = patch.get_width()
    h = patch.get_height()
    print(f"Bin {i}: x={x:.4f}, width={w:.4f}, height={h:.0f}, area={h*w:.2f}")

kde_line2 = ax2.lines[0]
kde_y2 = kde_line2.get_ydata()
print(f"\nMax KDE (log_scale=False): {max(kde_y2):.4f}")
print(f"Max bar (log_scale=False): {max(p.get_height() for p in patches2):.4f}")
print(f"Ratio: {max(kde_y2)/max(p.get_height() for p in patches2):.4f}")

plt.close()
