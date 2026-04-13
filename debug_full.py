"""带详细调试输出的测试"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Patch _compute_univariate_density
original_compute = sns.distributions._DistributionPlotter._compute_univariate_density

def debug_compute(self, data_variable, common_norm, common_grid, estimate_kws, log_scale, warn_singular=True, log_base=None):
    result = original_compute(self, data_variable, common_norm, common_grid, estimate_kws, log_scale, warn_singular, log_base)
    if log_scale and log_base and result:
        key = list(result.keys())[0]
        density = result[key]
        print(f"\n[_compute_univariate_density] After Jacobian correction:")
        print(f"  log_base: {log_base}")
        print(f"  density sum (integral): {np.trapz(density.values, density.index):.4f}")
        print(f"  density range: {density.min():.6f} to {density.max():.6f}")
    return result

sns.distributions._DistributionPlotter._compute_univariate_density = debug_compute

# 测试
np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

fig, ax = plt.subplots()
sns.histplot(x=vals, log_scale=True, kde=True, bins=20, ax=ax)

# 获取结果
patches = ax.patches
heights = [p.get_height() for p in patches]
max_bar = max(heights)

kde_line = ax.lines[0]
kde_x = kde_line.get_xdata()
kde_y = kde_line.get_ydata()
max_kde = max(kde_y)

print(f"\n=== Final Results ===")
print(f"Max bar height: {max_bar:.4f}")
print(f"Max KDE: {max_kde:.4f}")
print(f"Ratio: {max_kde/max_bar:.4f}")

# 验证 KDE 积分是否匹配直方图面积
# KDE 曲线下的面积应该约等于直方图的总面积
hist_sum = sum(p.get_height() * p.get_width() for p in patches)
kde_integral = np.trapz(kde_y, kde_x)
print(f"\nHistogram sum (areas): {hist_sum:.4f}")
print(f"KDE integral: {kde_integral:.4f}")
print(f"Ratio (KDE/hist): {kde_integral/hist_sum:.4f}")

plt.close()
