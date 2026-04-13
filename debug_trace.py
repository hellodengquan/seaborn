"""带调试输出的测试"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Patch _compute_univariate_density to add debug output
original_compute = sns.distributions._DistributionPlotter._compute_univariate_density

def debug_compute(self, data_variable, common_norm, common_grid, estimate_kws, log_scale, warn_singular=True, log_base=None):
    print(f"\n=== _compute_univariate_density called ===")
    print(f"log_scale: {log_scale}")
    print(f"log_base: {log_base}")
    result = original_compute(self, data_variable, common_norm, common_grid, estimate_kws, log_scale, warn_singular, log_base)
    if log_scale and log_base:
        key = list(result.keys())[0] if result else None
        if key:
            density = result[key]
            print(f"After Jacobian correction, density range: {density.min():.6f} to {density.max():.6f}")
    return result

sns.distributions._DistributionPlotter._compute_univariate_density = debug_compute

# Patch plot_univariate_histogram to add debug output
original_plot = sns.distributions._DistributionPlotter.plot_univariate_histogram

def debug_plot(self, *args, **kwargs):
    log_base = self._get_log_scale_base(self.data_variable)
    print(f"\n=== plot_univariate_histogram ===")
    print(f"log_scale: {self._log_scaled(self.data_variable)}")
    print(f"log_base: {log_base}")
    return original_plot(self, *args, **kwargs)

sns.distributions._DistributionPlotter.plot_univariate_histogram = debug_plot

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
kde_y = kde_line.get_ydata()
max_kde = max(kde_y)

print(f"\n=== Final Results ===")
print(f"Max bar height: {max_bar:.4f}")
print(f"Max KDE: {max_kde:.4f}")
print(f"Ratio: {max_kde/max_bar:.4f}")

plt.close()
