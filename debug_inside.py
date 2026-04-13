"""添加内部调试输出来验证代码逻辑"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.distributions import _DistributionPlotter

# Patch to add debug output
original_plot = _DistributionPlotter.plot_univariate_histogram

def debug_plot(self, *args, **kwargs):
    # 打印 widths 和 heights
    print("\n=== Inside plot_univariate_histogram ===")
    
    # 临时存储变量用于调试
    self._debug_info = {}
    
    result = original_plot(self, *args, **kwargs)
    
    return result

# Apply patch
_DistributionPlotter.plot_univariate_histogram = debug_plot

np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

fig, ax = plt.subplots()
sns.histplot(x=vals, log_scale=True, kde=True, bins=20, ax=ax)

patches = ax.patches
heights = [p.get_height() for p in patches]
max_bar = max(heights)

kde_line = ax.lines[0]
kde_y = kde_line.get_ydata()
max_kde = max(kde_y)

print(f"\n=== Final Results ===")
print(f"Max bar: {max_bar:.4f}")
print(f"Max KDE: {max_kde:.4f}")
print(f"Ratio: {max_kde/max_bar:.4f}")

plt.close()
