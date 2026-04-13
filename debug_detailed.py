"""深入调试 KDE 缩放问题"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn._stats.counting import Hist
from seaborn._statistics import KDE

np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

# 模拟 seaborn 的内部计算
# 1. comp_data = log10(vals)
comp_data = np.log10(vals)

print("=== Data info ===")
print(f"Original data range: {vals.min():.4f} to {vals.max():.4f}")
print(f"Log data range: {comp_data.min():.4f} to {comp_data.max():.4f}")

# 2. Histogram bins 在 log 空间
hist_est = Hist(bins=20)
bin_kws = hist_est._define_bin_params({"x": pd.Series(vals)}, "x", None)
res = hist_est._normalize(hist_est._eval({"x": pd.Series(vals)}, "x", bin_kws))
heights = res["count"].to_numpy()
widths_log = res["space"].to_numpy()  # log 空间的宽度
edges_log = res["x"].to_numpy() - widths_log / 2  # log 空间的边缘

print(f"\n=== Histogram in log space ===")
print(f"Bin edges (log): {edges_log[:5]}")
print(f"Bin widths (log): {widths_log[:5]}")

# 3. 转换到线性空间
edges_linear = np.power(10, edges_log)
widths_linear = np.power(10, edges_log + widths_log) - np.power(10, edges_log)

print(f"\n=== Histogram in linear space ===")
print(f"Bin edges (linear): {edges_linear[:5]}")
print(f"Bin widths (linear): {widths_linear[:5]}")

# 4. Histogram 的归一化因子
hist_norm = (heights * widths_linear).sum()
print(f"\nHistogram norm (heights * widths_linear.sum()): {hist_norm:.4f}")

# 5. KDE 计算（使用 comp_data = log10(vals)）
kde_est = KDE(gridsize=200, cut=0)
density_log, support_log = kde_est(comp_data)

# 6. 转换 KDE 到线性空间（没有 Jacobian 调整）
support_linear = np.power(10, support_log)

print(f"\n=== KDE (no Jacobian) ===")
print(f"Support range (log): {support_log.min():.4f} to {support_log.max():.4f}")
print(f"Support range (linear): {support_linear.min():.4f} to {support_linear.max():.4f}")
print(f"Density range (before scaling): {density_log.min():.6f} to {density_log.max():.6f}")

# 7. 用直方图归一化 KDE（没有 Jacobian）
density_scaled = density_log * hist_norm
print(f"Density range (after hist scaling): {density_scaled.min():.4f} to {density_scaled.max():.4f}")

# 8. 用 Jacobian 调整后的 KDE
jacobian = support_linear * np.log(10)
density_with_jac = density_log / jacobian * hist_norm
print(f"\n=== KDE (with Jacobian) ===")
print(f"Density range (with Jacobian + hist scaling): {density_with_jac.min():.6f} to {density_with_jac.max():.6f}")

# 9. 找到峰值位置
max_bar_idx = np.argmax(heights)
max_bar_height = heights[max_bar_idx]
max_bar_x = edges_linear[max_bar_idx] + widths_linear[max_bar_idx]/2
print(f"\nHistogram peak: x={max_bar_x:.4f}, height={max_bar_height:.0f}")

max_kde_no_jac_idx = np.argmax(density_scaled)
print(f"KDE peak (no Jacobian): x={support_linear[max_kde_no_jac_idx]:.4f}, y={density_scaled[max_kde_no_jac_idx]:.4f}")

max_kde_jac_idx = np.argmax(density_with_jac)
print(f"KDE peak (with Jacobian): x={support_linear[max_kde_jac_idx]:.4f}, y={density_with_jac[max_kde_jac_idx]:.4f}")

# 验证问题：KDE 和直方图的面积
print("\n=== Area verification ===")
kde_integral_no_jac = np.trapz(density_scaled, support_linear)
kde_integral_jac = np.trapz(density_with_jac, support_linear)
hist_area = hist_norm

print(f"Histogram area: {hist_area:.4f}")
print(f"KDE area (no Jacobian): {kde_integral_no_jac:.4f}")
print(f"KDE area (with Jacobian): {kde_integral_jac:.4f}")
