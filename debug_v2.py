"""深入调试 - 修正版"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

# 模拟 seaborn histplot 的实际行为
# 1. comp_data = log10(vals)
comp_data = np.log10(vals)

print("=== Data info ===")
print(f"Original data range: {vals.min():.4f} to {vals.max():.4f}")
print(f"Log data range: {comp_data.min():.4f} to {comp_data.max():.4f}")

# 2. Histogram bins - 这是关键！
# seaborn 默认使用 "auto" bins，它会根据原始数据的范围来确定
# 当 log_scale=True 时，bins 是在原始数据的 log 空间确定的

# 实际上，histplot 使用的是 comp_data 来确定 bins
# 因为 _define_bin_params 接收的是 comp_data
bins = 20
bin_edges_linear = np.histogram_bin_edges(vals, bins)
bin_edges_log = np.histogram_bin_edges(comp_data, bins)

print(f"\n=== Bin edges ===")
print(f"Linear space bin edges: {bin_edges_linear}")
print(f"Log space bin edges: {bin_edges_log}")

# 3. 直方图计算 - 在原始数据上计算 count
hist_counts, _ = np.histogram(vals, bins=bin_edges_linear)

# 4. 转换 bin 宽度到线性空间
# 这是关键步骤！
# edges 在 log 空间是均匀的，但转换到线性空间后不均匀
log_edges = np.log10(bin_edges_linear)
log_widths = np.diff(log_edges)
print(f"\n=== Widths ===")
print(f"Log widths: {log_widths[:5]}...")

linear_widths = []
for i in range(len(log_widths)):
    lo = np.power(10, log_edges[i])
    hi = np.power(10, log_edges[i+1])
    linear_widths.append(hi - lo)
linear_widths = np.array(linear_widths)
print(f"Linear widths: {linear_widths[:5]}...")

# 5. 验证直方图面积
hist_area = (hist_counts * linear_widths).sum()
print(f"\nHistogram area (heights * widths): {hist_area:.4f}")

# 6. 计算直方图密度
# density = count / (n * width) 
# 这是为了让直方图曲线下的面积等于 1
n = len(vals)
density = hist_counts / (n * linear_widths)
print(f"Density sum: {(density * linear_widths).sum():.4f}")

# 7. KDE - 在 log 空间计算
from scipy.stats import gaussian_kde
kde = gaussian_kde(comp_data)
log_support = np.linspace(comp_data.min(), comp_data.max(), 200)
density_log = kde(log_support)

# 8. 转换 KDE 到线性空间
linear_support = np.power(10, log_support)

# 9. KDE 积分验证
kde_integral_log = np.trapz(density_log, log_support)
print(f"\nKDE integral in log space: {kde_integral_log:.4f}")

# 10. 直接用直方图归一化因子缩放 KDE
# hist_norm = (heights * widths).sum() = count.sum() = n
hist_norm = hist_counts.sum()  # = n
density_scaled = density_log * hist_norm
kde_integral_scaled = np.trapz(density_scaled, linear_support)
print(f"KDE integral after scaling by n: {kde_integral_scaled:.4f}")

# 11. 应用 Jacobian 调整
jacobian = linear_support * np.log(10)
density_with_jac = density_log / jacobian * hist_norm
kde_integral_jac = np.trapz(density_with_jac, linear_support)
print(f"KDE integral after Jacobian: {kde_integral_jac:.4f}")

# 12. 对比
print(f"\n=== Peak comparison ===")
max_bar_idx = np.argmax(hist_counts)
print(f"Max bar: height={hist_counts[max_bar_idx]}, x={bin_edges_linear[max_bar_idx] + linear_widths[max_bar_idx]/2:.4f}")

max_kde_idx = np.argmax(density_scaled)
print(f"Max KDE (scaled): x={linear_support[max_kde_idx]:.4f}, y={density_scaled[max_kde_idx]:.4f}")

max_kde_jac_idx = np.argmax(density_with_jac)
print(f"Max KDE (with Jacobian): x={linear_support[max_kde_jac_idx]:.4f}, y={density_with_jac[max_kde_jac_idx]:.4f}")

# 计算峰值比例
print(f"\n=== Peak ratio ===")
print(f"KDE/bar (scaled): {density_scaled[max_kde_idx]/hist_counts[max_bar_idx]:.4f}")
print(f"KDE/bar (with Jacobian): {density_with_jac[max_kde_jac_idx]/hist_counts[max_bar_idx]:.4f}")
