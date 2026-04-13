"""验证修复效果"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

print("=" * 50)
print("Test 1: log_scale=True with lognormal data")
print("=" * 50)

fig, ax = plt.subplots()
sns.histplot(x=vals, log_scale=True, kde=True, bins=20, ax=ax)

patches = ax.patches
heights = [p.get_height() for p in patches]
max_bar = max(heights)

kde_line = ax.lines[0]
kde_y = kde_line.get_ydata()
max_kde = max(kde_y)

print(f"Max bar height: {max_bar:.4f}")
print(f"Max KDE: {max_kde:.4f}")
print(f"Ratio (KDE/bar): {max_kde/max_bar:.4f}")

if 0.8 <= max_kde/max_bar <= 1.2:
    print("✓ PASS: KDE peak is close to bar height")
else:
    print("✗ FAIL: KDE peak is too different from bar height")

plt.close()

print("\n" + "=" * 50)
print("Test 2: log_scale=False (reference)")
print("=" * 50)

fig2, ax2 = plt.subplots()
sns.histplot(x=vals, log_scale=False, kde=True, bins=20, ax=ax2)

patches2 = ax2.patches
heights2 = [p.get_height() for p in patches2]
max_bar2 = max(heights2)

kde_line2 = ax2.lines[0]
kde_y2 = kde_line2.get_ydata()
max_kde2 = max(kde_y2)

print(f"Max bar height: {max_bar2:.4f}")
print(f"Max KDE: {max_kde2:.4f}")
print(f"Ratio (KDE/bar): {max_kde2/max_bar2:.4f}")

plt.close()

print("\n" + "=" * 50)
print("Test 3: log_scale=True with exponential data")
print("=" * 50)

np.random.seed(42)
vals_exp = np.random.exponential(scale=1.0, size=1000)

fig3, ax3 = plt.subplots()
sns.histplot(x=vals_exp, log_scale=True, kde=True, bins=20, ax=ax3)

patches3 = ax3.patches
heights3 = [p.get_height() for p in patches3]
max_bar3 = max(heights3)

kde_line3 = ax3.lines[0]
kde_y3 = kde_line3.get_ydata()
max_kde3 = max(kde_y3)

print(f"Max bar height: {max_bar3:.4f}")
print(f"Max KDE: {max_kde3:.4f}")
print(f"Ratio (KDE/bar): {max_kde3/max_bar3:.4f}")

if 0.8 <= max_kde3/max_bar3 <= 1.2:
    print("✓ PASS: KDE peak is close to bar height")
else:
    print("✗ FAIL: KDE peak is too different from bar height")

plt.close()

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)
