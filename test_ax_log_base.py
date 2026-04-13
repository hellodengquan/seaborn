"""测试从 axes 获取 log 基数"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
vals = np.random.lognormal(mean=0, sigma=1, size=1000)

fig, ax = plt.subplots()
ax.set_xscale('log')

# 尝试从 axes 获取 log 基数
xaxis = ax.xaxis
print(f"Scale: {xaxis.get_scale()}")
print(f"Scale base: {xaxis._scale.base}")

plt.close()
