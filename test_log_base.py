"""测试如何获取 log 基数"""
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xscale('log')

# 获取 scale 对象
xaxis = ax.xaxis
print(f"Scale name: {xaxis.get_scale()}")

# 获取 log 基数
from matplotlib.scale import LogScale
scale = ax.xaxis._scale
print(f"Scale type: {type(scale)}")
if hasattr(scale, 'base'):
    print(f"Log base: {scale.base}")
else:
    print("No base attribute")

plt.close()
