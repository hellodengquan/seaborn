"""测试 pandas 版本"""
import pandas as pd
print(f"Pandas version: {pd.__version__}")

# 检查哪个 option 存在
try:
    pd.option_context('mode.use_inf_as_null', True)
    print("mode.use_inf_as_null exists")
except:
    print("mode.use_inf_as_null does NOT exist")

try:
    pd.option_context('mode.use_inf_as_na', True)
    print("mode.use_inf_as_na exists")
except:
    print("mode.use_inf_as_na does NOT exist")
