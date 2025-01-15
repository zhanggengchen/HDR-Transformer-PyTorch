import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数，供绘图使用
def my_function(x):
    y = np.clip(x, -3, 4)
    y = np.round(y)
    return y

def my_function_1(x):
    y = x - np.floor(x)
    # y = ((np.tanh(y - 0.5) / np.tanh(0.5)) / 2) + 0.5
    y = ((np.tanh(3*(y - 0.5)) / np.tanh(3 * 0.5)) / 2) + 0.5
    return y + np.floor(x)

# 设置 x 的取值范围
x = np.linspace(-3, 4, 1000)  # 从 -10 到 10，取 1000 个点

# 计算 y 值
y = my_function_1(x)
z = my_function(x)

# 创建图像
plt.figure(figsize=(8, 6))

# 绘制曲线
plt.plot(x, y, label='y = sin(x)', color='green', linewidth=2)
plt.plot(x, z, label='y = sin(x)', color='blue', linewidth=2)

# 添加标题和坐标轴标签
plt.title("STE", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)


# 显示图像
plt.savefig("test.png", dpi=200)