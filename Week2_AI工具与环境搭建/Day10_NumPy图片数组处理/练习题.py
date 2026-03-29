#1.利用 NumPy 创建一个3 行 4 列的全 0 整数数组，
#再创建一个5 行 5 列的随机浮点数组（取值范围 0-255，模拟图片像素值）
#分别打印两个数组的形状（shape）和数据类型（dtype）。
import numpy as np

arr1 = np.array([0,0,0,0],[0,0,0,0],[0,0,0,0])
arr2 = np.random.randint(0,255,size=(5,5))

#===============打印结果================
print("======第一个数组======")
print(f"形状: {arr1.shape}")
print(f"数据类型: {arr1.dtype}")
print(f"数组内容:\n",arr1)

print("======第二个数组======")
print(f"形状: {arr2.shape}")
print(f"数据类型: {arr2.dtype}")
print(f"数组内容:\n",arr2)

print("="*50)
#2.手动创建一个一维 NumPy 数组arr = [128, 200, 56, 189, 255]，完成以下操作并打印结果：
#对数组所有元素做除以 255 的归一化运算（图片预处理常用操作）；
#计算数组的最大值、最小值、平均值；
#将数组转换为8 位无符号整数类型（uint8）（图片像素的标准数据类型）。

arr = np.array([128, 200, 56, 189, 255])
print("--- 原始数组 ---")
print(arr)

# 归一化运算：所有元素除以 255
# 将像素值映射到 0~1 之间
arr_normalized = arr / 255.0
print("\n--- 归一化后 (0~1) ---")
print(arr_normalized)

# 计算最大值、最小值、平均值
max_val = np.max(arr)
min_val = np.min(arr)
mean_val = np.mean(arr)

print(f"\n--- 统计结果 ---")
print(f"最大值: {max_val}")
print(f"最小值: {min_val}")
print(f"平均值: {mean_val:.2f}") # 保留2位小数，更直观

# 转换为 8位无符号整数类型 (uint8)
# 注：先把归一化后的小数转回 0~255 整数，然后再改 dtype
# 因为直接把 0.5 转 uint8 会变成 0，不符合常理
arr_uint8 = (arr_normalized * 255).astype(np.uint8)
# 或者直接用原数组计算：arr.astype(np.uint8) 效果一样
print("\n--- 转换为 uint8 类型 ---")
print(arr_uint8)