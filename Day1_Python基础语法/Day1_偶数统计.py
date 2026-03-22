# -*- coding: utf-8 -*-
"""
Day1必做产出：统计1-100内的偶数个数（进阶版，含列表存储）
核心知识点：Python变量、for循环、if判断、列表定义与追加
适用：图片AI开发助理实习打卡第1周Day1
"""

# 定义变量统计偶数个数
even_count= 0
# 定义空列表，用于存储1-100内所有的偶数（贴合Day1列表知识点）
even_list= []

# 遍历1-100的数字
for x in range(1,101):
    if x%2 == 0:
        even_count +=1  # 偶数计数+1
        even_list.append(x)  # 将偶数添加到列表中

# 打印结果
print(f"1-100内的偶数总个数为：{even_count}")
print(f"1-100内的所有偶数为：{even_list}")