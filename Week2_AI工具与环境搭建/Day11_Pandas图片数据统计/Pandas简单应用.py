import os
import pandas as pd

img_folder = r"C:\Users\sally\Desktop\ceshi"
#支持的图片格式
img_formats = [".jpg",".jpeg",".png",".bmp",".gif"]

#收集所有图片信息（存成列表）
img_data = []
for file_name in os.listdir(img_folder):
    file_path = os.path.join(img_folder,file_name)
    #分离文件名和后缀（格式）
    file_ext = os.path.splitext(file_name)[-1].lower()  #转小写，统一格式

    #只筛选图片文件夹
    if file_ext in img_formats:
        img_data.append({
            "文件名": file_name,
            "格式": file_ext,
            "完整路径": file_path
        })

#转成Pandas表格（DataFrame）
df = pd.DataFrame(img_data)

#================核心统计结果=====================
print("===== 图片数据集统计报告 =====")
print(f"图片总数量：{len(df)} 张")
print("\n各格式数量统计:")
print(df["格式"].value_counts())  # 最核心的统计函数

#打印前5条数据，查看表格
print("\n===== 前5条图片信息 =====")
print(df.head()) #"取头部数据",默认返回前5行

#================生成Excel表格==================
df.to_excel("图片数据集统计.xlsx",index=False)
print("\n Excel表格已生成:图片数据集统计.xlsx")