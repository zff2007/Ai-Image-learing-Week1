# -*- coding: utf-8 -*-
"""
脚本功能：批量处理指定文件夹内的图片，将其统一转换为500*500的JPG格式
适用场景：图片AI开发助理实习打卡Day3必做产出
核心库：PIL（Pillow）用于图片处理，os用于文件路径操作
"""
from PIL import Image
import os

#================配置======================
#原始图片所在文件夹路径
SOURCE_DIR=r"path"
#处理后图片的保存文件夹名称
OUTPUT_DIR="processed_img"
#目标图片尺寸
TARGET_SIZE=(500,500)
#最多处理图片数量
MAX_IMAGES=5
#重命名前缀
RENAME_PREFIX="image"
#==================主逻辑=====================

def batch_process_images():
    """
    批量处理图片的主函数
    """
    #创造输出文件夹（如果不存在）
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    #初始化计数
    count=0
    #支持的所有图片格式
    support_formats=('.png','.jpg','.jpeg','.bmp','.gif','.webp')
    
    #遍历源文件夹
    for filename in os.listdir(SOURCE_DIR):
        #达到最大处理数量就停止循环
        if count >=MAX_IMAGES:
            break
        #拼接文件完整路径
        file_path=os.path.join(SOURCE_DIR,filename)
        #判断是否是文件及文件后缀是否满足
        if os.path.isfile(file_path) and filename.lower().endswith(support_formats):
            try:
                #安全打开图片，确保文件资源自动释放，不会导致图片被占用无法编辑
                with Image.open(file_path) as img:
                    #创建白色背景，等比例缩放
                    bg = Image.new("RGB", TARGET_SIZE, (255, 255, 255))
                    #等比例缩放，确保不变形
                    img.thumbnail(TARGET_SIZE,Image.Resampling.LANCZOS)
                    #计算画布居中放置
                    x=(TARGET_SIZE[0]-img.width)//2
                    y=(TARGET_SIZE[1]-img.height)//2
                    #将缩放后的图片粘贴到白色背景的中央
                    #mask参数用于处理PNG等透明图片，确保透明部分显示为白色背景
                    bg.paste(img,(x,y),mask=img.split()[-1] if img.mode=="RGBA" else None)
                    #重命名并保存
                    new_img= f"{RENAME_PREFIX}_{count + 1:03d}.jpg"
                    new_img_path=os.path.join(OUTPUT_DIR,new_img)
                    bg.save(new_img_path, "JPEG", quality=85)

                    print(f"处理成功：{filename}-->{new_img}")
                    count+=1
            #捕获所有可能异常，确保脚本不会因单个坏文件而中断
            except Exception as e:
                print(f"转换失败：{filename}，错误原因：{str(e)}")
        #处理完成后，打印总结消息
    print(f"\n 批量转换完成！共成功处理{count}张图片")
    print(f"处理后文件保存在：{os.path.abspath(OUTPUT_DIR)}")

#脚本入口，当脚本被直接运行时执行主函数
if __name__=="__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"错误:源文件夹'{SOURCE_DIR}'不存在，请检查路径是否正确")
    else:
        batch_process_images()