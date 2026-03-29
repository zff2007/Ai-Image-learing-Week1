# GitHub第一次推送验证代码
# 功能：简单的PIL库验证（Day2基础），用于测试远程推送成功后的代码可运行性
# 运行前提：已安装PIL库（pip install pillow）

from PIL import Image
import os

# 简单图片操作：创建空白图片并保存（无需本地图片，直接运行）
def create_test_image():
    """创建一张300*300的空白白色图片，用于验证代码"""
    # 创建空白图片：RGB模式，尺寸300*300，白色背景
    img = Image.new(mode="RGB", size=(300, 300), color=(255, 255, 255))
    # 保存图片到当前文件夹
    img.save("git_push_test_image.jpg")
    print("测试图片创建成功！")
    print("当前文件夹文件：", os.listdir("."))

# 主程序执行
if __name__ == "__main__":
    try:
        create_test_image()
    except Exception as e:
        print(f"运行报错：{e}")