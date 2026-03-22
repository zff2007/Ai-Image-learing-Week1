# 导入PIL库
from PIL import Image

# 1. 打开本地图片（用 r 前缀避免转义错误）
img = Image.open(r"path")
img.show()

print("图片格式是：",img.format)
print("图片大小",img.size)
print("图片高度",img.height,"图片宽度",img.width)
print("获取（100，100）处的像素值：",img.getpixel((100,100)))
print("图片模式：",img.mode)   #RGB彩色，L灰度、RGBA带透明通道等
# 2. 保存图片（另存为一张新图片）
img.save("new_test.jpg")

print("图片打开并保存成功啦！")