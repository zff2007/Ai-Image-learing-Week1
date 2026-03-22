from PIL import Image

img=Image.open(r"path")

#1.旋转rotate(角度,expand=True)
#角度：整数=逆时针，负数=顺时针
#expand=True：自动扩展画布，不裁剪图片
img_rotate=img.rotate(45,expand=True)
img_rotate=img.rotate(-90,expand=True)
img_rotate.save("旋转图.jpg")

#2.分离(分离RGB三个通道)
r,g,b=img.split()#分别存入变量r,g,b

#3.合并

#通道合并
merge_normal=Image.merge("RGB",(r,g,b))#正常合并：还原彩色图
merge_special=Image.merge("RGB",(g,r,b))#特效合并：交换红绿通道
merge_normal.save("通道合并——正常.jpg")
merge_special.save("通道合并——特效.jpg")

#拼接合并
img1=img
img2=img_rotate
#计算拼接后尺寸
width=img1.width + img2.width
height=max(img1.height,img2.height)
#创建新画布
img_concat=Image.new("RGB",(width,height))
#粘贴两张图
img_concat.paste(img,(0,0))
img_concat.paste(img2,(img1.width,0))
img_concat.save("横向拼接图.jpg")

