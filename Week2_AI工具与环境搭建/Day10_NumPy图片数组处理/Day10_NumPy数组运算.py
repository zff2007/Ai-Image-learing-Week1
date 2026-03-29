import numpy as np
from PIL import Image

#=====================基础配置=========================
# 原始图片路径
IMG_PATH = r"path1"
# 保存图片的文件夹
SAVE_DIR = r"path2"

#======================1.图片转换为NumPy数组=====================

img = Image.open(IMG_PATH)
#转为NumPy像素数组（3维：高×宽×3通道RGB）
img_np = np.array(img)

print("=" * 50)
print("图片形状：",img_np.shape)
print("=" * 50)

#====================2.标量运算:调整亮度/对比度===================

#-------------加法/减法：调整图片亮度------------- 
# 亮度提升：所有像素值+30（数值越接近255，图片越亮）
img_bright = img_np + 30
# 亮度降低： 所有像素值-30（数值越接近0，图片越暗）
img_dark = img_np - 30

# 像素值必须限定在0-255（超出范围就会失真，转成uint8自动截断）
img_bright = np.clip(img_bright,0,255).astype(np.uint8)
img_dark = np.clip(img_dark,0,255).astype(np.uint8)

Image.fromarray(img_bright).save(f"{SAVE_DIR}\\亮度提升.jpg")
Image.fromarray(img_dark).save(f"{SAVE_DIR}\\亮度降低.jpg")

#-------------乘法/除法：调整图片对比度------------
# 对比度提升：所有像素值*1.5（像素值差距变大，对比更明显）
img_contrast_up = img_np * 1.5
# 对比度降低：所有像素值/1.5（像素值差距变小，画面更柔和）
img_contrast_down = img_np / 1.5

#同样限定0-255，转格式
img_contrast_up = np.clip(img_contrast_up,0,255).astype(np.uint8)
img_contrast_down = np.clip(img_contrast_down,0,255).astype(np.uint8)

Image.fromarray(img_contrast_up).save(f"{SAVE_DIR}\\对比度提升.jpg")
Image.fromarray(img_contrast_down).save(f"{SAVE_DIR}\\对比度降低.jpg")

#======================3.索引切片：图片裁剪========================

#先获取图片的尺寸（shape中提取，不用手动数）
h,w,_ = img_np.shape # _ 表示忽略通道数，只取高和宽

#裁剪中间1/2的区域（eg：高度从h/4到3h/4，宽从w/4到3w/4）
start_h,end_h = int(h/4),int(3*h/4)
start_w,end_w = int(w/4),int(3*w/4)
img_crop = img_np[start_h:end_h,start_w:end_w, :]

#保存裁剪后的图片
Image.fromarray(img_crop).save(f"{SAVE_DIR}\\裁剪后的图片.jpg")
print(f"裁剪后数组形状：{img_crop.shape}")  # 查看裁剪后的尺寸

#=====================4.形状操作：维度变换=========================

#1.reshape()：重塑数组形状（展平/拉伸，不改变数据）
# 把3维图片数组展平为1维（所有像素组值排列成一列）
img_flat = img_np.reshape(-1)  #-1表示让NumPy自动计算长度
print(f"展平后数组形状：{img_flat.shape}")  #结果是 h*w*3的值

# 把1维还原为3维（必须和原形状一致）
img_restore = img_flat.reshape(h,w,3)
Image.fromarray(img_restore).save(f"{SAVE_DIR}\\还原后的图片.jpg")

#2.transpose()：调整维度顺序
#NumPy默认维度：（高度h，宽度w，通道c）
#转成PyTorch格式：（通道c，高度h，宽度w）
img_np_t =img_np.transpose(2,0,1) #2=通道，0=高度，1=宽度
print(f"转置后数组形状：{img_np_t.shape}") 

#再转回来，恢复成图片格式
#把原来的第1位→放到第0位;把原来的第2位→放到第1位;把原来的第0位→放到第2位
img_np_back = img_np_t.transpose(1,2,0)
Image.fromarray(img_np_back).save(f"{SAVE_DIR}\\转置还原.jpg")

#======================5.分离通道+灰度图==========================

#1.分离RGB通道
img_red = img_np[:,:,0] #2维数组，灰度图格式
img_green = img_np[:,:,1]
img_blue = img_np[:,:,2]

#转3维彩图格式（单通道转3通道，让图片显示为对应颜色）
img_red_3d = np.stack([img_red,np.zeros_like(img_red),np.zeros_like(img_red)],axis=2)
img_green_3d = np.stack([np.zeros_like(img_green), img_green, np.zeros_like(img_green)], axis=2)
img_blue_3d = np.stack([np.zeros_like(img_blue), np.zeros_like(img_blue), img_blue], axis=2)

# 保存单通道图片
Image.fromarray(img_red_3d).save(f"{SAVE_DIR}\\红色通道.jpg")
Image.fromarray(img_green_3d).save(f"{SAVE_DIR}\\绿色通道.jpg")
Image.fromarray(img_blue_3d).save(f"{SAVE_DIR}\\蓝色通道.jpg")

#2.灰度图转换
#灰度图公式：Gray = 0.299*R +0.587*G + 0.114* B
img_gray = 0.299 * img_red + 0.587 * img_green + 0.114 * img_blue
#转uint8格式，保存灰度图
img_gray = img_gray.astype(np.uint8)
Image.fromarray(img_gray).save(f"{SAVE_DIR}\\灰度图.jpg")

#=========================最终验证：数组转回图片=====================
img_new = Image.fromarray(img_np)
img_new.save(f"{SAVE_DIR}\\new.jpg")
print("转换完成！")
