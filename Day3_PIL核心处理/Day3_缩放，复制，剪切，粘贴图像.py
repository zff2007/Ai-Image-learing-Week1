from PIL import Image
import time #导入延时工具

# 1. 打开原始图片（替换为你的实际路径）
img = Image.open(r"path")

# 2. 等比例缩放（核心需求）
target_width = 500
scale = target_width / img.width
target_height = int(img.height * scale)
#Image.LANCZOS是PIL中的高清图像缩放算法
scaled_img = img.resize((target_width, target_height), Image.LANCZOS)  # 改名避免混淆

# 3. 复制原图 + 剪切 + 粘贴（基于原图操作，不影响缩放图）
img_copy = img.copy()
area = (100, 100, 400, 400)  # 剪切区域
crop_img = img_copy.crop(area)
img_copy.paste(crop_img, (50, 500))  # 粘贴到副本

# 4. 保存/显示结果（分两种：保存缩放图 / 保存粘贴后的原图）
# 保存并显示等比例缩放图
scaled_img.save("缩放效果.jpg")
scaled_img.show()

time.sleep(1)
# 保存并显示粘贴后的原图（可选，根据需求保留）
img_copy.save("粘贴效果.jpg")
img_copy.show()