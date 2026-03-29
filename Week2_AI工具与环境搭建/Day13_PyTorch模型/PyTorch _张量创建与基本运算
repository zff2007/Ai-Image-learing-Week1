import torch
import numpy as np
from PIL import Image

#==============================================
#一、环境验证（前置准备）
#==============================================
def check_environment ():
    #"""验证 PyTorch 环境与设备信息"""
    print ("="*50)
    print ("PyTorch 版本:", torch.version)
    print ("默认计算设备:", torch.device ("cpu"))
    print ("="*50)

#执行环境检查
check_environment()

#==============================================
#二、张量的创建
#==============================================

print ("\n【二、张量创建】")

#1. 直接创建（从列表 / 标量创建）
print ("\n1. 直接创建张量")
#0 维张量（标量）
t0 = torch.tensor(3.14, dtype=torch.float32, device="cpu")
#1 维张量（向量）
t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
#2 维张量（矩阵）
t2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
#3 维张量（批次数据，常用于图像）
t3 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print ("0 维标量:", t0, "形状:", t0.shape)
print ("1 维向量:", t1, "形状:", t1.shape)
print ("2 维矩阵:", t2, "形状:", t2.shape)
print ("3 维张量:", t3.shape)

#2. NumPy 数组转换（共享内存）
print ("\n2. NumPy 与张量转换")
np_arr = np.array ([[1, 2], [3, 4]], dtype=np.float32)
t = torch.from_numpy (np_arr) # 浅拷贝，共享内存
#验证内存共享特性
t [0, 0] = 100
print ("修改张量后:\n 张量:", t, "\nNumPy 数组:", np_arr)

#3. 创建特殊值张量
print ("\n3. 特殊张量创建")
t_zeros = torch.zeros ((2, 3), dtype=torch.float32) # 全 0
t_ones = torch.ones ((3, 3), dtype=torch.int64) # 全 1
t_eye = torch.eye (4) # 单位矩阵
t_rand = torch.rand ((2, 2)) # 0~1 随机
t_randint = torch.randint (low=0, high=10, size=(2, 3)) # 随机整数
print ("全 0 张量:\n", t_zeros)
print ("全 1 张量:\n", t_ones)
print ("单位矩阵:\n", t_eye)

#4. 按现有张量形状创建（*like 系列）
print ("\n4. 同形状张量创建")
base_tensor = torch.tensor ([[1, 2], [3, 4]])
t_zeros_like = torch.zeros_like (base_tensor)
t_ones_like = torch.ones_like (base_tensor)
print ("原张量形状:", base_tensor.shape)
print ("同形状全 0:\n", t_zeros_like)

#==============================================
#三、张量核心属性查看
#==============================================

print ("\n【三、张量属性】")
test_tensor = torch.tensor ([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print ("形状 (shape):", test_tensor.shape)
print ("数据类型 (dtype):", test_tensor.dtype)
print ("设备 (device):", test_tensor.device)
print ("维度数 (ndim):", test_tensor.ndim)
print ("元素总数 (numel):", test_tensor.numel ())

#==============================================
#四、张量核心运算
#==============================================

print ("\n【四、张量运算】")

#1. 矩阵乘法
print ("\n1. 矩阵乘法")
a = torch.tensor ([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
b = torch.tensor ([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
matmul_res = torch.matmul (a, b) # 通用矩阵乘法
at_res = a @ b # 简洁写法
mm_res = torch.mm (a, b) # 仅限 2 维矩阵
print ("乘法结果形状:", matmul_res.shape)

#2. 索引与切片（图像数据常用）
print ("\n2. 索引与切片")
img_batch = torch.tensor ([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
img_1 = img_batch [0] # 取第一个样本
img_val = img_batch [1, 1, 2] # 取指定位置元素
img_row = img_batch [:, 0, :] # 取所有样本的第一行
print ("切片结果:\n", img_row)

#3. 形状变换
print ("\n3. 形状变换")
#重塑形状
t = torch.tensor ([[1,2,3],[4,5,6]])
reshape1 = t.reshape (3, 2)
reshape2 = t.reshape (-1, 1) # -1 = 自动计算维度
#维度重排（CNN 图像必备：HWC → CHW）
hwc_tensor = torch.randn (1, 2, 3, 3) # (batch, H, W, C)
chw_tensor = hwc_tensor.permute (0, 3, 1, 2) # (batch, C, H, W)
print ("维度重排:", hwc_tensor.shape, "→", chw_tensor.shape)
#维度增删（单张图→批次）
single_img = torch.randn (3, 2, 3)
batch_img = single_img.unsqueeze (0) # 增加批次维度
squeeze_img = batch_img.squeeze (0) # 删除维度

#4. 数学统计运算
print ("\n4. 统计运算")
t = torch.tensor ([[1,2,3],[4,5,6]], dtype=torch.float32)
print ("总和:", t.sum ().item ())
print ("均值:", round (t.mean ().item (), 2))
print ("最大值:", t.max ().item ())
#数据归一化（图像预处理必备）
norm_tensor = (t - t.min()) / (t.max() - t.min())

#==============================================
#五、张量 ↔ NumPy ↔ PIL 图像互转
#==============================================

print ("\n【五、数据格式转换】")
def tensor_image_conversion ():

    #模拟图像数据转换流程

    #方案 1：模拟图像数据（无本地文件，推荐运行）
    np_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    #方案 2：真实图片读取（取消注释即可使用）
    img = Image.open("test.jpg").convert("RGB")
    np_img = np.array(img)
    #NumPy → PyTorch 张量
    tensor_img = torch.from_numpy(np_img).float()
    #张量 → NumPy
    np_img_back = tensor_img.numpy()
    #标量张量 → Python 数值
    scalar = torch.tensor(255.0).item()
    print ("图像数组形状:", np_img.shape)
    print ("张量形状:", tensor_img.shape)
    print ("标量转换结果:", scalar, type (scalar))

tensor_image_conversion()
print ("\n" + "="*50)
print ("PyTorch 张量基础操作执行完成！")
print ("="*50)