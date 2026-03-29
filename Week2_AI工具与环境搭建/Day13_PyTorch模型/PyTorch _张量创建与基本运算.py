import torch

#一.前置准备

#验证PyTorch版本，验证导入成功
#print(torch.__version__)
#查看当前设备
#print(torch.device("cpu"))

#二.张量的创建

#1.直接创建：从列表/元组创建
#创建0维张量(标量)
t0 = torch.tensor(3.14,dtype=torch.float32,device="cpu")
#创建1维向量（向量）
t1 = torch.tensor([1,2,3,4],dtype=torch.int64)
#创建2维向量（矩阵）
t2 = torch.tensor([[1,2],[3,4]],dtype=torch.float32)
#创建3维向量（eg:2个2*3的矩阵，贴合图片批次基础形态）
t3 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

#打印张量及属性
#print("0维张量:",t0,"形状：",t0.shape)
#print("1维张量:",t1,"形状：",t1.shape)
#print("2维张量:",t2,"形状：",t2.shape)
#print("3维张量:",t3,"形状：",t3.shape)

#2.从NumPy数组转换
import numpy as np
#创建Numpy数组
np_arr = np.array([[1,2],[3,4]],dtype=np.float32)
#转换为PyTorch张量
#torch.from_numpy(np_arr) 共享内存，相当于给同一份数据起了两个名字
t = torch.from_numpy(np_arr)

#print("NumPy数组:",np_arr)
#print("转换后的张量：",t)

#验证内存共享
t[0,0] = 100
#print("修改后张量:",t)
#print("同步修改的NumPy数组:",np_arr)

#3.创建特殊张量
#1）.全0张量：torch.zeros(shape)
t_zeros = torch.zeros((2,3),dtype=torch.float32) #2*3的全0矩阵
#2）.全1张量：torch.ones(shape)
t_ones = torch.ones((3,3),dtype=torch.int64)     #3*3的全1矩阵
#3）.单位张量(对角为1，其余为0)：torch.eye(n)（n*n矩阵）
t_eye = torch.eye(4)                             #4*4的单位矩阵
#4）.随机张量（0-1均匀分布）：torch.rand(shape)
t_rand = torch.rand((2,2))                       #2*2的0-1的随机张量
#5）.随机整数张量（指定范围）:torch.randint(low,high,shape)
t_randint = torch.randint(0,10,(2,3))            #2*3的0-9的随机整数张量

print("全0张量:\n",t_zeros)
print("全1张量:\n",t_ones)
print("单位张量:\n",t_eye)
print("0-1随机张量:\n",t_rand)
print("随机整数张量:\n",t_randint)

#4.按已有张量形状创建（*like方法）
#已有张量（如图片转换后的张量）
t = torch.tensor([[1,2],[3,4]])
#创建与t形状相同的全0张量
t_zeros_like = torch.zeros_like(t)
#创建与t形状相同的全1张量
t_ones_like = torch.ones_like(t)

print("原张量：\n", t, "形状：", t.shape)
print("同形状全0张量:\n", t_zeros_like)
print("同形状全1张量:\n", t_ones_like)

#三.张量的基本属性查看

t = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, device='cpu')
# 1. 形状：tensor.shape / tensor.size()
print("形状：", t.shape)  # 推荐使用shape，更直观
# 2. 数据类型：tensor.dtype
print("数据类型：", t.dtype)
# 3. 设备：tensor.device（CPU版固定为cpu）
print("设备：", t.device)
# 4. 维度：tensor.ndim
print("维度：", t.ndim)
# 5. 元素总数：tensor.numel()
print("元素总数：", t.numel())

#四.基本运算

#1.矩阵乘法：第一个张量的列数 = 第二个张量的行数
#定义符合矩阵乘法规则的张量（2*3和3*2）
a = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)
b = torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.float32)

#法1：torch.matmul()
mat1 = torch.matmul(a,b)
#法2：@ 运算符
mat2 = a@b
#法3：torch.mm()(仅支持2维矩阵)
mat3 = torch.mm(a,b)

#2.索引切片
#2张照片，每2*3的像素矩阵
t = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print("原3维张量:\n",t,"形状：",t.shape)

#1).单维度索引：提取第1张图片（索引从0开始）
t1 = t[0]
print("第1张图片：\n", t1, "形状：", t1.shape)
#2).多维度索引：提取第2张照片的第2行第3列元素
t2 = t[1,1,2]
print("第2张图片第2行第3列元素：", t2)
#3).切片：提取所有图片的第一行所有列
t3 = t[:,0,:]
print("所有图片的第1行：\n", t3, "形状：", t3.shape)

#4).步长切片：提取第一张图片的所有行，按步长2取列
t4 = t[0,:,::2]
print("第1张图片按步长2取列：\n", t4)

#3.形状变换

#1）reshape()：灵活调整形状（无维度顺序变化）
t = torch.tensor([[1,2,3],[4,5,6]])
# 变换为3×2的张量
t5 = t.reshape(3,2)
# 用-1自动计算维度：变换为6×1，-1表示自动计算该维度为6
t6 = t.reshape(-1,1)

#2) permute()：维度重排（图片通道转换）
#定义4维张量（模拟：1张图片，批次=1，高=2，宽=3，通道=3）
#形状：(batch,H,W,C) = (1,2,3,3)
t = torch.randn(1,2,3,3)
print("原张量形状(batch,H,W,C):",t.shape)

#维度重排为：(batch,C,H,W)(CNN模型输入格式)
# permute内的参数为原维度的索引：0=batch，3=C，1=H，2=W
t_permute = t.permute(0,3,1,2)
print("重排后形状（batch,C,H,W）：", t_permute.shape)  # (1,3,2,3)

#）squeeze()/unsqueeze()：维度增删（批量处理）
#squeeze(dim)：删除维度为 1的维度，减少冗余维度
#unsqueeze(dim)：增加维度为 1的维度，用于添加批次 / 通道维度（如单张图片(C,H,W)转为批次(1,C,H,W)）
# 定义3维张量（C,H,W）=(3,2,3)（单张图片）
t = torch.randn(3,2,3)
print("原张量形状（C,H,W）：", t.shape)

# unsqueeze：在第0维增加批次维度，变为(1,3,2,3)
t_unsq = t.unsqueeze(0)
print("增加批次维度后：", t_unsq.shape)

# squeeze：删除第0维的批次维度，恢复为(3,2,3)
t_sq = t_unsq.squeeze(0)
print("删除批次维度后：", t_sq.shape)

#4.其他基础运算
# 定义张量（模拟图片像素数据）
t = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)

# 1. 求和：tensor.sum()，可指定求和维度（dim）
sum_all = t.sum()  # 所有元素求和
sum_dim0 = t.sum(dim=0)  # 按第0维求和（列求和）
# 2. 均值：tensor.mean()
mean = t.mean()
# 3. 最大值/最小值：tensor.max() / tensor.min()
max_val = t.max()
min_val = t.min()
# 4. 归一化（图片预处理核心：将像素值缩放到0-1）
t_norm = (t - t.min()) / (t.max() - t.min())


#五张量与 NumPy 的互转
import torch
import numpy as np
from PIL import Image
import os

# 1. 模拟PIL读取图片并转为NumPy数组（Day10内容）
img = Image.open("test.jpg").convert("RGB")  # 读取图片
np_img = np.array(img)  # 转为NumPy数组，形状：(H,W,C)
print("NumPy图片数组形状:", np_img.shape)

# 2. NumPy数组转PyTorch张量（CPU）
tensor_img = torch.from_numpy(np_img).float()  # 转为浮点型（模型要求）
print("张量形状(H,W,C):", tensor_img.shape)

# 3. 张量转NumPy数组
np_img2 = tensor_img.numpy()
# 若张量为标量，用tensor.item()转为Python数值
tensor_scalar = torch.tensor(255.0)
py_scalar = tensor_scalar.item()

print("张量转回NumPy的形状:", np_img2.shape)
print("张量标量转Python数值:", py_scalar, type(py_scalar))