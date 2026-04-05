import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import tensorflow as tf

#====================PyTorch 图片预处理（ResNet标准流程）=====================

#1.定义ResNet要求的预处理管道（尺寸调整→张量转换→归一化）
preprocess = transforms.Compose([
    transforms.Resize(256),                 #尺寸调整：短边缩到256，长边按比例自动调整
    transforms.CenterCrop(224),             #中心裁剪到224*224（模式固定输入）
    transforms.ToTensor(),                  #张量转换：图片→PyTorch张量
    transforms.Normalize(                   #归一化：ImadeNet官方均值/标准差
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

#2.读取本地图片（替换成自己的图片路径）
img = Image.open(r"path").convert("RGB") #统一转换为RGB，避免灰度图报错

#3.执行预处理→得到模型可识别张量
img_tensor_torch = preprocess(img)          #返回变换Normalize输出的张量
#增加批次维度（模型必须要：[batch,channel,height,width]）
img_tensor_torch = img_tensor_torch.unsqueeze(0)

#4.打印PyTorch张量形状
print("="*50)
print("PyTorch预处理完成!")
print("PyTorch张量形状:",img_tensor_torch.shape)    #输出：torch.Size([1,3,224,224])
print("="*50)

#=========================TensorFlow创建同维度张量===========================

#创建和PyTorch同形状的TensorFlow张量([1,3,224,224])
#TF默认通道在后，这里创建和PyTorch同维度方便对比
img_tensor_tf = tf.random.normal(shape=(1,3,224,224))

#打印TensorFlow张量形状
print("TensorFlow张量创建完成")
print("TensorFlow张量形状:",img_tensor_tf.shape)    #输出(1,3,224,224)
print("="*50)

#=========================PyTorch ↔ TensorFlow 张量简单转换/对比===================

#用numpy作为中转
#1.Pytorch张量→numpy数组
np_array = img_tensor_torch.numpy()
#2.numpy数组→TensorFlow张量
tf_tensor_from_torch = tf.convert_to_tensor(np_array)

print("张量转换完成！")
print("PyTorch→NumPy→TensorFlow 张量形状：",tf_tensor_from_torch.shape)
print("两个框架张量形状是否一致：",img_tensor_torch.shape == img_tensor_tf.shape)
print("="*50)