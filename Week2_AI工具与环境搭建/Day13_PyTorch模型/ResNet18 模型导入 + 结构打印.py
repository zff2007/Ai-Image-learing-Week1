import torch
from torchvision import models

#加载ResNet18模型
#weights=None:随机初始化（无预训练权重）
#weights=models.ResNet18_Weights.DEFAULT:加载官方预训练权重
model = models.resnet18(weights=models.RegNet18_Weights.DEFAULT)

#打印模型完整结构（原生打印）
print("="*50)
print("ResNet18 原生模型结构")
print("="*50)
print(model)


#详细打印
from torchsummary import summary

#切换模型
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
model = model.to(device)

#打印详细结构：输入尺寸为ResNet标准输入（3通道，224*224）
print("\n" + "="*50)
print("ResNet18 详细结构（含参数量/尺寸）")
print("="*50)
summary(model, (3, 224, 224))