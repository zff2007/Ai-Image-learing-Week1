import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

#---------------------------1.加载模型+预处理-----------------------------
#加载预训练ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()   #推理模型

#图片预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

# -------------------------- 2. 读取图片（替换你的猫狗图片路径） --------------------------
img = Image.open("cat.jpg").convert("RGB")  # 改成你的图片：dog.jpg / cat.jpg
img_tensor = preprocess(img).unsqueeze(0)

# -------------------------- 3. 模型推理 + 结果解析 --------------------------
with torch.no_grad():
    output = model(img_tensor)

# 转置信度概率
prob = torch.nn.functional.softmax(output, dim=1)#沿着类别维度归一化
# 加载ImageNet类别
with open("imagenet_classes.txt", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

# 获取结果
#.item()：将 PyTorch 张量转为 Python 原生整数
idx = torch.argmax(prob, 1).item()#找到概率张量prob中，沿指定维度的最大值对应的索引
result = classes[idx]
#prob[0][idx]	提取对应类别的概率值
#.item()	张量转 Python 浮点数
#* 100	概率转百分比
#round(..., 2)	四舍五入保留 2 位小数
conf = round(prob[0][idx].item() * 100, 2)

# -------------------------- 4. 输出猫狗识别结果 --------------------------
if "cat" in result.lower():
    print(f"识别结果：猫 | 置信度：{conf}%")
elif "dog" in result.lower():
    print(f"识别结果：狗 | 置信度：{conf}%")
else:
    print(f"识别结果：{result} | 置信度：{conf}%（不是猫也不是狗）")