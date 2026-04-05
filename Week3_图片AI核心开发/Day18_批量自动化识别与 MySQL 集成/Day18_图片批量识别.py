import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

#=============================1.模型加载 + 预处理===================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

#图片预处理
preprocess = transforms.Compose([
    transforms.Resize(256),          # 短边缩到256，严格保持原图宽高比，无变形
    transforms.CenterCrop(224),      # 从中心裁剪224×224，保留图片核心主体
    transforms.ToTensor(),           # 转为PyTorch张量，同时把像素值[0,255]归一化到[0,1]
    transforms.Normalize(            # 用ImageNet数据集的均值/标准差标准化，和预训练权重严格对齐
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#==========================2.内置ImaNet 猫狗判断=======================
#直接内置猫/狗类别关键词，避免读取文件报错
CAT_KEYWORDS = ["cat","Egyptian cat","tiger cat","Persian cat","Siamese cat"]
DAG_KEYWORDS = ["dor","beagle","bulldag","poodle","golden retriever","Labrador"]

#==========================3.单张图片识别函数======================

def recognize_one_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        #推理
        with torch.no_grad():
            output = model(img_tensor)
        #转概率
        prob = torch.nn.functional.softmax(output, dim=1)#把output 实数向量归一化为「和为 1 的概率分布」
        idx = torch.argmax(prob,1).item()
        #内置1000类标签（只用到猫狗判断）
        classes = ["cat" if i in [281,282,283,284,285]else "dag" if i in range[151,269] else "other" for i in range[1000]]              
        result = classes[idx]
        confidence = round(prob[0][idx].iten()*100,2)
        return result,confidence
    except Exception as e:
        #异常：损坏图/非图片/无法读取→直接返回None
        return None,None

#==============================4.批量识别+异常处理========================

def batch_recognize_images(folder_path):
    #支持的图片格式
    img_suffix = [".jpg",".jpeg",".png",".bmp"]
    #遍历文件夹
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path,filename)
        #只处理文件
        if not any(filename.lower().endswith(suf) for suf in img_suffix):
            print(f"跳过非文件文件：{filename}")
            continue

        #调用识别+异常判断
        result,conf = recognize_one_image(file_path)
        if result is None:
            print(f"跳过损坏/无法读取图片：{filename}")
            continue

        #输出正常识别结果
        if result =="cat":
            print(f"{filename}→识别结果：猫 | 置信度：{conf}%")
        elif result == "dog":
            print(f"{filename}→识别结果：狗 | 置信度：{conf}%")
        else:
            print(f"{filename}→识别结果：非猫狗 | 置信度：{conf}%")

#===============================5.住运行==========================
if __name__ =="__main__":
     # 替换成自己的图片文件夹路径
    IMAGE_FOLDER = "path"
    print("="*60)
    print("开始批量识别文件夹内所有图片...")
    print("="*60)
    batch_recognize_images(IMAGE_FOLDER)
    print("="*60)
    print("批量识别完成！")
    print("="*60)