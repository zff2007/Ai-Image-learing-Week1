import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet         #导入模型结构
from tensorflow.keras.applications.mobilenet import preprocess_input as tf_preprocess   #导入图像预处理函数
from tensorflow.keras.applications.imagenet_utils import decode_predictions     #导入预测结果解码器

#========================PyTorch + ResNet18 图片分类=============================
def pytorch_resnet_classify(img_path):
    #1.加载预训练ResNet8模型（CPU版）
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()#切换推理模式

    #2.图片预处理（和Day16一致，ResNet标准流程）
    preprocess = transforms.Compose([
        transforms.transforms.Resize(256),
        transforms.transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    #3.读取并预处理图片
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)   #增加批次维度

    #4.模型推理（禁用梯度计算）
    with torch.no_grad():
        outputs = model(img_tensor)

    #--------------------模型输出结果解析-------------------
    #步骤1：将输出分数转为概率（softmax）
    probabilities = torch.nn.functional.softmax(outputs,dim=1)

    #步骤2：取最大概率值（置信度）和对应类别索引
    confidence,pred_idx = torch.max(probabilities,1)
    
    #步骤3：类别索引→真实标签（ImageNet 100类）
    with open("imagenet_classes.txt","r",encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]
    pred_class = classes[pred_idx.item()]
    pred_conf = round(confidence.item()*100,2)

    print("="*60)
    print("PyTorch ResNet18 识别结果")
    print(f"图片路径：{img_path}")
    print(f"识别类别：{pred_class}")
    print(f"置信度：{pred_conf}%")
    print("="*60)
    return pred_class, pred_conf

#======================TensorFlow + MobileNet 图片分类========================

def tensorflow_mobilenet_classify(img_path):
    #1.加载预训练MobileNet模型
    model = MobileNet(weights="imagenet",input_shape=(224,224,3))

    #2.读取并预处理图片（TF专用预处理）
    img = Image.open(img_path).convert("RGB").resize((224,224))
    img_array = np.array(img)
    img_array = tf_preprocess(img_array)    #TF模型预处理
    img_array = np.expand_dims(img_array,axis=0)    #增加批次维度

    #3.模型推理
    predictions = model.predict(img_array,verbose=0)

    #--------------------模型输出结果解析-------------------
    #TF自带解码函数，直接转标签+置信度
    results = decode_predictions(predictions,top=1)[0][0]
    pred_class = results[1]
    pred_conf = round(results[2]*100,2)

    print("TensorFlow MobileNet 识别结果")
    print(f"图片路径：{img_path}")
    print(f"识别类别：{pred_class}")
    print(f"置信度：{pred_conf}%")
    print("="*60)
    return pred_class, pred_conf

#==============================主函数：运行双框架分类========================
if __name__ =="__main__":
    #替换成图片的路径
    IMAGE_PATH = "path"

    # 运行PyTorch分类
    pytorch_resnet_classify(IMAGE_PATH)
    # 运行TensorFlow分类
    tensorflow_mobilenet_classify(IMAGE_PATH)




