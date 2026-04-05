import torch 
import cv2
import os
import pymysql
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models,transforms


#=====================基础配置===================
#图片文件夹
IMAGE_DIR =r"path"
#可视化结果保存文件夹
VISUAL_DIR = "result_visual"
os.makedirs(IMAGE_DIR,exist_ok=True)
os.makedirs(VISUAL_DIR,exist_ok=True)

#MySQL配置
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "123456"#改成自己的密码
DB_NAME = "ai_image_recognize"

# 设备配置（CPU/GPU自动识别）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#模型加载
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = model.to(device)
model.eval()

#图片预处理
transform = transforms.Compose([
    transforms.Resize(256),          # 短边缩到256，严格保持原图宽高比，无变形
    transforms.CenterCrop(224),      # 从中心裁剪224×224，保留图片核心主体
    transforms.ToTensor(),           # 转为PyTorch张量，同时把像素值[0,255]归一化到[0,1]
    transforms.Normalize(            # 用ImageNet数据集的均值/标准差标准化，和预训练权重严格对齐
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#ImageNet 猫狗索引
CAT_IDX = list(range(281,286))
DOG_IDX = list(range(151,269))

#======================MySQL初始化=============================
def init_mysql():
    """创建数据库和表"""
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        charset="utf8mb4"
    )  #连接MySQL服务器
    cursor = db.cursor()    #创建游标（相当于 MySQL 的命令输入框）
    #创建数据库（相当于新建一个项目专属文件夹）
    #cursor.execute()：把括号里的 SQL 命令发给 MySQL 执行
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    #切换到目标数据库（相当于打开刚才建的文件夹）
    cursor.execute(f"USE {DB_NAME}")
    #创建结果表（相当于文件夹里的 Excel 表格，存分类结果）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results(
            id INT AUTO_INCREMENT PRIMARY KEY,
            img_path VARCHAR(255) NOT NULL,
            label VARCHAR(20) NOT NULL,
            confidence FLOAT NOT NULL,
            save_path VARCHAR(255)
        )
    """)
    db.commit()
    cursor.close()
    db.close()

#======================保存结果到MySQL===============================
def save_to_mysql(img_path,label,confidence,save_path):
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4"
    )
    cursor = db.cursor()
    sql = "INSERT INTO results(img_path,label,confidence,save_path) VALUES(%s, %s, %s, %s)"
    cursor.execute(sql,(img_path,label,confidence,save_path))
    db.commit()
    last_id = cursor.lastrowid
    cursor.close()
    db.close()
    return last_id

#========================MySQL导出Excel=============================
def export_to_excel():
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4"
    )
    df = pd.read_sql("SELECT * FROM results", db)
    df.to_excel("AI识别结果.xlsx", index=False)
    db.close()
    print("Excel导出完成:AI识别结果.xlsx")

#=====================单张图片识别===================
def predict_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        prob = torch.nn.functional.softmax(out,dim=1)
        idx = torch.argmax(prob,1).item()
        conf = round(float(prob[0][idx])*100,2)

        if idx in CAT_IDX:
            return "猫",conf
        elif idx in DOG_IDX:
            return "狗",conf
        else:
            return "其他",conf
    except:
        return None,None

#===================结果可视化（标注在图上）=================
def visualize(img_path,save_path,label,confidence):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    #标注文字
    cv2.putText(img,f"Label:{label}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
    cv2.putText(img, f"Conf: {confidence}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    cv2.putText(img, f"Save: {save_path[:40]}...", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.imencode(os.path.splitext(save_path)[1], img)[1].tofile(save_path)

#===================批量识别主系统========================
def run_day19():
    init_mysql()
    print("=" * 80)
    print("批量识别 | 可视化 | MySQL | Excel 导出")
    print("=" * 80)
    # 格式化表头
    print(f"{'数据库ID':<10}{'图片路径':<30}{'识别结果':<10}{'置信度':<10}{'状态'}")
    print("-" * 80)

    # 遍历图片
    for filename in os.listdir(IMAGE_DIR):
        img_path = os.path.join(IMAGE_DIR, filename)
        # 跳过非图片
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            print(f"{'':<10}{filename:<30}{'-':<10}{'-':<10}跳过非图片")
            continue

        # 识别
        label, conf = predict_image(img_path)
        if not label:
            print(f"{'':<10}{filename:<30}{'-':<10}{'-':<10}损坏/无法识别")
            continue

        # 可视化保存
        save_path = os.path.join(VISUAL_DIR, filename)
        visualize(img_path, save_path, label, conf)

        # 存入MySQL
        db_id = save_to_mysql(img_path, label, conf, save_path)

        # 优化格式输出
        print(f"{db_id:<10}{filename:<30}{label:<10}{str(conf)+'%':<10}识别成功")

    print("=" * 80)
    # 导出Excel
    export_to_excel()
    print("全部任务完成！")
    print("=" * 80)

# ===================== 运行 =====================
if __name__ == "__main__":
    run_day19()