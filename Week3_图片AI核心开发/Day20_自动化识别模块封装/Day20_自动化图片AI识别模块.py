import os
import numpy as np
import pandas as pd 
import pymysql
from PIL import Image
import torch
from torchvision import models
from torchvision.models import ResNet18_Weights  # 修复PyTorch版本警告
import tensorflow as tf
from datetime import datetime

#核心：自动获取代码文件所在的绝对目录，彻底解决所有路径问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#消除TensorFlow oneDNN警告，让日志更干净
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#打印关键路径，方便你一键排查（运行后看终端输出）
print(f"代码所在目录：{BASE_DIR}")

#---------------------------图片预处理（双框架适用）---------------------------
class ImagePreprocessor:
    """图片预处理类:统一适配PyTorch/TensorFlow输入要求"""
    def __init__(self,target_size=(224,224)):
        self.target_size = target_size  #模型标准输入尺寸

    def pytorch_preprocess(self,img_path):
        """PyTorch格式预处理: ResNet18专用"""
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(self.target_size)
            #转张量+归一化
            img_np = np.array(img).astype(np.float32)/255.0
            img_tensor = np.transpose(img_np,(2,0,1)) #HWC→CHW
            return np.expand_dims(img_tensor,axis=0)  # 增加batch维度
        except Exception as e:
            print(f"[PyTorch预处理失败]{img_path}:{str(e)}")       
            return None 
        
    def tensorflow_preprocess(self,img_path):
        """TensorFlow格式预处理:MobileNetV2专用"""
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(self.target_size)
            img_np = np.array(img).astype(np.float32)/255.0
            return np.expand_dims(img_np,axis=0)    #保持BHWC格式
        except Exception as e:
            print(f"[TensorFlow预处理失败] {img_path}:{str(e)}")
            return None
        
#------------------------------双框架模型调用--------------------------
class DualModelPredictor:
    """双框架模型调用类：加载预训练模型，返回识别结果"""
    def __init__(self):
        # 1. 加载PyTorch ResNet18（CPU），彻底修复pretrained过时警告
        self.torch_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.torch_model.eval()  # 推理模式
        
        label_path = os.path.join(BASE_DIR, "imagenet_classes.txt")
        print(f"标签文件路径：{label_path}")
        # 提前检查文件，给你明确的错误提示
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件不存在!请将imagenet_classes.txt放到:{BASE_DIR} 文件夹下")
        
        with open(label_path, "r", encoding="utf-8") as f:
            self.torch_classes = [line.strip() for line in f.readlines()]

        # 2.加载TensorFlow MobileNetV2（CPU）
        self.tf_model = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    def pytorch_predict(self,img_tensor):
        """PyTorch模型推理"""
        if img_tensor is None:
            return None,0.0
        with torch.no_grad():
            output = self.torch_model(torch.from_numpy(img_tensor))
            prob = torch.nn.functional.softmax(output,dim=1)
            top_prob,top_idx = torch.max(prob,1)
            return self.torch_classes[top_idx.item()],round(top_prob.item(),4)
        
    def tensorflow_predict(self,img_tensor):
        """TensorFlow模型推理"""
        if img_tensor is None:
            return None,0.0
        predictions = self.tf_model.predict(img_tensor,verbose=0)
        result = tf.keras.applications.mobilenet_v2.decode_predictions(predictions,top=1)[0][0]
        return result[1],round(float(result[2]),4)
    
#-----------------------------------MySQL数据库操作------------------------------
class MySQLHandler:
    """MySQL操作类:链接、建表、插入识别结果"""
    def __init__(self,host="localhost",user="root",password="你的密码",db="image_ai"):
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_table()

    def connect(self):
        """连接数据库"""
        try:
            self.conn = pymysql.connect(
                host=self.host, user=self.user, password=self.password,
                database=self.db, charset="utf8mb4"
            )
            self.cursor = self.conn.cursor()
            print("MySQL连接成功")
        except Exception as e:
            print(f"MySQL连接失败:{str(e)}")

    def create_table(self):
        """创建图片识别结果表"""
        sql = """
        CREATE TABLE IF NOT EXISTS image_recognize (
            id INT PRIMARY KEY AUTO_INCREMENT,
            img_path VARCHAR(512) NOT NULL,
            torch_label VARCHAR(100),
            torch_conf FLOAT,
            tf_label VARCHAR(100),
            tf_conf FLOAT,
            create_time DATETIME DEFAULT NOW()
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self.cursor.execute(sql)
        self.conn.commit()

    def insert_result(self, img_path, torch_label, torch_conf, tf_label, tf_conf):
        """插入识别结果"""
        sql = """
        INSERT INTO image_recognize (img_path, torch_label, torch_conf, tf_label, tf_conf)
        VALUES (%s, %s, %s, %s, %s)
        """
        self.cursor.execute(sql, (img_path, torch_label, torch_conf, tf_label, tf_conf))
        self.conn.commit()

    def close(self):
        """关闭连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

#-------------------------------------结果导出---------------------------------------
class ResultExporter:
    """结果导出类:MySQL数据导出Excel"""
    def __init__(self,mysql_handler):
        self.mysql = mysql_handler

    def export_to_excel(self,save_path=None):
        """导出全量数据到Excel，默认保存在代码目录"""
        if save_path is None:
            save_path = os.path.join(BASE_DIR, "image_recognize_result.xlsx")
        sql = "SELECT * FROM image_recognize"
        df = pd.read_sql(sql,self.mysql.conn)
        df.to_excel(save_path,index=False)
        print(f"结果已导出至:{save_path}")

#------------------------------------主自动化执行函数-----------------------------
def auto_image_recognize(folder_path):
    """
    自动化批量识别主函数
    流程:遍历文件夹→预处理→双框架识别→MySQL存储→异常跳过
    """     
    #初始化组件
    preprocessor = ImagePreprocessor()
    predictor = DualModelPredictor()
    mysql = MySQLHandler()
    exporter = ResultExporter(mysql)

    #遍历图片文件
    img_suffix = [".jpg",".jpeg",".png",".bmp"]
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path,filename)
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in img_suffix:
            print(f"\n正在处理:{file_path}")

            # 1. 双框架预处理
            torch_tensor = preprocessor.pytorch_preprocess(file_path)
            tf_tensor = preprocessor.tensorflow_preprocess(file_path)

            # 2. 双框架推理
            torch_label, torch_conf = predictor.pytorch_predict(torch_tensor)
            tf_label, tf_conf = predictor.tensorflow_predict(tf_tensor)

            # 3. 存储结果
            if torch_label and tf_label:
                mysql.insert_result(file_path, torch_label, torch_conf, tf_label, tf_conf)
                print(f"PyTorch:{torch_label}({torch_conf}) | TensorFlow:{tf_label}({tf_conf})")

    # 导出结果
    exporter.export_to_excel()
    mysql.close()
    print("\n自动化图片AI识别全流程完成!")

# -------------------------- 执行入口 --------------------------
if __name__ == "__main__":
    #图片文件夹固定在代码目录下，彻底解决路径混乱
    IMAGE_FOLDER = os.path.join(BASE_DIR, "test_images")
    print(f"图片文件夹路径：{IMAGE_FOLDER}")
    
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"请将图片放入：{IMAGE_FOLDER}")
    else:
        auto_image_recognize(IMAGE_FOLDER)