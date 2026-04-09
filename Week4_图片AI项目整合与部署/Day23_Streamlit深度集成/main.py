import streamlit as st
import torch
import os
import cv2
import pymysql
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from datetime import datetime

#=============================基础配置============================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "你的密码",
    "database": "ai_image_recognize",
    "charset": "utf8mb4"
}
DB_NAME = "ai_image_recognize"
os.makedirs("result_visual",exist_ok=True)
DEVICE = torch.device("cpu")
CAT_IDX = list(range(281,286))
DOG_IDX = list(range(151,269))

#=============================缓存模型（只加载一次）============================
@st.cache_resource
def load_pytorch_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_tf_model():
    model = MobileNetV2(weights="imagenet")
    return model

torch_model = load_pytorch_model()
tf_model = load_tf_model()

#===============================PyTorch 预处理=============================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

#===========================识别函数（双模型）===========================
def predict_by_pytorch(img_pil):
    try:
        img = img_pil.convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = torch_model(tensor)
        prob = torch.nn.functional.softmax(out,dim=1)
        cat_p = prob[0,CAT_IDX].max().item()
        dog_p = prob[0,DOG_IDX].max().item()
        top_p = prob.max().item()

        threshold = 0.3
        if cat_p > dog_p and cat_p > threshold:
            return "猫",round(cat_p*100,2)
        elif dog_p >cat_p and dog_p >threshold:
            return "狗",round(dog_p*100,2)
        else:
            return "其他",round(top_p*100,2)
    except :
        return None,None

def predict_by_tensorflow(img_pil):
    try:
        img = img_pil.convert("RGB").resize((224,224))
        x = np.array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x,axis=0)
        preds = tf_model.predict(x,verbose=0)
        top = decode_predictions(preds,top=1)[0][0]
        label = top[1]
        conf = round(top[2]*100,2)
        if "cat" in label.lower():
            return "猫",conf
        elif "dog" in label.lower():
            return "狗",conf
        else:
            return "其他",conf
    except:
        return None,None

#===============================可视化标注=====================
def visualize_img(img_pil,label,conf):
    img = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)
    cv2.putText(img, f"Result: {label}", (20,40), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
    cv2.putText(img, f"Conf: {conf}%", (20,80), cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#==========================MySQL初始化（自动建库）======================
def init_db():
    try:
        # 1.只连接服务器
        db = pymysql.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            charset=DB_CONFIG["charset"]
        )
        cursor = db.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        db.commit()
        cursor.close()
        db.close()

        # 2.再次连接
        db = pymysql.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_NAME,  
            charset=DB_CONFIG["charset"]
        )
        cursor = db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognize_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255),
                result VARCHAR(20),
                confidence FLOAT,
                model VARCHAR(20),
                create_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()
        cursor.close()
        db.close()
    except Exception as e:
        st.error(f"数据库初始化失败: {str(e)}")

#========================保存/查询/导出===================
def save_record(filename,result,conf,model):
    try:
        db = pymysql.connect(**DB_CONFIG,database=DB_NAME)
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO recognize_records(filename,result,confidence,model) VALUES(%s,%s,%s,%s)",
            (filename,result,conf,model)
        ) 
        db.commit()
        cursor.close()
        db.close()
        return True
    except:
        return False
    
def get_all_records():
    try:
        db = pymysql.connect(**DB_CONFIG,database=DB_NAME)
        df = pd.read_sql("SELECT * FROM recognize_records ORDER BY create_time DESC",db)
        db.close()
        return df
    except:
        return pd.DataFrame()
    
#======================主界面========================
def main():
    st.set_page_config(
        page_title="AI猫狗识别Agent",
        page_icon="🐱🐶",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    init_db()

    #侧边栏菜单
    with st.sidebar:
        st.header("功能菜单")
        option= st.radio("功能选择",["图片识别","历史记录"],index=0)
        st.divider()
        model_choice = st.selectbox("选择识别模型",
                                    ["PyTorch(ResNet18)","TensorFlow(MobileNetV2)"])
        
    st.title("🐱🐶 双模型AI图像识别工具")
    st.markdown("---")

    #==========识别功能============
    if option =="图片识别":
        st.subheader("📸 图片上传与识别")
        upload_type = st.radio(
            "上传方式",["单张图片","批量图片"],
            horizontal=True,index=0
        )

        #单张上传
        if upload_type =="单张图片":
            uploaded_file = st.file_uploader("选择图片",type=["jpg","png","jpeg"])
            if uploaded_file:
                img = Image.open(uploaded_file)
                st.image(img,caption="原图",use_column_width=True)

                if st.button("🚀 开始识别", type="primary", use_container_width=True):
                    with st.spinner("识别中..."):
                        #模型选择
                        if model_choice.startswith("PyTorch"):
                            res,conf = predict_by_pytorch(img)
                            use_model = "PyTorch"
                        else:
                            res,conf = predict_by_tensorflow(img)
                            use_model = "TensorFlow"

                        if not res:
                            st.error("识别失败")
                            return
                        
                        #可视化&保存
                        res_img = visualize_img(img,res,conf)
                        save_record(uploaded_file.name,res,conf,use_model)

                        #展示结果
                        st.success(f"识别完成：{res}({conf}%)")
                        col1,col2 = st.columns(2)
                        with col1:
                            st.image(res_img,caption="识别结果",use_column_width=True)
                        with col2:
                            st.metric("识别结果", res, f"{conf}% 置信度")
                            st.metric("使用模型", use_model)
                            st.dataframe(pd.DataFrame({
                                "文件名": [uploaded_file.name],
                                "结果": [res],
                                "置信度": [f"{conf}%"],
                                "模型": [use_model],
                                "时间": [datetime.now().strftime("%Y-%m-%d %H:%M")]
                            }), hide_index=True)

        # 批量上传
        else:
            files = st.file_uploader("批量选择图片", type=["jpg","png","jpeg"], accept_multiple_files=True)
            if files:
                # 预览图
                st.subheader("已上传图片预览")
                cols = st.columns(3)
                for i, f in enumerate(files):
                    with cols[i % 3]:
                        img = Image.open(f)
                        st.image(img, caption=f.name, use_column_width=True)

                if st.button("📥 批量识别", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total = len(files)
                    results = []

                    for i, f in enumerate(files):
                        progress = (i+1)/total
                        progress_bar.progress(progress)
                        status_text.text(f"正在处理：{i+1}/{total}")

                        try:
                            img = Image.open(f)
                            if model_choice.startswith("PyTorch"):
                                res, conf = predict_by_pytorch(img)
                                use_model = "PyTorch"
                            else:
                                res, conf = predict_by_tensorflow(img)
                                use_model = "TensorFlow"

                            if res:
                                res_img = visualize_img(img, res, conf)
                                save_record(f.name, res, conf, use_model)
                                results.append({
                                    "name": f.name, "res": res, "conf": conf,
                                    "model": use_model, "img": res_img
                                })
                            else:
                                results.append({"name": f.name, "res": "识别失败", "conf": "-"})
                        except:
                            results.append({"name": f.name, "res": "异常", "conf": "-"})

                    progress_bar.empty()
                    status_text.success("批量识别完成！")

                    # 展开器展示结果
                    st.subheader("识别结果列表")
                    for item in results:
                        label = f"{item['name']} | {item['res']} {item['conf'] if item['conf'] != '-' else ''}"
                        with st.expander(label):
                            if "img" in item:
                                st.image(item["img"], use_column_width=True)
                            st.dataframe(pd.DataFrame({
                                "文件名": [item["name"]],
                                "结果": [item["res"]],
                                "置信度": [str(item["conf"])+"%" if item["conf"] != "-" else "-"],
                                "模型": [item.get("model", "-")],
                                "时间": [datetime.now().strftime("%Y-%m-%d %H:%M")]
                            }), hide_index=True)

    # ========== 历史记录 ==========
    elif option == "历史记录":
        st.subheader("📊 历史识别记录")
        if st.button("🔍 查询记录", type="primary", use_container_width=True):
            df = get_all_records()
            if df.empty:
                st.info("暂无记录")
            else:
                # 格式化时间
                df["create_time"] = pd.to_datetime(df["create_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                df.rename(columns={
                    "id":"ID", "filename":"文件名", "result":"结果",
                    "confidence":"置信度(%)", "model":"模型", "create_time":"识别时间"
                }, inplace=True)
                st.dataframe(df, use_container_width=True)

                # 导出Excel
                @st.cache_data
                def export_excel(df):
                    df.to_excel("识别记录.xlsx", index=False)
                export_excel(df)
                st.success("记录已导出为：识别记录.xlsx")

if __name__ == "__main__":
    main()