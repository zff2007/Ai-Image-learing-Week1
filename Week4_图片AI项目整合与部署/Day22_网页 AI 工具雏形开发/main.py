import streamlit as st
import torch
import cv2
import numpy as np
import pymysql
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import os
from datetime import datetime

#===============================基础配置============================
#MySQL配置
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "你的密码"
DB_NAME = "ai_image_recognize"

#设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ImaNet 猫狗索引
CAT_IDX = list(range(281,286))
DOG_IDX = list(range(151,269))

#可视化结果保存文件夹
VISUAL_DIR = "result_visual"
os.makedirs(VISUAL_DIR,exist_ok=True)

#=========================模型加载（Streamlit缓存，避免重复加载）==================
@st.cache_resource  #关键：只加载一次模型，大幅提升速度
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = model.to(device)
    model.eval()
    return model

model = load_model()

#==============================图片预处理==========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

#==============================单张图片识别==========================
def predict_image(img_pil):
    try:
        img = img_pil.convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
        prob = torch.nn.functional.softmax(out,dim=1)

        #猫狗判断逻辑
        cat_prob = prob[0,CAT_IDX].max().item()
        dog_prob = prob[0,DOG_IDX].max().item()
        top1_prob = prob.max().item()

        threshold = 0.3
        if cat_prob > dog_prob and cat_prob >threshold:
            return "猫",round(cat_prob*100,2)
        elif dog_prob >cat_prob and dog_prob >threshold:
            return "狗",round(dog_prob*100,2)
        else:
            return "其他",round(top1_prob*100,2)
    except Exception as e:
        st.error(f"识别失败:{str(e)}")
        return None,None
    
#============================结果可视化==========================
def visualize(img_pil,label,confidence,save_path):
    #把PIL图片转成OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)

    #标注文字
    cv2.putText(img_cv,f"Label:{label}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
    cv2.putText(img_cv,f"Conf:{confidence}",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),2)
    
    #保存图片(兼容中文路径)
    cv2.imencode(os.path.splitext(save_path)[1],img_cv)[1].tofile(save_path)

    #把OpenCv图片转回PIL，用于Streamlitz展示
    img_result = Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_RGB2BGR))
    return img_result

#=============================MySQL操作==========================
def init_mysql():
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        charset="utf8mb4"
    )
    cursor = db.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cursor.execute(f"USE {DB_NAME}")
    
    # 先检查表是否存在，不存在则创建
    cursor.execute("SHOW TABLES LIKE 'results'")
    table_exists = cursor.fetchone()
    if not table_exists:
        cursor.execute("""
            CREATE TABLE results(
                id INT AUTO_INCREMENT PRIMARY KEY,
                img_name VARCHAR(255) NOT NULL,
                label VARCHAR(20) NOT NULL,
                confidence FLOAT NOT NULL,
                save_path VARCHAR(255),
                create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    else:
        # 检查表结构是否正确，不正确则重建
        cursor.execute("DESCRIBE results")
        columns = [col[0] for col in cursor.fetchall()]
        required_columns = ['id', 'img_name', 'label', 'confidence', 'save_path', 'create_time']
        if set(columns) != set(required_columns):
            cursor.execute("DROP TABLE IF EXISTS results")
            cursor.execute("""
                CREATE TABLE results(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    img_name VARCHAR(255) NOT NULL,
                    label VARCHAR(20) NOT NULL,
                    confidence FLOAT NOT NULL,
                    save_path VARCHAR(255),
                    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    db.commit()
    cursor.close()
    db.close()

def save_to_mysql(img_name,label,confidence,save_path):
    """保存识别结果到MySQL"""
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4"
    )
    cursor = db.cursor()
    sql = "INSERT INTO results(img_name,label,confidence,save_path) VALUES(%s, %s, %s, %s)"
    cursor.execute(sql,(img_name,label,confidence,save_path))
    db.commit()
    cursor.close()
    db.close()

def query_mysql():
    """查询MySQL所有历史记录"""
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4"
    )
    df = pd.read_sql("SELECT * FROM results ORDER BY create_time DESC",db)
    db.close()
    return df

#===============================Streamlit 网页界面=========================== 
def main():
    #页面配置
    st.set_page_config(
        page_title="AI猫狗图像识别工具",
        page_icon="🐱🐶",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    #初始化MySQL（首次运行自动创建表）
    init_mysql()

    #页面标题
    st.title("🐱🐶 AI猫狗图像识别工具(网页版)")
    st.markdown("---")

    #侧边栏：功能选择
    with st.sidebar:
        st.header("功能菜单")
        option = st.radio(
            "请选择功能",
            ["图片识别","历史记录查询"],
            index=0
        )

    #==================功能1:图片识别=================
    if option == "图片识别":
        st.subheader("📸 图片上传与识别")

        #上传方式选择
        upload_type = st.radio(
            "选择上传方式",
            ["单张图片上传","批量图片上传"],
            index=0,
            horizontal=True
        )

        if upload_type =="单张图片上传":
            #单张图片上传
            uploaded_file = st.file_uploader(
                "请选择一张图片(支持jpg/png/jpeg)",
                type = ["jpg","png","jpeg"],
                accept_multiple_files=False
            )

            if uploaded_file is not None:
                #读取图片
                img_pil = Image.open(uploaded_file)
                #展示原图
                st.image(img_pil,caption="上传的原图",use_column_width=True)

                #识别按钮
                with st.spinner("正在识别中..."):
                    #1.识别
                    label,conf = predict_image(img_pil)
                    if not label:
                        st.error("图片识别失败，请检查图片格式！")
                        return
                    
                    #2.可视化标注
                    img_name = uploaded_file.name
                    save_path = os.path.join(VISUAL_DIR,img_name)
                    img_result = visualize(img_pil,label,conf,save_path)

                    #3.保存到MySQL
                    save_to_mysql(img_name,label,conf,save_path)

                    #4.展示结果
                    st.success("识别完成！")
                    st.subheader("识别结果")
                    col1,col2 = st.columns(2)
                    with col1:
                        st.image(img_result,caption="标注后的图片",use_column_width=True)
                    with col2:
                        st.metric(label="识别结果",value=label,delta=f"{conf}% 置信度")
                        st.dataframe(
                            pd.DataFrame({
                                "图片名称": [img_name],
                                "识别结果": [label],
                                "置信度": [f"{conf}%"],
                                "识别时间": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                            }),
                            hide_index = True,
                            use_container_width=True
                        )
        else:
            #批量上传
            uploaded_files = st.file_uploader(
                "请选择多张图片(支持jpg/png/jpeg)",
                type=["jpg","png","jpeg"],
                accept_multiple_files=True
            )

            if uploaded_files:
                #展示所有上传图片
                st.subheader("上传图片")
                cols = st.columns(3)
                for i,uploaded_file in enumerate(uploaded_files):
                    with cols[i % 3]:
                        img_pil = Image.open(uploaded_file)
                        st.image(img_pil,caption=uploaded_file.name,use_column_width=True)

                #批量识别按钮
                if st.button("批量识别开始",type="primary",use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    for i,uploaded_file in enumerate(uploaded_files):
                        #更新进度
                        progress = (i+1)/len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"正在识别第:{i+1}/{len(uploaded_files)}张照片...")

                        #读取图片
                        img_pil = Image.open(uploaded_file)
                        #识别
                        label,conf = predict_image(img_pil)
                        if not label:
                            results.append({
                                "图片名称": uploaded_file.name,
                                "识别结果": "识别失败",
                                "置信度": "-",
                                "标注图片": None
                            })
                            continue

                        #可视化标注
                        img_name = uploaded_file.name
                        save_path = os.path.join(VISUAL_DIR,img_name)
                        img_result = visualize(img_pil,label,conf,save_path)

                        #保存到MySQL
                        save_to_mysql(img_name,label,conf,save_path)

                        #收集结果
                        results.append({
                            "图片名称": img_name,
                            "识别结果": label,
                            "置信度": f"{conf}%",
                            "标注图片": img_result
                        })

                    #识别完成，展示结果
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"批量识别完成！共识别{len(uploaded_files)}张图片")

                    #展示所有结果
                    st.subheader("批量识别结果")
                    for res in results:
                        with st.expander(f"{res['图片名称']} - {res['识别结果']}({res['置信度']})"):
                            if res["标注图片"]:
                                st.image(res["标注图片"],caption="标注后的图片",use_column_width=True)
                                st.dataframe(
                                    pd.DataFrame({
                                        "图片名称": [res["图片名称"]],
                                        "识别结果": [res["识别结果"]],
                                        "置信度": [res["置信度"]],
                                        "识别时间": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] 
                                    }),
                                    hide_index=True,
                                    use_container_width=True
                                )
                            else:
                                st.error("该图片识别失败")

    elif option =="历史记录查询":
        st.subheader("📋 历史识别记录查询")

        #查询按钮
        if st.button("🔍 查询所有历史记录",type="primary",use_container_width=True):
            with st.spinner("正在查询数据库..."):
                df = query_mysql()
                if df.empty:
                    st.info("暂无历史识别记录")
                else:
                    #格式化时间
                    df["create_time"]=pd.to_datetime(df["create_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    #重命名列，更友好
                    df = df.rename(columns={
                        "id": "数据库ID",
                        "img_name": "图片名称",
                        "label": "识别结果",
                        "confidence": "置信度(%)",
                        "save_path": "保存路径",
                        "create_time": "识别时间"
                    })
                    #展示表格
                    st.dataframe(df,hide_index=True,use_container_width=True)
                    #导出Excel按钮
                    @st.cache_data
                    def convert_df(df):
                        return df.to_excel("历史识别记录.xlsx",index=False)
                    convert_df(df)
                    st.success("历史记录已导出为「历史识别记录.xlsx」")


if __name__ =="__main__":
    main()