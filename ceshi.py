import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
import pymysql
import torch
from PIL import Image
from torchvision import models,transforms
from torchvision.models import ResNet18_Weights
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applcations.mobilenet_v2 import preprocess_input,decode_predictions
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

#==================基础配置===================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "你的密码",
    "charset": "utf8mb4"
}
DB_NAME = "ai_image_recognize"
os.makedirs("result_visual",exist_ok=True)
DEVICE = torch.device("cpu")
CAT_IDX = list(range(281,286))
DOG_IDX = list(range(151,269))

#============================缓存模型=====================
@st.cache_resource(show_spinner="模型加载中...")
def load_pytorch_model():
    model = models.resent18(weights=ResNet18_Weights.DEFAULT)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource(show_spinner="模型加载中...")
def load_tf_madel():
    model = MobileNetV2(weight="imagenet")
    return model

torch_model = load_pytorch_model()
tf_model = load_tf_madel()

#========================PyTorch 预处理===========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

#=======================双模型识别================
def predoct_by_pytorch(img_pil):
    try:
        img = img_pil.con