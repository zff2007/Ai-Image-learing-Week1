# -*- coding: utf-8 -*-
#PyTorch CPU版安装验证脚本

import torch
import torchvision
def verify_pytorch():
    try:
        #验证库导入    
        print("成功导入Pytorch和torchvision库")
        #打印版本信息
        print(f"PyTorch版本:{torch.__version__}")
        print(f"tprchvision版本:{torchvision.__version__}")
        #验证CPU设备
        if not torch.cuda.is_available():
            print("确认当前为纯CPU环境，无CUDA依赖，符合学习要求")
            print(f"可用设备：{torch.device('cpu')}")
        #简单张量运算验证
        x = torch.tensor([1,2,3],device="cpu")
        y = torch.tensor([4,5,6],device="cpu")
        print(f"简单CPU张量运算测试: x+y ={x+y}")
        print(f"所有验证通过！PyTorch CPU版安装成功，可以开始后续学习！")
    except ImportError as e:
        print(f"库导入失败：{e},请重新安装PyTorch")
    except Exception as e:
        print(f"验证失败:{e}")

if __name__ =="__main__":
    verify_pytorch()