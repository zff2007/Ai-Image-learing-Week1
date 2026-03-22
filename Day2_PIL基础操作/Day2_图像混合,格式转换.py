from PIL import Image
#透明度混合
img1=Image.open(r"path").convert(mode="RGB")
img2=Image.new("RGB",img1.size,"red")#创建图片
img2.show
Image.blend(img1,img2,alpha=0.5).show()

#遮罩混合图片
img3=Image.open(r"path").convert(mode="RGB")
img3=img3.resize(img1.size)#确保跟二者大小相同
r,g,b=img3.split()
Image.composite(img3,img1,b).show()

#格式转换--->save(png转jpg是要填充白色背景)
import os
#批量把文件夹里的所有jpg转png
# 定义目标文件夹路径（避免重复写长路径）
folder_path = r"path"

# 遍历文件夹内所有内容
for filename in os.listdir(folder_path):
    # 拼接完整文件路径（安全拼接，避免路径分隔符问题）
    file_path = os.path.join(folder_path, filename)
    #os.listdir()会把文件夹也打开，
    # 只处理【文件】，且后缀为 .jpg/.jpeg（兼容大小写）
    if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg")):
        #异常捕获：防止脚本因坏文件崩溃
        #try：----->包裹可能出错的代码
            #转换逻辑
        #except Exception as e:----->捕获异常，不会让整个脚本中断，坏文件跳过，继续下一个
            #print(f转换失败：{filename}，错误原因：{str（e}")
        try:
            # 打开当前图片
            img = Image.open(file_path)
            # 生成新文件名：替换后缀为 .png
            #os.path.splitext()会返回两个元组（“文件名”，“后缀”）
            new_filename = os.path.splitext(filename)[0] + ".png"
            # 拼接新文件的完整保存路径（保存到原文件夹）
            new_file_path = os.path.join(folder_path, new_filename)
            # 保存为 PNG 格式
            img.save(new_file_path)
            print(f"转换成功：{filename} → {new_filename}")
        except Exception as e:
            print(f"转换失败：{filename}，错误原因：{str(e)}")

print("\n批量转换完成！")

