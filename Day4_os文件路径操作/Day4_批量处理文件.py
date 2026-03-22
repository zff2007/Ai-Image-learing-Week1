import os

SOURCE_DIR = r"path"

def batch_traverse_folder():
    print("遍历文件夹：", SOURCE_DIR)
    count = 0
    for filename in os.listdir(SOURCE_DIR):
        file_path = os.path.join(SOURCE_DIR, filename)
        if os.path.isfile(file_path):
            print(f"[{count+1}] 文件名：{filename} | 路径：{file_path}")
            count += 1
    print(f"\n 共找到 {count} 个文件")

if __name__ == "__main__":
    batch_traverse_folder()