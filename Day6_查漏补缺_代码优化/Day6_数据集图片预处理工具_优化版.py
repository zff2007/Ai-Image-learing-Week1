# -*- coding: utf-8 -*-
"""
数据集图片预处理工具（Day6优化版）
适用场景：图片AI开发助理实习第1周Day5/6必做产出
核心功能：批量重命名+格式统一(JPG)+尺寸统一(500*500)+透明图处理
查漏补缺优化：细化异常处理/补充全逻辑注释/增加空文件夹判断/兼容更多图片模式
核心库：PIL(Pillow) - 图片处理，os - 文件路径操作
"""
# 导入核心库，增加库导入异常捕获，解决未安装依赖的报错问题
try:
    from PIL import Image
    import os
except ImportError as e:
    print(f"库导入失败：请先执行 pip install pillow 安装依赖，错误原因：{str(e)}")
    exit()

#==================== 配置区（只需修改此处，其余代码无需改动）====================
# 原始图片所在文件夹绝对路径
SOURCE_DIR = r"path"
# 处理后图片的保存文件夹名称
OUTPUT_DIR = "processed_img"
# 目标图片尺寸（宽, 高）
TARGET_SIZE = (500, 500)
# 最多处理图片数量
MAX_IMAGES = 5
# 重命名前缀
RENAME_PREFIX = "image"
# 支持的图片格式
SUPPORT_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
# 图片保存质量（1-100），85兼顾清晰度和文件大小
SAVE_QUALITY = 85
#=================================================================================

def batch_process_images():
    """
    批量处理图片主函数：整合所有预处理逻辑，高内聚低耦合
    核心流程：路径校验 → 创建输出文件夹 → 遍历源文件 → 过滤图片 → 处理图片 → 保存 → 统计结果
    """
    # 初始化计数器：成功处理/失败处理，分别统计方便后续排查
    success_count = 0
    fail_count = 0
    # 初始化图片序号：用于重命名的三位数字编号
    img_index = 1

    # 【os路径操作优化】校验源文件夹是否存在
    if not os.path.exists(SOURCE_DIR):
        print(f"错误：源文件夹不存在，请检查路径！路径：{SOURCE_DIR}")
        return
    # 【os路径操作优化】校验源文件夹是否为空
    if len(os.listdir(SOURCE_DIR)) == 0:
        print(f"错误：源文件夹为空白文件夹，无文件可处理！路径：{SOURCE_DIR}")
        return

    # 【os路径操作优化】创建输出文件夹，exist_ok=True避免文件夹已存在的报错
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 获取输出文件夹的绝对路径，方便用户查找文件
    output_abs_path = os.path.abspath(OUTPUT_DIR)
    print(f"输出文件夹已准备好，绝对路径：{output_abs_path}")
    print(f"开始批量处理图片，最多处理{MAX_IMAGES}张，目标尺寸{TARGET_SIZE},格式统一为JPG")
    print("-" * 80)

    # 遍历源文件夹中的所有文件/文件夹
    for filename in os.listdir(SOURCE_DIR):
        # 达到最大处理数量，立即停止循环，避免多余处理
        if success_count >= MAX_IMAGES:
            break
        # 拼接文件完整绝对路径，避免相对路径混乱
        file_abs_path = os.path.join(SOURCE_DIR, filename)

        # 【os路径操作优化】过滤子文件夹，只处理文件
        if not os.path.isfile(file_abs_path):
            continue
        # 【格式过滤】过滤非支持的图片格式，忽略大小写（如PNG/Png均兼容）
        if not filename.lower().endswith(SUPPORT_FORMATS):
            continue

        # 【异常处理优化】细化异常类型，捕获不同报错原因
        try:
            # 安全打开图片：with语句自动释放文件资源，避免图片被占用无法编辑
            with Image.open(file_abs_path) as img:
                # 【PIL操作优化】创建白色背景画布，解决透明图转JPG的黑底问题
                # RGB模式为JPG标准模式，(255,255,255)为纯白色背景
                bg_canvas = Image.new("RGB", TARGET_SIZE, (255, 255, 255))

                # 【PIL操作优化】等比例缩放图片，避免变形
                # Image.Resampling.LANCZOS：最高质量的缩放算法，适合AI数据集预处理
                img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)

                # 【PIL操作优化】计算居中放置的坐标，让缩放后的图片在画布中央
                x_pos = (TARGET_SIZE[0] - img.width) // 2
                y_pos = (TARGET_SIZE[1] - img.height) // 2

                # 【PIL操作优化】处理透明通道，兼容RGBA/P模式的透明图片
                # mask参数：透明区域显示为白色背景，非透明区域正常显示
                mask = img.split()[-1] if img.mode in ("RGBA", "P") else None
                bg_canvas.paste(img, (x_pos, y_pos), mask=mask)

                # 【批量重命名】前缀+三位序号.jpg，03d保证序号为3位（001/002），方便文件排序
                new_filename = f"{RENAME_PREFIX}_{img_index:03d}.jpg"
                # 拼接处理后图片的保存路径
                new_file_abs_path = os.path.join(OUTPUT_DIR, new_filename)

                # 保存处理后的图片：JPG格式，指定保存质量
                bg_canvas.save(new_file_abs_path, "JPEG", quality=SAVE_QUALITY)

                # 打印处理成功日志，清晰展示原文件名→新文件名
                print(f"处理成功：{filename} → {new_filename}")
                # 成功后更新计数器和序号
                success_count += 1
                img_index += 1

        # 异常1：图片文件损坏，无法打开/解析
        except (IOError, SyntaxError):
            fail_count += 1
            print(f"处理失败：{filename} - 图片文件损坏/无法解析")
        # 异常2：权限不足，无法读取/保存文件
        except PermissionError:
            fail_count += 1
            print(f"处理失败：{filename} - 系统权限不足，无法访问文件")
        # 异常3：其他未预见的错误，保留通用异常捕获，避免程序中断
        except Exception as e:
            fail_count += 1
            # 异常信息截断为60字符，避免超长信息影响日志可读性
            print(f"处理失败：{filename} - 未知错误：{str(e)[:60]}")

    # 处理完成后，打印统计总结，清晰展示处理结果
    print("-" * 80)
    print(f"批量处理完成！总计处理：{success_count + fail_count}张 | 成功：{success_count}张 | 失败：{fail_count}张")
    print(f"处理后图片保存位置：{output_abs_path}")
    print(f"提示：若有失败图片，请检查文件是否损坏/是否被其他程序打开")

#==================== 脚本入口（程序运行的起始点）====================
if __name__ == "__main__":
    # 调用主函数执行批量处理
    batch_process_images()