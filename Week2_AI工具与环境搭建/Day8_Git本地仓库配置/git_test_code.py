# Git本地仓库提交测试代码
# 功能：统计1-100内偶数个数（用于测试Git提交流程）

def count_even_number(start, end):
    """统计指定区间内的偶数个数"""
    even_count = 0
    for num in range(start, end + 1):
        if num % 2 == 0:
            even_count += 1
    return even_count

# 主程序执行
if __name__ == "__main__":
    start_num = 1
    end_num = 100
    result = count_even_number(start_num, end_num)
    print(f"[{start_num}-{end_num}] 范围内的偶数个数为：{result}")
    # 运行结果：50