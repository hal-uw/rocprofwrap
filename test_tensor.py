import torch
import time

# 检查CUDA是否可用
if not torch.cuda.is_available():
    print("CUDA不可用，请检查PyTorch和CUDA安装")
    exit()

print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA设备数量: {torch.cuda.device_count()}")
print(f"当前CUDA设备: {torch.cuda.current_device()}")
print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
print("-" * 50)

# 设置设备为cuda:0
device = torch.device("cuda:0")

# 创建两个2048x2048的随机tensor
print("创建2048x2048的tensor...")
tensor_a = torch.randn(16384, 16384, device=device)
tensor_b = torch.randn(16384, 16384, device=device)

# 矩阵乘法
print("执行矩阵乘法...")
start_time = time.time()
result = torch.matmul(tensor_a, tensor_b)
torch.cuda.synchronize()  # 等待GPU计算完成
matmul_time = time.time() - start_time
print(f"矩阵乘法耗时: {matmul_time:.4f}秒")

# 元素级运算
print("执行元素级加法...")
start_time = time.time()
result_add = tensor_a + tensor_b
torch.cuda.synchronize()
add_time = time.time() - start_time
print(f"元素级加法耗时: {add_time:.4f}秒")

# 其他运算示例
print("执行其他运算...")
result_mean = result.mean()
result_std = result.std()
result_max = result.max()
result_min = result.min()

print("-" * 50)
print("计算结果统计:")
print(f"矩阵乘法结果均值: {result_mean.item():.4f}")
print(f"矩阵乘法结果标准差: {result_std.item():.4f}")
print(f"矩阵乘法结果最大值: {result_max.item():.4f}")
print(f"矩阵乘法结果最小值: {result_min.item():.4f}")

# 显示GPU内存使用情况
print("-" * 50)
print(f"GPU内存已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"GPU内存已缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

print("\n计算完成!")
