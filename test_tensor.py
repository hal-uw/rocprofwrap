import torch
import time

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available, please check your PyTorch and CUDA installation")
    exit()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print("-" * 50)

# Set device to cuda:0
device = torch.device("cuda:0")

# Create two 2048x2048 random tensors
print("Creating 16384x16384 tensors...")
tensor_a = torch.randn(16384, 16384, device=device)
tensor_b = torch.randn(16384, 16384, device=device)

# Matrix multiplication
print("Performing matrix multiplication...")
start_time = time.time()
result = torch.matmul(tensor_a, tensor_b)
torch.cuda.synchronize()  # Wait for GPU computation to finish
matmul_time = time.time() - start_time
print(f"Matrix multiplication time: {matmul_time:.4f}s")

# Element-wise operations
print("Performing element-wise addition...")
start_time = time.time()
result_add = tensor_a + tensor_b
torch.cuda.synchronize()
add_time = time.time() - start_time
print(f"Element-wise addition time: {add_time:.4f}s")

# Other operations
print("Performing other operations...")
result_mean = result.mean()
result_std = result.std()
result_max = result.max()
result_min = result.min()

print("-" * 50)
print("Result statistics:")
print(f"Matrix multiplication result mean: {result_mean.item():.4f}")
print(f"Matrix multiplication result std dev: {result_std.item():.4f}")
print(f"Matrix multiplication result max: {result_max.item():.4f}")
print(f"Matrix multiplication result min: {result_min.item():.4f}")

# Display GPU memory usage
print("-" * 50)
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

print("\nComputation complete!")
