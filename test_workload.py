import torch
import time

def main():
    device = torch.device("cuda:0")
    N = 65536
    num_iters = 10

    print(f"GEMM size: {N}x{N}, iterations: {num_iters}, device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Allocate matrices on GPU 0
    A = torch.randn(N, N, device=device, dtype=torch.float16)
    B = torch.randn(N, N, device=device, dtype=torch.float16)

    # Warm-up
    torch.cuda.synchronize()
    _ = torch.mm(A, B)
    torch.cuda.synchronize()
    print("Warm-up done.")

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_iters):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Total time: {elapsed:.3f} s")
    print(f"Avg per iteration: {elapsed / num_iters * 1000:.3f} ms")

    # TFLOPS calculation: 2*N^3 per GEMM
    flops_per_iter = 2 * (N ** 3)
    total_flops = flops_per_iter * num_iters
    tflops = total_flops / elapsed / 1e12
    print(f"Throughput: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    main()
