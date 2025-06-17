import subprocess
import numpy as np
import torch
import time
import psutil
import os
from pathlib import Path


def get_memory_usage() -> tuple[float, float]:
    """Return (used_gb, total_gb)"""
    vm = psutil.virtual_memory()
    return vm.used / (1024 ** 3), vm.total / (1024 ** 3)


def estimate_sparse_bytes(nz: int) -> float:
    """Estimate bytes for COO (row, col int64 + value float32)"""
    # 8 bytes each index, 4 bytes for value
    return nz * (2 * 8 + 4)


def create_sparse_coo(N: int, M: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a sparse COO matrix A and dense vector x"""
    coords = []
    vals = []
    for i in range(N):
        for j in range(M):
            if (i + j) % 3 == 0:
                coords.append([i, j])
                vals.append(float(i + j))
    index = torch.tensor(coords, dtype=torch.long).t()
    value = torch.tensor(vals, dtype=torch.float32)
    A = torch.sparse_coo_tensor(index, value, (N, M)).coalesce()
    x = torch.ones((M, 1), dtype=torch.float32, device=A.device)
    return A, x


def compile_cuda(src: Path, exe: Path) -> None:
    cmd = ["nvcc", str(src), "-o", str(exe)]
    subprocess.run(cmd, check=True)


def run_cuda_exe(exe: Path, N: int, M: int, threshold_ratio: float = 0.7) -> float:
    with open(exe.parent / 'main.cu', 'r') as f:
        src = f.read()
    threshold = int(np.floor(N * threshold_ratio))
    src = src.replace('const int N = 1000;', f'const int N = {N};')
    src = src.replace('const int M = 1000;', f'const int M = {M};')
    src = src.replace('const int threshold = 700;', f'const int threshold = {threshold};')
    temp = exe.parent / 'temp.cu'
    temp.write_text(src)
    compile_cuda(temp, exe)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if 'CUDA kernel time:' in line:
            return float(line.split(':')[1].strip().split()[0])
    raise RuntimeError('CUDA timing not found')


def run_torch_sparse(N: int, M: int, iterations: int = 50) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Warm-up & verify
    A, x = create_sparse_coo(N, M)
    A = A.to(device)
    x = x.to(device)
    _ = torch.sparse.mm(A, x)
    torch.cuda.synchronize()

    times = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start_evt.record()
        _ = torch.sparse.mm(A, x)
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))
    torch.cuda.empty_cache()
    return float(np.mean(times) / 1000.0)


def benchmark(sizes: list[tuple[int, int]]) -> dict:
    results = {'size': [], 'cuda': [], 'torch': [], 'mem_usage': []}
    exe = Path('./mainy')
    for N, M in sizes:
        print(f"\nBenchmarking {N}x{M}")
        used, total = get_memory_usage()
        print(f"Memory: {used:.2f}GB / {total:.2f}GB")
        nnz = (N * M) // 3
        est_bytes = estimate_sparse_bytes(nnz)
        print(f"Estimated sparse: {est_bytes/1e9:.2f}GB")

        try:
            cuda_time = run_cuda_exe(exe, N, M)
        except Exception as e:
            print(f"CUDA error: {e}")
            cuda_time = None
        try:
            torch_time = run_torch_sparse(N, M)
        except Exception as e:
            print(f"Torch error: {e}")
            torch_time = None

        results['size'].append((N, M))
        results['cuda'].append(cuda_time)
        results['torch'].append(torch_time)
        results['mem_usage'].append((used, total))

        torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    test_sizes = [(10, 10), (1000, 1000), (2000, 2000), (5000, 5000), (10000, 10000)]
    res = benchmark(test_sizes)
    print("\nResults:")
    for size, ct, tt in zip(res['size'], res['cuda'], res['torch']):
        print(f"Size {size}: CUDA={ct}, Torch={tt}")
