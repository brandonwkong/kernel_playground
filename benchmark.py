"""
Kernel timing playground — play session 1.

Goal: learn to TIME an op properly and read the result.
No custom kernels yet. You time 3 versions of RMSNorm against each other:

  1. naive_torch   — RMSNorm written from plain PyTorch ops (several kernels)
  2. torch_native  — F.rms_norm (PyTorch's fused implementation)
  3. torch_compile — naive_torch after torch.compile (a compiler tries to fuse it)

Run on a machine with a CUDA GPU:
    python kernel_timing_playground.py

Things to look at in the output:
  - native and compiled should beat naive, especially on big shapes.
    The gap IS the cost of extra kernel launches + extra trips to memory.
  - GB/s: RMSNorm moves ~read input + read weight + write output and does
    almost no math, so it is memory-bound. Compare your GB/s to your GPU's
    advertised memory bandwidth — how close do you get?
  - Change DTYPE to torch.float32 and re-run. Times roughly double but
    GB/s should stay similar: same bandwidth, twice the bytes per element.
    That is what "memory-bound" feels like in numbers.
"""

import torch
import torch.nn.functional as F

DTYPE = torch.float16
EPS = 1e-6

SHAPES = [
    (256, 1024),
    (2048, 4096),   
    (4096, 8192),
    (1024, 16384),
]


def rmsnorm_naive(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RMSNorm from plain ops, computed in float32 like real implementations."""
    x32 = x.float()
    rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (x32 * rms).to(x.dtype) * w


def rmsnorm_native(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), w, EPS)


rmsnorm_compiled = torch.compile(rmsnorm_naive)


def bench_us(fn, x, w, warmup=25, iters=100) -> float:
    """Median microseconds per call, timed with CUDA events (never time.time()
    around GPU work — launches are async, the CPU clock lies)."""
    for _ in range(warmup):
        fn(x, w)
    torch.cuda.synchronize()

    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(x, w)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1000)  # ms -> us
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    if not torch.cuda.is_available():
        print("No CUDA GPU visible. Run this on your GPU machine, not a CPU box.")
        return
    if not hasattr(F, "rms_norm"):
        print("This PyTorch is too old for F.rms_norm (needs ~2.4+). Upgrade torch.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}   dtype: {DTYPE}\n")
    print(f"{'shape':>14} {'naive_us':>10} {'native_us':>10} {'compile_us':>11} {'best_GB/s':>10}")
    print("-" * 60)

    for rows, cols in SHAPES:
        x = torch.randn(rows, cols, device="cuda", dtype=DTYPE)
        w = torch.randn(cols, device="cuda", dtype=DTYPE)

        # correctness first — a fast wrong kernel is worth nothing
        ref = rmsnorm_native(x, w)
        assert torch.allclose(rmsnorm_naive(x, w), ref, atol=1e-2, rtol=1e-2), "naive != native"

        t_naive = bench_us(rmsnorm_naive, x, w)
        t_native = bench_us(rmsnorm_native, x, w)
        t_compile = bench_us(rmsnorm_compiled, x, w)

        best_us = min(t_naive, t_native, t_compile)
        # approx bytes touched: read x, read w, write out (fp16 = 2 bytes/elem)
        nbytes = (rows * cols * 2 + cols + rows * cols) * x.element_size()
        gbs = nbytes / (best_us * 1e-6) / 1e9

        print(f"{str((rows, cols)):>14} {t_naive:>10.1f} {t_native:>10.1f} {t_compile:>11.1f} {gbs:>10.0f}")

    print("\nPlay next:")
    print("  1. Set DTYPE = torch.float32, re-run. What happens to us vs GB/s?")
    print("  2. Add a tiny shape like (8, 1024). Why does everything get slow per-element?")
    print("     (kernel launch overhead dominates when the work is tiny)")
    print("  3. torch.cuda.Event vs time.time(): comment out synchronize and watch it lie.")


if __name__ == "__main__":
    main()
