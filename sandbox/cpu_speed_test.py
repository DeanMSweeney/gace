#!/usr/bin/env python3
"""
cpu_speed_test.py – quick CPU benchmark for
* package-import latency
* dense matrix multiplication throughput (PyTorch)

Usage examples
--------------
python cpu_speed_test.py                                 # default: import torch only
python cpu_speed_test.py --imports torch,numpy,scipy     # time several imports
python cpu_speed_test.py --sizes 1024 2048 --trials 20   # change matrix sizes / trials
"""

import argparse, importlib, time, platform, os, sys
from typing import List

# ----------------------------------------------------------------------------- #
# Utilities
# ----------------------------------------------------------------------------- #
def timed_import(package: str):
    """Import *package* and return (module, elapsed_seconds)."""
    t0 = time.perf_counter()
    module = importlib.import_module(package)
    elapsed = time.perf_counter() - t0
    return module, elapsed

def benchmark(torch_mod, sizes: List[int], trials: int):
    """Time torch.matmul for square matrices on the CPU."""
    print(f"\nDevice: CPU   dtype: float32   trials/size: {trials}")
    torch_mod.manual_seed(0)

    for n in sizes:
        a = torch_mod.randn(n, n)
        b = torch_mod.randn(n, n)

        # Warm-up (helps stabilize timing / JIT)
        for _ in range(5):
            _ = a @ b

        t0 = time.perf_counter()
        for _ in range(trials):
            _ = a @ b
        t1 = time.perf_counter()

        elapsed = (t1 - t0) / trials                 # seconds per multiply
        gflops  = (2.0 * n ** 3) / (elapsed * 1e9)   # 2·n³ floating-point ops
        print(f"n={n:<5}  time={elapsed*1e3:>8.3f} ms   throughput={gflops:>8.2f} GFLOPS")

# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="CPU import-time & matmul benchmark")
    parser.add_argument(
        "--imports",
        default="torch",
        help="comma-separated list of packages to time importing (default: torch)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="square matrix sizes to benchmark",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="matmul repetitions per size (default: 10)",
    )
    args = parser.parse_args()

    print(f"Python {platform.python_version()}  |  Host {platform.node()}  |  PID {os.getpid()}")

    # ------------------------------------------------------------------ #
    # 1. Time package imports
    # ------------------------------------------------------------------ #
    print("\n--- Import-time measurement ---")
    modules = {}
    for pkg in [p.strip() for p in args.imports.split(",") if p.strip()]:
        try:
            mod, elapsed = timed_import(pkg)
            modules[pkg] = mod
            print(f"import {pkg:<10}: {elapsed*1e3:7.2f} ms")
        except ModuleNotFoundError:
            print(f"import {pkg:<10}: FAILED (not installed)")
    if "torch" not in modules:
        print("\n❌  Torch is required for the matmul benchmark. Exiting.")
        sys.exit(1)

    torch_mod = modules["torch"]

    # ------------------------------------------------------------------ #
    # 2. Matrix-multiplication benchmark
    # ------------------------------------------------------------------ #
    print("\n--- Matrix-multiplication benchmark ---")
    benchmark(torch_mod, sizes=args.sizes, trials=args.trials)

if __name__ == "__main__":
    main()

