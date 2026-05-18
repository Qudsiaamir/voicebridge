"""Small CuPy smoke test for CUDA-enabled environments."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny CuPy operation to confirm CUDA-backed arrays work."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()

    import cupy as cp

    gpu_array = cp.array([1, 2, 3, 4, 5])
    result = gpu_array * 2
    result_numpy = cp.asnumpy(result)

    print("CuPy array:")
    print(gpu_array)
    print("\nResult on GPU:")
    print(result)
    print("\nResult transferred back to NumPy:")
    print(result_numpy)


if __name__ == "__main__":
    main()
