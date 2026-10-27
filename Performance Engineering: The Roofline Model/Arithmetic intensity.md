# Arithmetic Intensity: From FLOPs per Byte to GPU and LLM Inference Optimization

> A complete, beginner-to-advanced guide to arithmetic intensity, memory bandwidth, the Roofline model, matrix multiplication, CUDA optimization, transformer prefill and decode, attention, quantization, and practical PyTorch GPU profiling.

## Table of Contents

1. [Why arithmetic intensity matters](#1-why-arithmetic-intensity-matters)
2. [Foundations: bits, bytes, floating-point operations, and bandwidth](#2-foundations-bits-bytes-floating-point-operations-and-bandwidth)
3. [Definition and careful accounting](#3-definition-and-careful-accounting)
4. [Memory-bound and compute-bound execution](#4-memory-bound-and-compute-bound-execution)
5. [The Roofline performance model](#5-the-roofline-performance-model)
6. [Worked calculations: vector operations and reductions](#6-worked-calculations-vector-operations-and-reductions)
7. [Matrix multiplication and data reuse](#7-matrix-multiplication-and-data-reuse)
8. [GPU tiling, registers, caches, and shared memory](#8-gpu-tiling-registers-caches-and-shared-memory)
9. [Operator fusion and memory traffic](#9-operator-fusion-and-memory-traffic)
10. [Transformer inference: prefill versus decoding](#10-transformer-inference-prefill-versus-decoding)
11. [Continuous batching and request scheduling](#11-continuous-batching-and-request-scheduling)
12. [Attention, FlashAttention, and the KV cache](#12-attention-flashattention-and-the-kv-cache)
13. [Quantization and arithmetic intensity](#13-quantization-and-arithmetic-intensity)
14. [Hierarchical Roofline, cache effects, and limitations](#14-hierarchical-roofline-cache-effects-and-limitations)
15. [A complete Windows/PyTorch CUDA experiment](#15-a-complete-windowspytorch-cuda-experiment)
16. [Interpreting results and profiling with Nsight Compute](#16-interpreting-results-and-profiling-with-nsight-compute)
17. [Production engineering decisions](#17-production-engineering-decisions)
18. [Common misconceptions and interview questions](#18-common-misconceptions-and-interview-questions)
19. [Mathematical cheat sheet](#19-mathematical-cheat-sheet)
20. [References and further reading](#20-references-and-further-reading)

---

## 1. Why arithmetic intensity matters

A GPU can be extremely fast at arithmetic while executing a particular kernel slowly. The reason is that arithmetic units cannot operate without inputs. Moving tensors and model weights from GPU device memory into caches, shared memory, and registers takes time and consumes finite bandwidth. **Arithmetic intensity** helps quantify whether each byte of data movement is supporting enough useful computation to keep the arithmetic hardware busy.

The core question is not merely “How many FLOPs does this model require?” but **“How many FLOPs are performed per byte moved across the memory interface we are studying?”** This matters when choosing between larger GPUs, faster memory, weight quantization, batching, kernel fusion, FlashAttention, and more efficient matrix multiplication kernels. It is particularly consequential in LLM inference because processing an entire prompt and generating a single next token create very different patterns of weight reuse.

## 2. Foundations: bits, bytes, floating-point operations, and bandwidth

A **bit** is one binary digit. A **byte** consists of eight bits. Standard numerical storage formats include FP32 (4 bytes per scalar), FP16 and BF16 (2 bytes), and INT8 (1 byte). Packed 4-bit weights nominally occupy half a byte each, before padding and any quantization metadata. A tensor's storage size is its number of elements multiplied by the number of bytes stored per element.

A **floating-point operation**, abbreviated **FLOP**, is one mathematical arithmetic operation on floating-point numbers, under a specified counting convention. Multiplication and addition each normally count as one FLOP, so a fused multiply-add `a * b + c` counts as **two FLOPs** even though hardware may issue it as one FMA instruction. Hardware comparisons, exponentials, integer arithmetic, and quantized operations cannot always be compared using the same simple FLOP convention. FLOPs measure *work*, while **FLOP/s** measure *throughput*: 1 GFLOP/s is 10^9 operations per second and 1 TFLOP/s is 10^12 operations per second.

**Bandwidth** is a rate of transferring bytes. For example, 1 TB/s means 10^12 bytes/s under the decimal convention. **Latency** is how long a particular operation or memory request takes. A device can have high bandwidth but still suffer high per-request latency, poor coalescing, or too little concurrency to use its bandwidth effectively.

**Important distinction:** a 100 MB tensor does not imply 100 MB of actual DRAM traffic. Reading it ten times without caching can yield approximately 1 GB of reads; retaining it in cache and reusing it may substantially reduce DRAM traffic. Likewise, allocation size and peak VRAM usage are **not** measurements of transferred bytes.

## 3. Definition and careful accounting

Arithmetic intensity is defined as

$$
\boxed{I=\frac{F}{Q}}
$$

where $I$ is arithmetic intensity in FLOPs/byte, $F$ is the number of floating-point operations, and $Q$ is the number of bytes transferred across a **specified memory interface**.

For a computation that performs 1,000 FLOPs and moves 200 bytes, intensity is 5 FLOPs/byte. Performing the same mathematics with only 50 bytes of movement increases intensity to 20 FLOPs/byte. This is often achievable through **data reuse**: once data is loaded into a register, cache, or shared-memory tile, several arithmetic operations can use it without repeated accesses to external memory.

There are at least three different and useful quantities. **Algorithmic/theoretical intensity** derives bytes moved from a stated implementation assumption; **ideal compulsory-traffic intensity** assumes each unique input is read and each unique output written only once; **measured intensity** uses actual hardware memory-traffic counters. All can be valid, but they answer different questions. Every serious analysis should explicitly state precision, counted operations, memory boundary (DRAM, L2, L1, etc.), whether output reads/writes are included, and whether the bytes are assumed or measured.

## 4. Memory-bound and compute-bound execution

A **memory-bandwidth-bound** kernel is limited mainly by the rate at which it can move its required bytes. A **compute-bound** kernel is limited mainly by arithmetic execution capacity. These categories depend on the hardware and kernel configuration; they are not inherent, permanent labels assigned to an operation.

Consider a hypothetical accelerator capable of **100 TFLOP/s** of a particular floating-point operation and **1 TB/s** of DRAM bandwidth. A kernel at 2 FLOPs/byte can ideally sustain no more than $2\times1=2$ TFLOP/s through the DRAM bandwidth ceiling. A kernel at 200 FLOPs/byte could theoretically support 200 TFLOP/s from its memory traffic, but the GPU's compute peak caps it at 100 TFLOP/s. These are **upper bounds**, not guaranteed speeds.

For a workload with 10 GFLOPs and 10 GB of memory movement, compute time has an ideal lower bound of 0.1 ms and memory time a lower bound of 10 ms on that accelerator. Memory is the obvious first optimization target. For 1 TFLOP of work and 2 GB of traffic, the compute lower bound is 10 ms while the memory lower bound is 2 ms, so arithmetic capacity is the more restrictive theoretical resource.

## 5. The Roofline performance model

The **Roofline model** formalizes the relationship between computation and data movement:

$$
\boxed{P_{\mathrm{achievable}}\ \leq\ \min\left(P_{\mathrm{peak}},\ I\, B_{\mathrm{peak}}\right)}
$$

Here $P$ is performance in FLOP/s, $I$ is arithmetic intensity in FLOPs/byte, and $B$ is bandwidth in bytes/s. The horizontal part of a conventional Roofline plot is the compute ceiling. The upward-sloping part represents the bandwidth ceiling. Roofline diagrams commonly use log-log axes.

The **ridge point** is the intensity where the two limits intersect:

$$
\boxed{I_{\mathrm{ridge}}=\frac{P_{\mathrm{peak}}}{B_{\mathrm{peak}}}}
$$

For our 100 TFLOP/s and 1 TB/s hypothetical accelerator, the ridge is **100 FLOPs/byte**. An algorithm with 10 FLOPs/byte has a DRAM bandwidth ceiling of 10 TFLOP/s; an algorithm with 500 FLOPs/byte has a compute ceiling of 100 TFLOP/s.

| Arithmetic intensity | Bandwidth ceiling | Compute ceiling | Roofline upper bound |
|---:|---:|---:|---:|
| 1 FLOP/byte | 1 TFLOP/s | 100 TFLOP/s | 1 TFLOP/s |
| 10 FLOPs/byte | 10 TFLOP/s | 100 TFLOP/s | 10 TFLOP/s |
| 50 FLOPs/byte | 50 TFLOP/s | 100 TFLOP/s | 50 TFLOP/s |
| 100 FLOPs/byte | 100 TFLOP/s | 100 TFLOP/s | 100 TFLOP/s |
| 200 FLOPs/byte | 200 TFLOP/s | 100 TFLOP/s | 100 TFLOP/s |

**Precision must match the roof.** The FP32 CUDA-core limit is generally not the same as the FP16/BF16 Tensor Core limit. If a workload uses Tensor Cores, use a matching peak throughput. Moreover, kernels can run below both ceilings because of memory latency, instruction dependencies, launch overhead, occupancy, synchronization, or unfavorable instruction mixes. A kernel below the ridge is *predicted* to be bandwidth-limited by this simplified model—not proven to achieve peak bandwidth in practice.

## 6. Worked calculations: vector operations and reductions

### FP32 vector addition

For each element of $C_i=A_i+B_i$, assume one 4-byte read from A, one 4-byte read from B, and one 4-byte write to C, with one floating-point addition:

$$
I_{\mathrm{add}}=\frac{1}{4+4+4}=\boxed{\frac{1}{12}\approx0.0833\ \mathrm{FLOPs/byte}}
$$

A large streaming vector addition is typically memory-bandwidth-sensitive. If the inputs are resident in cache, however, DRAM-level intensity may differ from this streaming estimate.

### FP32 multiply-add over separate arrays

For $D_i=A_iB_i+C_i$, count one multiply and one add, three 4-byte input reads, and one 4-byte output write:

$$
I=\frac{2}{4+4+4+4}=\boxed{0.125\ \mathrm{FLOPs/byte}}
$$

### FP32 dot product

For $s=\sum_{i=1}^{N}A_iB_i$, approximate the work as $2N$ FLOPs and traffic as $8N$ input bytes, ignoring the final scalar write:

$$
I_{\mathrm{dot}}\approx\frac{2N}{8N}=\boxed{0.25\ \mathrm{FLOPs/byte}}
$$

### Why adding more arithmetic may not make an operation compute-bound

The denominator matters. An operator might double its FLOPs but also double its input/output traffic, leaving intensity unchanged. Conversely, an optimization might do *more* arithmetic but remove sufficiently many global memory transfers to execute faster overall.

## 7. Matrix multiplication and data reuse

Let $A,B\in\mathbb{R}^{N\times N}$ and $C=AB$. Each of the $N^2$ outputs requires roughly $N$ multiplications and $N$ additions, so:

$$
F\approx 2N^3.
$$

Under idealized **FP32** compulsory traffic, each input is read once and the output written once:

$$
Q_{\mathrm{ideal}}=4N^2+4N^2+4N^2=12N^2\ \mathrm{bytes}.
$$

Thus:

$$
\boxed{I_{\mathrm{ideal}}\approx\frac{2N^3}{12N^2}=\frac{N}{6}\ \mathrm{FLOPs/byte}}
$$

| Square dimension $N$ | Approximate FLOPs | Minimum FP32 input/output traffic | Ideal intensity |
|---:|---:|---:|---:|
| 64 | 0.524 million | 49,152 bytes | 10.7 FLOPs/byte |
| 256 | 33.55 million | 786,432 bytes | 42.7 FLOPs/byte |
| 1,024 | 2.147 billion | 12.58 MB | 170.7 FLOPs/byte |
| 4,096 | 137.44 billion | 201.33 MB | 682.7 FLOPs/byte |

Unlike vector addition, matrix multiplication can reuse each input value in many output calculations. This gives it a much higher *potential* arithmetic intensity as matrix dimensions increase.

But merely writing a triple-nested-loop matrix multiplication does not guarantee ideal DRAM reuse. A pessimistic implementation that reloads operands for every scalar multiply might transfer on the order of $8N^3$ input bytes, giving an intensity near 0.25 FLOPs/byte. Actual hardware caches partly mitigate this, and optimized GEMM kernels deliberately manage reuse.

### General rectangular GEMM

For $A\in\mathbb{R}^{M\times K}$, $B\in\mathbb{R}^{K\times N}$, and $C\in\mathbb{R}^{M\times N}$, a first-order formula is:

$$
F\approx2MKN,
\qquad Q_{\mathrm{ideal}}=s(MK+KN+MN),
$$

$$
\boxed{I_{\mathrm{ideal}}\approx\frac{2MKN}{s(MK+KN+MN)}}
$$

where $s$ is bytes per scalar. If GEMM has a beta term that reads an existing C matrix ($C=\alpha AB+\beta C$), its C-read traffic must be included. This general formula is especially useful for analyzing transformer projection shapes and batch sizes.

## 8. GPU tiling, registers, caches, and shared memory

A tiled matrix multiplication loads submatrices into on-chip storage and reuses their elements for many multiply-add operations before fetching new tiles. For two $T\times T$ input tiles, one tiled multiply performs roughly $2T^3$ FLOPs. Reading each input tile once requires $2T^2s$ bytes. Ignoring output traffic and additional tile reuse,

$$
I_{\mathrm{tile}}\approx\frac{2T^3}{2T^2s}=\frac{T}{s}.
$$

With FP32 ($s=4$), a 64-element tile implies about 16 FLOPs per input byte, and a 128-element tile about 32 FLOPs per input byte for this isolated tile calculation. Real kernels use non-square tiles, loop over the K dimension, accumulate C in registers, and write C when the output tile is complete.

Larger tiles are not always better. Registers and shared memory are finite, and oversized tiles can reduce the number of resident warps, increase register pressure, or cause spills into slower local/device memory. Production kernels balance tiling, occupancy, tensor-core shape alignment, cache locality, launch configuration, and synchronization overhead.

## 9. Operator fusion and memory traffic

Consider $Y=\operatorname{ReLU}(A+B)$ using FP32 tensors. An unfused design might read A and B (8 bytes), write an intermediate T (4 bytes), read T again (4 bytes), and write Y (4 bytes), totaling **20 nominal bytes per element**. A fused implementation reads A and B and writes Y directly, totaling **12 nominal bytes per element**—a **40% reduction in the assumed memory traffic**.

The exact FLOP definition for ReLU is not necessarily one FLOP because its comparison is often counted separately from floating-point arithmetic, but fusion's data-movement benefit remains. Fusing bias, activation, normalization, residual, and other adjacent operations can improve effective arithmetic intensity and reduce kernel launch overhead. It may also be counterproductive if the fused kernel uses too many registers or sacrifices parallelism.

## 10. Transformer inference: prefill versus decoding

A transformer block contains dense projections, attention, normalization, and an MLP. Their different memory patterns mean they have very different intensities. The particularly useful distinction is between **prefill**—processing an existing prompt—and **decoding**, which typically generates one new token per active sequence per step.

### Prefill: many input tokens share a weight matrix

Suppose a BF16 or FP16 projection has $W\in\mathbb{R}^{H\times H}$ and input $X\in\mathbb{R}^{S\times H}$. Computing $Y=XW$ requires approximately $2SH^2$ FLOPs. Reading X and W once and writing Y once requires $2SH+2H^2+2SH=2H^2+4SH$ bytes, hence:

$$
\boxed{I_{\mathrm{prefill}}\approx\frac{2SH^2}{2H^2+4SH}}
$$

For $H=4096$ and $S=1024$, this is approximately **682.7 FLOPs/byte**, with 34.36 GFLOPs and 50.33 MB of ideal compulsory traffic. It is a highly reusable matrix-multiplication workload.

### Single-sequence decode: one token shares little weight reuse

With $S=1$, the same expression becomes a matrix-vector multiply:

$$
I_{\mathrm{decode}}\approx\frac{2H^2}{2H^2+4H}\approx\boxed{1\ \mathrm{FLOP/byte}}
$$

For $H=4096$, the projected layer performs about 33.55 MFLOPs while its BF16 weight matrix alone occupies about 33.55 MB. This makes **single-sequence decoder projections** natural candidates for bandwidth-bound behavior. This does not prove the *entire LLM decode step* has intensity exactly one: attention, KV-cache reads, normalization, weight/cache residency, and launch overhead also matter.

The contrast explains a common production pattern: prefill can be compute-intensive, while low-batch decoding can be bandwidth-sensitive. The same model can bottleneck on different resources at different stages of serving.

## 11. Continuous batching and request scheduling

Continuous or in-flight batching executes token steps for multiple active requests together, allowing them to share model-weight reads. With $B$ concurrent decode tokens and a $H\times H$ BF16 projection:

$$
\boxed{I_{\mathrm{batched\ decode}}\approx\frac{2BH^2}{2H^2+4BH}}
$$

For $H=4096$, a batch of 1 has intensity near **1**, while a batch of 32 has intensity near **31.5 FLOPs/byte**. The weight bytes are approximately constant under the idealized assumption that the weights are read once for the batch, while useful arithmetic grows with $B$.

This gives schedulers a hardware reason to combine active decode sequences, but throughput improvement is not indefinitely linear. Larger batches consume more activation and KV-cache memory, produce more attention traffic, can create scheduling interference, and can increase individual request latency. In real serving engines, admission control and scheduling policies therefore balance **throughput, token latency, memory capacity, and fairness** rather than simply maximizing batch size.

Paged KV-cache allocators can reduce fragmentation and allow flexible sequence management, but they also introduce metadata and access-pattern considerations. Chunked prefill can improve the coexistence of long prompts and decoding tasks; disaggregated prefill/decode separates phases with different hardware pressure, but introduces KV transfer and network costs. The arithmetic-intensity model helps explain the motivation, while end-to-end performance still requires measuring scheduling and communication overhead.

## 12. Attention, FlashAttention, and the KV cache

Scaled dot-product attention is

$$
\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V.
$$

For one head with sequence length $S$ and head dimension $d$, the $QK^T$ and subsequent probability-times-V matrix multiplications together require approximately $4S^2d$ FLOPs, ignoring softmax operations. A straightforward implementation may materialize an $S\times S$ score matrix, then reread it for normalization and additional work. At $S=4096$, one FP16 score matrix occupies approximately **33.55 MB** for a single head; at $S=16384$, it occupies approximately **536.87 MB**.

**FlashAttention** uses tiling and online softmax to avoid storing the full quadratic attention matrix in GPU DRAM. It changes the memory-I/O pattern, not the fundamental quadratic amount of standard full attention computation. This often increases effective reuse and improves performance. During training, recomputing some intermediates in the backward pass can be beneficial if it avoids materializing large memory buffers.

### Cached attention in one-token decoding

Let a single query attend over $L$ cached key/value vectors of dimension $d$. The two dominant matrix-vector-style operations require approximately $4Ld$ FLOPs. Reading K and V in a format of $s$ bytes per element moves approximately $2Lds$ bytes:

$$
\boxed{I_{\mathrm{single\ query\ KV}}\approx\frac{4Ld}{2Lds}=\frac{2}{s}}
$$

For BF16/FP16 ($s=2$), this is roughly **1 FLOP/byte**, not counting other reads/writes. It helps explain why long-context single-query decode attention is often bandwidth-sensitive. Grouped-query attention (GQA) and multi-query attention (MQA) reduce the number of stored K/V heads; when multiple query heads can reuse those K/V loads, effective intensity can increase. Actual memory traffic also depends on page layout, cache locality, head grouping, kernel design, and the number of active sequences.

## 13. Quantization and arithmetic intensity

By storing model weights in fewer bytes, quantization reduces the amount of data that must be transferred. For a large matrix-vector multiply where weight movement dominates, an approximate count of **mathematical operations per stored weight byte** is $2/s$:

| Weight storage | Approx. stored bytes per weight | Equivalent operations per weight byte |
|---|---:|---:|
| FP32 | 4 | 0.5 |
| FP16/BF16 | 2 | 1 |
| INT8 | 1 | 2 |
| Packed INT4 | 0.5 | 4 |

**Caution:** integer and mixed-precision quantized kernels do not necessarily execute floating-point FLOPs matching this table. Real kernels incur packing, scale/zero-point metadata, dequantization, mixed-precision accumulation, and possibly different tensor-core throughput. The table isolates **storage traffic**; it does not guarantee a 4x speedup when changing from FP16 to INT4.

Quantization often benefits low-batch decoding because weight bandwidth is important there. Prefill can also benefit from smaller weights and specialized low-precision compute, but may already be compute-limited in some regimes. Quantizing the KV cache is a separate strategy that targets cached-attention capacity and traffic rather than projection-weight traffic.

## 14. Hierarchical Roofline, cache effects, and limitations

GPUs have registers, shared memory, L1 cache, L2 cache, and DRAM. A kernel can have one intensity at the DRAM interface and another at an inner cache interface:

$$
I_{\mathrm{DRAM}}=\frac{F}{Q_{\mathrm{DRAM}}},\qquad I_{\mathrm{L2}}=\frac{F}{Q_{\mathrm{L2}}}.
$$

For example, if a kernel performs 1 billion FLOPs and moves 100 MB at DRAM but 400 MB through an L2-related interface, then $I_{\mathrm{DRAM}}=10$ FLOPs/byte and $I_{\mathrm{L2}}=2.5$ FLOPs/byte. The two values are not contradictory: they measure different boundaries. A hierarchical Roofline approach can identify whether the binding resource is DRAM, cache bandwidth, or compute throughput.

Roofline alone does not capture all performance pathologies. Poor memory coalescing can cause extra transactions; insufficient occupancy can fail to hide latency; register spills add traffic; tiny kernels may be launch-limited; synchronization and instruction dependencies can limit utilization. To know the true bottleneck, first use Roofline as a **hypothesis**, then test it with profiler counters and controlled benchmarks.

## 15. A complete Windows/PyTorch CUDA experiment

This experiment compares streaming FP32 vector addition with FP32 dense matrix multiplication. It reports an **analytical FLOP count and idealized bytes moved**, then measures elapsed GPU time. The computed bandwidth is **effective bandwidth under the traffic assumption**, **not actual measured DRAM throughput**. For real DRAM traffic, use hardware performance counters in Nsight Compute.

### Requirements and setup

Use Windows 11, Python 3.10+ or an otherwise PyTorch-supported Python version, a compatible NVIDIA GPU driver, and a CUDA-enabled PyTorch build. Choose the appropriate installation command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). No Redis, Docker, WSL, or hosted inference service is required.

```powershell
# Verify the active environment and CUDA availability.
python --version
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

Save the following program as `arithmetic_intensity_benchmark.py`:

```python
"""Demonstrate theoretical arithmetic intensity and measure CUDA execution time.

The program measures elapsed kernel/operator execution using CUDA events.
Its byte counts represent idealized compulsory input/output traffic, not
hardware DRAM transaction counters. Run Nsight Compute for measured traffic.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from dataclasses import dataclass
from typing import Callable

import torch

LOGGER = logging.getLogger("arithmetic_intensity")


@dataclass(frozen=True)
class Result:
    """One benchmark's mathematical work, assumed traffic, and GPU timing."""

    name: str
    flops: int
    assumed_bytes: int
    median_ms: float

    @property
    def ai(self) -> float:
        """Return theoretical FLOPs per assumed byte."""
        if self.assumed_bytes <= 0:
            raise ValueError("assumed_bytes must be positive")
        return self.flops / self.assumed_bytes

    @property
    def tflops(self) -> float:
        """Return achieved throughput from the theoretical FLOP count."""
        if self.median_ms <= 0:
            raise ValueError("median_ms must be positive")
        return (self.flops / 1e12) / (self.median_ms / 1e3)

    @property
    def effective_gbs(self) -> float:
        """Return bandwidth based on assumed, not measured, transferred bytes."""
        if self.median_ms <= 0:
            raise ValueError("median_ms must be positive")
        return (self.assumed_bytes / 1e9) / (self.median_ms / 1e3)


def time_cuda_operation(
    operation: Callable[[], torch.Tensor], *, warmup: int, repeats: int
) -> float:
    """Return median device elapsed time in milliseconds using CUDA events."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch build and GPU are required")

    # Warm up operators and any required library/kernel initialization.
    for _ in range(warmup):
        _ = operation()
    torch.cuda.synchronize()

    elapsed_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = operation()
        end.record()
        end.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))

    return statistics.median(elapsed_ms)


def bench_vector_add(size: int, warmup: int, repeats: int) -> Result:
    """Benchmark one FP32 addition per element with nominal 12 bytes of I/O."""
    if size <= 0:
        raise ValueError("Vector length must be positive")

    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)

    # Use an out= tensor to avoid allocating a new output on every trial.
    elapsed = time_cuda_operation(
        lambda: torch.add(a, b, out=out), warmup=warmup, repeats=repeats
    )
    torch.testing.assert_close(out, a + b)
    return Result("FP32 vector addition", size, 3 * size * 4, elapsed)


def bench_matmul(n: int, warmup: int, repeats: int) -> Result:
    """Benchmark FP32 NxN GEMM with ideal 3-matrix compulsory traffic."""
    if n <= 0:
        raise ValueError("Matrix dimension must be positive")

    a = torch.randn((n, n), device="cuda", dtype=torch.float32)
    b = torch.randn((n, n), device="cuda", dtype=torch.float32)
    out = torch.empty((n, n), device="cuda", dtype=torch.float32)

    # Do not silently vary TF32 settings across benchmark runs.
    # Disabling TF32 favors consistent FP32 arithmetic semantics, though
    # kernel selection and attainable peak depend on the GPU architecture.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False

    elapsed = time_cuda_operation(
        lambda: torch.mm(a, b, out=out), warmup=warmup, repeats=repeats
    )
    if not torch.isfinite(out).all().item():
        raise RuntimeError("Matrix multiplication generated non-finite output")

    return Result("FP32 matrix multiplication", 2 * n**3, 3 * n**2 * 4, elapsed)


def main() -> int:
    """Validate inputs, execute the benchmarks, and log performance metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-size", type=int, default=2_000_000)
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.vector_size <= 0 or args.matrix_size <= 0:
            raise ValueError("Vector and matrix sizes must be positive")
        if args.warmup < 0 or args.repeats <= 0:
            raise ValueError("warmup must be nonnegative and repeats positive")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: verify the PyTorch build/driver")

        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))
        benchmarks = (
            bench_vector_add(args.vector_size, args.warmup, args.repeats),
            bench_matmul(args.matrix_size, args.warmup, args.repeats),
        )
        for item in benchmarks:
            LOGGER.info("%s", item.name)
            LOGGER.info("  FLOPs: %s", f"{item.flops:,}")
            LOGGER.info("  Assumed traffic: %.3f MB", item.assumed_bytes / 1e6)
            LOGGER.info("  Theoretical AI: %.3f FLOPs/byte", item.ai)
            LOGGER.info("  Median CUDA time: %.4f ms", item.median_ms)
            LOGGER.info("  Achieved arithmetic rate: %.4f TFLOP/s", item.tflops)
            LOGGER.info("  Effective bandwidth (assumed traffic): %.3f GB/s", item.effective_gbs)
        return 0
    except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        LOGGER.error("Benchmark failed: %s", exc)
        return 1
    except Exception:
        LOGGER.exception("Unexpected benchmark failure")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Run it from PowerShell:

```powershell
python .\arithmetic_intensity_benchmark.py

# Optional: change the amount of work for your GPU's VRAM capacity.
python .\arithmetic_intensity_benchmark.py --vector-size 4000000 --matrix-size 2048 --warmup 15 --repeats 50
```

**What to expect:** The vector addition has a theoretical arithmetic intensity of about **0.0833 FLOPs/byte**, independent of vector length under the one-read/one-write assumption. Square FP32 matrix multiplication with $N=1024$ has an ideal compulsory-traffic intensity of about **170.7 FLOPs/byte**. The measured runtimes and throughput vary with GPU clock speed, thermal state, library selection, concurrent workloads, tensor sizes, and caching. These values are intentionally **not fabricated** in this guide: you should run the code and examine actual device timings.

**Important benchmark caveat:** repeated vector operations over arrays that fit partially or entirely in L2 cache may show an effective bandwidth computed from assumed DRAM bytes that exceeds physical DRAM specifications; that does not imply impossible DRAM performance. It means the assumed byte denominator is not the *measured* DRAM traffic. Increase working-set sizes within your VRAM budget, rotate buffers carefully, and consult Nsight Compute counters for definitive DRAM-level measurements. The CUDA-event interval also includes relevant device-side execution overhead; for tiny operations, benchmark launch and synchronization effects separately.

## 16. Interpreting results and profiling with Nsight Compute

Start by writing down the workload's theoretical FLOP count and a transparent traffic model. Run enough warmup iterations to avoid first-use initialization costs. Measure several repetitions and report a median, while also retaining variability information for production benchmarking. Avoid comparing kernels under different precision modes, thermal conditions, or background loads.

**PyTorch Profiler** is useful for operator timing, shapes, execution traces, and supported FLOP estimates, but its memory-allocation metrics are not cumulative GPU DRAM traffic counters. `torch.cuda.max_memory_allocated()` similarly reports peak allocator usage, not bytes transferred. For kernel-level memory traffic and Roofline analysis, use **NVIDIA Nsight Compute**, when supported by your GPU, driver, and installed tooling.

Nsight Compute can present Roofline analysis and hardware-counter-based views of memory throughput and compute utilization. Select the correct arithmetic precision and inspect DRAM, L2, occupancy, stalls, and tensor-core utilization, rather than attributing all low throughput to “memory bottlenecks.” The profiler may replay kernels and perturb timings; treat profiler measurements and production end-to-end latency as complementary evidence.

A simple diagnostic workflow is: determine operation shapes and precision, estimate ideal bytes and FLOPs, benchmark with CUDA events, profile actual memory and compute utilization, hypothesize a bottleneck, change exactly one algorithmic feature, and rerun both a correctness test and the performance experiment. Examples of changes include increasing batched decode occupancy, fusing a residual operation, changing GEMM tile sizes, using a more efficient attention kernel, or quantizing stored weights.

## 17. Production engineering decisions

When evaluating LLM serving, do not optimize for arithmetic intensity alone. **Throughput**, **time to first token (TTFT)**, **time per output token (TPOT)**, **end-to-end latency**, **requests per second**, **GPU cost per million output tokens**, and **memory utilization** can favor different policies. Batching generally increases weight reuse, but may worsen latency for isolated or high-priority requests. Large prefill batches make arithmetic units busy but can delay already-running decode steps unless chunking and scheduling are carefully managed.

Memory efficiency also affects reliability. Admission control should calculate an enforceable KV-cache budget, account for model weights and runtime workspaces, and avoid memory over-commitment. Quantized caches and paged allocation may improve the feasible number of active sequences, but must be evaluated for numerical quality, fragmentation, allocator overhead, and actual GPU kernel support. Disaggregated prefill/decode can isolate compute-heavy and bandwidth-heavy phases, yet transferring KV blocks adds networking, serialization, placement, and failure-recovery complexity.

Production deployments also require security and observability. Use least-privilege IAM for inference services and profiling infrastructure, encrypt sensitive requests and transport traffic, restrict GPU profiling and trace artifacts because prompts and tensor dumps can be confidential, and establish retention policies for logs and profiles. Monitor token throughput, TTFT/TPOT distributions, GPU utilization, DRAM bandwidth, HBM/VRAM pressure, cache hit rates, OOM incidents, admission queue length, and serving errors. Autoscaling policies should account for both GPU compute pressure and memory footprint, while request routing should consider model residency and KV-cache locality. A scalable system is one that remains stable under mixed short/long prompts and bursty decode traffic, not merely one that reports a high synthetic benchmark TFLOP/s.

### Selecting an optimization from the bottleneck

| Observed situation | Candidate intervention | Trade-off to verify |
|---|---|---|
| Low-batch decode reads many weights per token | Continuous batching, weight quantization | Latency, KV capacity, numerical quality |
| Prefill GEMM fails to use tensor cores | Shape/layout alignment, kernel library selection | Padding, precision, kernel support |
| Large attention intermediate traffic | FlashAttention or equivalent fused attention | Kernel constraints, sequence/head shapes |
| Elementwise kernels repeatedly materialize tensors | Operator fusion | Register pressure, compilation complexity |
| High theoretical intensity but poor throughput | Nsight occupancy/stall investigation | Launch overhead, dependencies, layout |
| Long contexts strain KV-cache bandwidth | GQA/MQA, KV quantization, efficient paged attention | Accuracy, memory layout, kernel support |
| Mixed prompt and decode contention | Chunked prefill or separate prefill/decode pools | Scheduling fairness, KV transfer overhead |

## 18. Common misconceptions and interview questions

### Does higher arithmetic intensity always mean faster code?

No. It increases the amount of computation per byte of traffic but may add synchronization, dependencies, resource pressure, or extra instructions. On the compute-limited side of the Roofline, increasing intensity without reducing FLOPs or improving hardware utilization may not improve speed at all.

### Is arithmetic intensity the same as FLOP/s?

No. Arithmetic intensity is **FLOPs divided by bytes**; FLOP/s is **FLOPs divided by time**. The Roofline model connects them through available memory bandwidth and peak compute throughput.

### Does a small tensor always have low arithmetic intensity?

No. An algorithm can reuse a small tensor many times while performing extensive arithmetic. Conversely, a huge streaming operation may process each value once and have low intensity. Input size alone does not determine intensity.

### Can arithmetic intensity exceed the ridge point and still perform poorly?

Yes. Kernels can be limited by instruction issue rate, register dependencies, occupancy, launch overhead, or other non-Roofline factors. The roof is an **upper bound**, not a prediction of exact elapsed time.

### Is prefill always compute-bound and decoding always memory-bound?

No. This is a useful tendency for certain model sizes and hardware, especially low-batch decode and large, efficiently batched prefill. Actual behavior depends on batch size, prompt length, precision, model architecture, attention kernel, cache residency, and accelerator characteristics.

### Why does batching improve decode arithmetic intensity?

Because one loaded weight can contribute to multiple token projections. This increases the FLOP numerator faster than the idealized weight-traffic denominator, until activation traffic and other system limits become significant.

### Is peak VRAM usage the denominator in arithmetic intensity?

No. Peak VRAM usage is a capacity metric; bytes transferred are a traffic metric. The same memory allocation can be read or written many times, and caches can suppress external reads.

### Is INT4 automatically four times faster than FP16?

No. INT4 weights have approximately four times less nominal storage per weight, but runtime speed depends on packing, dequantization, instruction support, non-weight traffic, and whether bandwidth was the original bottleneck.

## 19. Mathematical cheat sheet

| Concept | Formula | Assumptions |
|---|---|---|
| Arithmetic intensity | $I=F/Q$ | Choose a memory boundary |
| Roofline upper bound | $P\le\min(P_{\mathrm{peak}},I B_{\mathrm{peak}})$ | Matching precision and bandwidth |
| Ridge point | $I_{\mathrm{ridge}}=P_{\mathrm{peak}}/B_{\mathrm{peak}}$ | Relevant compute peak |
| Memory time lower bound | $T_{\mathrm{mem}}\ge Q/B_{\mathrm{peak}}$ | Peak bandwidth achievable |
| Compute time lower bound | $T_{\mathrm{compute}}\ge F/P_{\mathrm{peak}}$ | Peak arithmetic throughput achievable |
| FP32 vector add | $1/12\approx0.0833$ FLOPs/byte | 2 loads and 1 store |
| FP32 dot product | $\approx 1/4$ FLOP/byte | Streaming 2-vector inputs |
| Rectangular GEMM | $2MKN/[s(MK+KN+MN)]$ | Ideal compulsory traffic |
| Square FP32 GEMM | $N/6$ FLOPs/byte | Ideal compulsory traffic |
| BF16 square projection, S tokens | $2SH^2/(2H^2+4SH)$ | Read X and W, write Y once |
| BF16 single-token projection | $\approx1$ FLOP/byte | Weight traffic dominates |
| BF16 batched token projection | $2BH^2/(2H^2+4BH)$ | Shared weight read for B tokens |
| Standard attention main matmuls | $\approx4S^2d$ FLOPs | One head, excludes softmax |
| Single-query KV-attention intensity | $\approx2/s$ | Streaming K and V; one query |

The key mental model is straightforward: **FLOPs quantify how much work you want to do; bytes quantify how much information you must transport; arithmetic intensity tells you how effectively the transport supports that work.** The Roofline model then connects this ratio to the hardware's available bandwidth and arithmetic throughput.

## 20. References and further reading

The following primary documentation and research papers are useful for validating the assumptions and learning the relevant profiling techniques:

1. Williams, Waterman, and Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. [DOI](https://doi.org/10.1145/1498765.1498785).
2. NVIDIA, *CUDA C++ Best Practices Guide*. [Documentation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
3. NVIDIA, *Nsight Compute Profiling Guide*. [Documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
4. NVIDIA, *Nsight Compute Documentation*. [Documentation](https://docs.nvidia.com/nsight-compute/).
5. PyTorch, *Profiler API*. [Documentation](https://docs.pytorch.org/docs/stable/profiler.html).
6. PyTorch, *CUDA Event API*. [Documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html).
7. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. [Paper](https://arxiv.org/abs/2205.14135).
8. Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. [Paper](https://arxiv.org/abs/2307.08691).
9. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*. [Paper](https://arxiv.org/abs/2309.06180).
10. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. [Paper](https://arxiv.org/abs/2305.13245).

---

**Scope note:** All calculated intensities in this README are explicitly **analytical estimates** under stated data-movement and numerical precision assumptions. Measured DRAM intensity requires actual device-memory traffic counters. Roofline throughput numbers use a **hypothetical GPU** for explanation, not a claim about a specific NVIDIA product.
