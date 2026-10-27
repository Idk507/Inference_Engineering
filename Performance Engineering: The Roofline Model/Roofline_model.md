# The Roofline Model: Peak Compute, Peak Bandwidth, and the Ridge Point

> A beginner-to-advanced, engineering-focused guide to reasoning about GPU performance, identifying bottlenecks, and optimizing AI inference. All numerical accelerator examples are illustrative unless explicitly identified otherwise.

## 1. Why the Roofline model matters

A fast GPU is not necessarily a fast GPU *for your workload*. Consider an accelerator with enormous arithmetic throughput. If its cores must continuously wait for weights or activations to arrive from device memory, much of the compute capacity remains unused. Conversely, if the data is reused many times, performance may be dominated by how quickly the cores execute calculations rather than by memory bandwidth.

The **Roofline model** connects these facts through one inequality:

```text
Attainable throughput <= min(peak compute throughput,
                             arithmetic intensity × peak memory bandwidth)
```

Equivalently, with standard GitHub-rendered math:

$$
P_{\mathrm{attainable}}\leq \min\left(P_{\mathrm{peak}},\ I\,B_{\mathrm{peak}}\right)
$$

Here, `P_peak` is the hardware compute ceiling (FLOP/s), `B_peak` is the bandwidth ceiling (bytes/s), and `I` is arithmetic intensity (FLOP/byte) **at the corresponding memory interface**. Their product has units of FLOP/s. This yields an upper bound, not a prediction of exact elapsed time.

## 2. Essential vocabulary: FLOP, FLOP/s, byte, and bandwidth

A **FLOP** is one floating-point arithmetic operation. For conventional operation counting, a floating-point addition counts as one FLOP and a floating-point multiplication counts as one FLOP. A fused multiply-add (FMA), which calculates `a*b+c`, conventionally counts as **two FLOPs**, despite potentially executing as one hardware instruction. FLOP counts are workload sizes, whereas **FLOP/s** measures computational throughput.

**Memory bandwidth** is the number of bytes transferred per second, often measured in GB/s or TB/s. Crucially, bandwidth is not the same as memory capacity: a 24 GB GPU has 24 GB of storage, but that does not specify how many GB/s it can transfer. Bandwidth is also different from latency. Latency describes the time required for an individual request, while bandwidth describes aggregate transfer rate under sufficient concurrency.

**Arithmetic intensity** measures work performed per byte transferred:

$$
I=\frac{F}{Q}
$$

where `F` is the counted number of FLOPs and `Q` is the number of bytes moved across a chosen memory interface. Data moved between DRAM and the GPU can differ from data moved between L2 and L1 caches. Therefore, a workload can have distinct DRAM-level, L2-level, and L1-level arithmetic intensities.

Consider a kernel performing 20 GFLOPs and transferring 10 GB across DRAM. Its DRAM arithmetic intensity is `20/10 = 2 FLOPs/byte`. If another implementation performs the same operations with only 2 GB of DRAM traffic, its arithmetic intensity becomes `10 FLOPs/byte`. Reusing data is often the most effective way to raise intensity.

## 3. Peak compute throughput

**Peak compute throughput** is the theoretical upper limit on arithmetic performed per second on a specific hardware execution pathway. It depends on clock speed, arithmetic units, operations per instruction, precision, and utilization of specialized compute units. For example, FP32 CUDA-core peak throughput and BF16 Tensor Core peak throughput can differ enormously on the same GPU. Dense, sparse, FP8, INT8, FP16/BF16, FP32, and FP64 performance figures are not interchangeable.

An illustrative calculation for a simplified architecture is:

```text
Peak FLOP/s = number_of_execution_units
            × operations_per_unit_per_cycle
            × clock_cycles_per_second
```

A simplified GPU with 5,000 arithmetic lanes, each capable of two FLOPs per cycle and clocked at 1.5 GHz, has an ideal peak of 15 TFLOP/s under those assumptions. Actual products can have more complex execution schedules, instruction mixes, tensor cores, and thermal/power limits. Always match the peak throughput figure to the operations actually used by the kernel.

For example, a tensor-core GEMM using BF16 inputs and FP32 accumulation should be analyzed against a relevant BF16 Tensor Core ceiling, not an unrelated FP64 compute ceiling.

## 4. Peak memory bandwidth

**Peak memory bandwidth** is the upper bound on the rate of data transfer across a specified memory interface. For GPU DRAM, manufacturers commonly derive a theoretical bandwidth from the effective memory data rate and the bus width:

```text
Bandwidth (bytes/s) = effective_transfers_per_second
                    × memory_bus_width_bits / 8
```

A hypothetical memory subsystem with 16 billion effective transfers per second and a 256-bit bus provides `16 × 10^9 × 256/8 = 512 × 10^9 bytes/s`, or **512 GB/s** of theoretical bandwidth. This does not guarantee that a real kernel can sustain 512 GB/s. Access coalescing, transaction efficiency, cache behavior, scheduling, power limits, and contention all affect achievable bandwidth.

When estimating effective application bandwidth from a known logical traffic volume, use:

$$
B_{\mathrm{effective}}=\frac{Q}{T}
$$

For a kernel that logically reads 2 GB and writes 1 GB over 6 ms, logical effective bandwidth is `3 GB / 0.006 s = 500 GB/s`. A profiler measuring actual DRAM traffic may report another value if caches, additional transactions, or repeated accesses alter physical traffic.

## 5. Deriving the Roofline equation from execution time

Let the kernel perform `F` floating-point operations and transfer `Q` bytes. If the accelerator cannot exceed `P_peak` FLOP/s, the compute portion of the execution cannot take less than:

$$
T_{\mathrm{compute}}\geq\frac{F}{P_{\mathrm{peak}}}
$$

Likewise, if the memory interface cannot exceed `B_peak` bytes/s, the memory-transfer component cannot take less than:

$$
T_{\mathrm{memory}}\geq\frac{Q}{B_{\mathrm{peak}}}
$$

Regardless of the extent to which computation and memory access overlap, total time must be at least the larger of these two ideal lower bounds:

$$
T\geq\max\left(\frac{F}{P_{\mathrm{peak}}},\frac{Q}{B_{\mathrm{peak}}}\right)
$$

Achieved throughput is `F/T`, so:

$$
\frac{F}{T}\leq\min\left(P_{\mathrm{peak}},\frac{F}{Q}B_{\mathrm{peak}}\right)
$$

Substituting `I = F/Q` produces the Roofline equation. Notice the units: `(FLOP/byte) × (byte/s) = FLOP/s`. This dimensional consistency is a quick check for a correct calculation.

**Important:** The `max(compute time, memory time)` relation is an optimistic *lower bound*, not a claim that real kernels achieve perfect overlap. Serial phases, launch overhead, synchronization, instruction dependencies, and other bottlenecks can make elapsed time substantially longer.

## 6. The Roofline graph

The horizontal axis is arithmetic intensity (FLOP/byte); the vertical axis is attainable throughput (FLOP/s). Roofline diagrams conventionally use logarithmic scales for both axes, where the bandwidth roof has slope 1 and the compute roof is horizontal.

```text
Performance (FLOP/s)
       ^
       |                         +------------------ Peak compute roof
       |                      +--+
       |                   +--+
       |                +--+      Compute-roof-limited region
       |             +--+
       |          +--+
       |       +--+  Memory-bandwidth-roof-limited region
       |    +--+
       +-----------------------------------------------> Arithmetic intensity
                         ^
                      Ridge point
```

The rising section, `P = I × B_peak`, is the **memory-bandwidth roof**. The horizontal section, `P = P_peak`, is the **compute roof**. A measured kernel is normally plotted **below** these theoretical boundaries. The vertical gap between the measured performance and the corresponding roof can reveal unrealized performance potential. Merely being left or right of the ridge identifies which *ideal ceiling* is lower, not definitive proof of the kernel's actual bottleneck.

## 7. Deriving the ridge point

The **ridge point** is where the two theoretical ceilings intersect:

$$
P_{\mathrm{peak}}=I_{\mathrm{ridge}}B_{\mathrm{peak}}
$$

Therefore:

$$
\boxed{I_{\mathrm{ridge}}=\frac{P_{\mathrm{peak}}}{B_{\mathrm{peak}}}}
$$

Its unit is FLOP/byte. It represents the minimum arithmetic intensity necessary for the *compute ceiling*, rather than the *DRAM bandwidth ceiling*, to be the lower of the two ideal ceilings. It does **not** guarantee peak compute utilization.

Suppose a hypothetical GPU offers **100 TFLOP/s** for the target precision and **1 TB/s** of DRAM bandwidth:

```text
Ridge intensity = 100 TFLOP/s / 1 TB/s
                = 100 FLOPs/byte
```

At 5 FLOPs/byte, the bandwidth roof permits up to `5 TFLOP/s`. At 50 FLOPs/byte, it permits up to `50 TFLOP/s`. At 100 FLOPs/byte, the two roofs intersect at `100 TFLOP/s`. At 300 FLOPs/byte, the compute roof still limits the theoretical result to `100 TFLOP/s`.

| Arithmetic intensity | Bandwidth ceiling | Compute ceiling | Roofline bound | Lower ceiling |
|---:|---:|---:|---:|---|
| 1 FLOP/byte | 1 TFLOP/s | 100 TFLOP/s | 1 TFLOP/s | Bandwidth |
| 10 FLOPs/byte | 10 TFLOP/s | 100 TFLOP/s | 10 TFLOP/s | Bandwidth |
| 50 FLOPs/byte | 50 TFLOP/s | 100 TFLOP/s | 50 TFLOP/s | Bandwidth |
| 100 FLOPs/byte | 100 TFLOP/s | 100 TFLOP/s | 100 TFLOP/s | Ridge |
| 200 FLOPs/byte | 200 TFLOP/s | 100 TFLOP/s | 100 TFLOP/s | Compute |

## 8. Why the ridge point changes between accelerators

Consider three hypothetical devices, measured for a **single, consistent numerical datatype and operation class**.

| Accelerator | Peak compute | Peak bandwidth | Ridge point |
|---|---:|---:|---:|
| A | 20 TFLOP/s | 1 TB/s | 20 FLOPs/byte |
| B | 80 TFLOP/s | 1 TB/s | 80 FLOPs/byte |
| C | 80 TFLOP/s | 2 TB/s | 40 FLOPs/byte |

Device B has 4× the compute capacity of A, but unchanged bandwidth. Its ridge point moves to the right, so a workload needs higher arithmetic intensity to make full use of the additional compute. Device C doubles bandwidth compared with B, reducing its ridge point from 80 to 40 FLOPs/byte. This is why comparing accelerators on headline TFLOP/s figures alone is misleading for memory-heavy inference.

The ridge also changes **within one accelerator** depending on the active compute pathway: FP32 scalar instructions may have a completely different peak from FP16 tensor-core operations. When quantization changes the arithmetic type, update both the compute ceiling and the memory-traffic model.

## 9. Worked example: vector addition

For an FP32 elementwise addition `C[i] = A[i] + B[i]`, each output requires reading two FP32 values and writing one FP32 value. That is `4 + 4 + 4 = 12 bytes` of minimum logical traffic per element. It performs one floating-point addition, giving:

$$
I\approx\frac{1}{12}=0.0833\ \mathrm{FLOPs/byte}
$$

On the 100 TFLOP/s, 1 TB/s hypothetical accelerator, the bandwidth ceiling is:

```text
0.0833 FLOPs/byte × 1 TB/s ≈ 0.0833 TFLOP/s
```

This is far below the compute ceiling. Large streaming vector addition is a classic low-intensity workload. A faster arithmetic unit is unlikely to help if data movement is already saturated. Fusion, reducing redundant reads/writes, and improving coalescing are more relevant.

## 10. Worked example: dense matrix multiplication

Let `C = A @ B`, where `A`, `B`, and `C` are `N × N` matrices. Dense multiplication performs approximately `2N³` FLOPs. For FP32 inputs and outputs, an optimistic compulsory-traffic estimate reads A once, reads B once, and writes C once:

$$
Q_{\min}\approx 3N^2\times4=12N^2\ \mathrm{bytes}
$$

Therefore:

$$
I_{\mathrm{ideal}}\approx\frac{2N^3}{12N^2}=\frac{N}{6}\ \mathrm{FLOPs/byte}
$$

For `N = 1024`, this yields approximately **170.7 FLOPs/byte**. On the hypothetical accelerator with a ridge of 100 FLOPs/byte, this shape has sufficient *potential* reuse to reach the compute-roof-limited region. But actual performance depends on whether the implementation realizes that reuse.

A poorly implemented matrix multiplication can reload input elements many times, drastically increasing memory traffic. Blocked/tiled algorithms keep operand tiles in registers or shared memory and reuse them for many multiply-add operations. Large tile sizes can improve locality, but excessively large tiles may reduce occupancy or cause register spills. Matrix multiplication is thus a case where the theoretical `N/6` intensity represents an idealized traffic bound, not a promise for a particular kernel.

## 11. Kernel fusion and how an optimization moves a point on the Roofline

Consider `Y = relu(A + B)`. If addition and activation are separate kernels, the first reads A and B and writes a temporary T; the second reads T and writes Y. Under a simple FP32 streaming model, this moves `12 + 8 = 20 bytes` per element. A fused kernel reads A and B and writes Y: `12 bytes` per element.

The same high-level operation now moves **40% fewer nominal bytes**. If the floating-point operation count is unchanged, the DRAM arithmetic intensity increases by `20/12 ≈ 1.67×`. If the unfused kernel was DRAM-bandwidth-limited, this can raise attainable throughput. The actual effect may be smaller when intermediates already remain in cache or greater when fusion additionally avoids kernel-launch overhead.

On a Roofline graph, fusion may move the kernel point **rightward** (higher FLOPs/byte) and **upward** (higher achieved FLOP/s). If it was already compute-roof-limited, changing intensity might not translate into comparable runtime improvement.

## 12. Why LLM prefill and decode lie in different Roofline regions

Transformer inference illustrates why arithmetic intensity, not parameter count alone, determines performance behavior.

### Prefill: many tokens reuse the same weights

For a dense projection `Y = XW`, let `X` have shape `S × H`, and let `W` have shape `H × H`. With BF16/FP16 operands and output, a simplified single-load traffic estimate gives:

$$
F\approx2SH^2
$$

$$
Q\approx2H^2+4SH\ \mathrm{bytes}
$$

Hence:

$$
I_{\mathrm{prefill}}\approx\frac{2SH^2}{2H^2+4SH}
$$

For `S = 1024` and `H = 4096`, this is approximately **682.7 FLOPs/byte**. Reusing the same projection weights over many prompt tokens produces high theoretical intensity. Long-prompt prefill often gives Tensor Cores enough work to become compute-intensive, although real shapes, attention, and memory behavior still matter.

### Decode: one token per sequence

For a single-token projection, set `S = 1`. The same expression yields approximately **1 FLOP/byte** for large `H` and BF16 weight storage. A large weight matrix is read to perform relatively little work, so a single-sequence decode projection is often dominated by memory traffic.

### Continuous batching: reuse weights across users

If `S` now denotes the number of simultaneously processed decode tokens from different active sequences, increasing that number raises weight reuse and the intensity of the projection GEMM. With `H = 4096` and `S = 32`, the ideal projection intensity is approximately **31.5 FLOPs/byte**. This helps explain why continuous batching improves serving throughput.

However, decoding also performs KV-cache attention reads. Longer contexts, many concurrent sequences, grouped-query attention, KV precision, and cache layout can change the model-wide bottleneck. A high projection intensity does not imply that the complete transformer decode step is compute-bound.

## 13. Quantization and the Roofline

Reducing weight precision decreases the number of bytes required to stream weights. For a very large matrix-vector projection, a simplified *equivalent mathematical operations per stored weight byte* metric gives approximately `0.5` for FP32 weights, `1` for FP16/BF16, `2` for INT8, and `4` for tightly packed INT4.

This is useful for understanding why weight quantization can improve bandwidth-limited decoding. However, INT8 and INT4 operations are not literally floating-point operations, and real kernels have scale/zero-point metadata, dequantization costs, activation traffic, and different compute ceilings. Never compare an INT4 model's derived operation-per-byte estimate to an unrelated FP32 FLOP/s peak and call the result a measured Roofline.

Likewise, quantizing the KV cache can reduce attention bandwidth pressure, but any realized speedup depends on decode attention implementation, data reuse, and dequantization overhead. End-to-end latency must be measured.

## 14. Hierarchical Roofline: DRAM, L2, and L1

A kernel can appear far below its DRAM roof and still be limited by a different memory level. Hierarchical Roofline analysis creates distinct intensity/throughput relationships for DRAM, L2, L1, and sometimes additional execution constraints.

For example, suppose a kernel performs 2 billion FLOPs, moves 100 MB across the DRAM interface, and moves 500 MB through an L2 interface. Its DRAM intensity is `20 FLOPs/byte`; its L2 intensity is `4 FLOPs/byte`. These characterize different traffic bottlenecks. A strong cache hit rate may make DRAM traffic small while the L2 data path remains heavily utilized.

The optimization decision should be made using the memory level that is actually restrictive. Improving DRAM locality might do little for a kernel already limited by L1/shared-memory behavior, register dependencies, or tensor-core instruction throughput.

## 15. Why theoretical Roofline limits are not automatically achieved

A Roofline curve assumes idealized peak performance and sufficient useful work. Real kernels may sit well below both main roofs because of launch overhead, poor occupancy, divergent branches, instruction dependency chains, inadequate parallelism, low memory-level parallelism, inefficient memory transactions, register spilling, synchronization, or clock/power limitations.

For example, a short kernel may not run long enough to reach sustained peak bandwidth, even though its algorithmic intensity predicts a bandwidth-limited ceiling. Similarly, a large GEMM may have high theoretical intensity but fail to approach the compute ceiling if its tensor dimensions are unfriendly to the chosen tensor-core tile or if the kernel launches too few thread blocks.

A careful analyst asks two questions separately: **Which theoretical roof is lower?** and **How close is the measured kernel to that roof?** The first suggests a fundamental resource balance; the second tells you whether implementation efficiency and other bottlenecks remain important.

## 16. Production profiling with Nsight Compute

On NVIDIA GPUs, Nsight Compute supports roofline charts and hierarchical memory ceilings. For a CUDA application, a starting command is:

```bash
ncu --section SpeedOfLight_RooflineChart -o roofline_report ./my_cuda_application
```

To profile a Python program with CUDA kernels:

```bash
ncu --section SpeedOfLight_RooflineChart -o roofline_report python benchmark.py
```

Nsight Compute may require elevated profiling-counter permissions on some systems; use approved organizational procedures instead of bypassing access controls. Save the report and open it in the Nsight Compute UI. GPU hardware, driver compatibility, and the version of Nsight Compute determine which sections are supported. Other precision-specific or hierarchical roofline sections can be selected when available.

Record the actual GPU SKU, driver, GPU clock behavior, input shapes, dtype, peak compute pathway, measured FLOP count, measured DRAM bytes, runtime, achieved FLOP/s, and relevant cache-level traffic. Profiling a production inference server additionally requires reporting request concurrency, batch sizes, prefill/decode ratio, input/output lengths, KV-cache utilization, latency percentiles, and tokens/second.

For observability, combine CUDA kernel-level analysis with server-level telemetry. Useful metrics include GPU utilization, memory bandwidth utilization, VRAM occupancy, power draw, model/token throughput, time to first token, time per output token, p50/p95/p99 latency, queue depth, batching efficiency, and OOM/admission-control events. Beware that application-level memory allocation is **not** the same as cumulative DRAM bytes moved.

## 17. Runnable Python Roofline simulator and benchmark

The script below first plots the *theoretical* Roofline of a configurable imaginary accelerator and then, optionally, benchmarks vector addition and matrix multiplication on a CUDA GPU. The CUDA measurements use known FLOP counts and **idealized compulsory traffic**, not GPU performance-counter measurements of DRAM traffic. Thus the observed points are illustrative estimates and should not be mistaken for hardware-counter Roofline results.

Save the script as `roofline_demo.py`:

```python
"""Draw a theoretical GPU Roofline and optionally benchmark CUDA kernels.

Requirements: numpy, matplotlib; PyTorch only for --cuda-benchmark.
The plotted benchmark points use assumed logical traffic, not measured DRAM bytes.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger("roofline_demo")


@dataclass(frozen=True)
class Result:
    """Store a kernel's calculated work, assumed traffic, and observed duration."""

    name: str
    flops: float
    bytes_moved: float
    seconds: float

    @property
    def intensity(self) -> float:
        """Return FLOPs per assumed logical byte."""
        return self.flops / self.bytes_moved

    @property
    def tflops(self) -> float:
        """Return achieved TFLOP/s from the elapsed wall-clock GPU time."""
        return self.flops / self.seconds / 1e12


def calculate_roofline(
    peak_tflops: float, peak_bandwidth_tbps: float, intensity: np.ndarray
) -> np.ndarray:
    """Calculate a theoretical Roofline throughput vector in TFLOP/s."""
    if peak_tflops <= 0 or peak_bandwidth_tbps <= 0:
        raise ValueError("Peak compute and bandwidth must be positive.")
    if not np.all(np.isfinite(intensity)) or np.any(intensity <= 0):
        raise ValueError("All arithmetic intensities must be positive and finite.")
    return np.minimum(peak_tflops, intensity * peak_bandwidth_tbps)


def benchmark_cuda(repeats: int = 30) -> list[Result]:
    """Measure CUDA timings for elementwise add and a dense FP32 GEMM.

    The GEMM uses a single-load lower bound for input/output traffic.
    CUDA event timings are synchronized; launch overhead is partly excluded.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for CUDA benchmarking.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("No available CUDA device was detected.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    device = torch.device("cuda")
    vector_size = 2_000_000
    matrix_size = 1024

    a = torch.randn(vector_size, device=device, dtype=torch.float32)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    x = torch.randn(matrix_size, matrix_size, device=device)
    y = torch.randn_like(x)
    z = torch.empty_like(x)

    def time_kernel(operation) -> float:
        """Return median CUDA event duration in seconds after warm-up."""
        try:
            for _ in range(10):
                operation()
            torch.cuda.synchronize()
            samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                operation()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 1000.0)
            return float(np.median(samples))
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("Insufficient GPU memory for benchmark.") from exc

    # Preallocate outputs: we benchmark the arithmetic kernels, not allocation.
    add_time = time_kernel(lambda: torch.add(a, b, out=c))
    gemm_time = time_kernel(lambda: torch.mm(x, y, out=z))
    return [
        Result(
            "FP32 vector add",
            float(vector_size),
            float(vector_size * 3 * 4),
            add_time,
        ),
        Result(
            "FP32 GEMM (1024)",
            float(2 * matrix_size**3),
            float(3 * matrix_size**2 * 4),
            gemm_time,
        ),
    ]


def main() -> None:
    """Parse arguments, calculate ceilings, and save a Roofline figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-tflops", type=float, default=100.0)
    parser.add_argument("--bandwidth-tbps", type=float, default=1.0)
    parser.add_argument("--cuda-benchmark", action="store_true")
    parser.add_argument("--output", default="roofline.png")
    args = parser.parse_args()

    if not np.isfinite(args.peak_tflops) or not np.isfinite(args.bandwidth_tbps):
        parser.error("Peak performance and bandwidth must be finite.")
    if args.peak_tflops <= 0 or args.bandwidth_tbps <= 0:
        parser.error("Peak performance and bandwidth must be positive.")

    ridge = args.peak_tflops / args.bandwidth_tbps
    intensities = np.logspace(-2, 4, 600)
    bound = calculate_roofline(
        args.peak_tflops, args.bandwidth_tbps, intensities
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(intensities, bound, label="Theoretical Roofline", linewidth=2)
    ax.axvline(ridge, linestyle="--", label=f"Ridge = {ridge:g} FLOPs/byte")
    ax.axhline(args.peak_tflops, linestyle=":", label="Peak compute")

    if args.cuda_benchmark:
        for result in benchmark_cuda():
            LOGGER.info(
                "%s: AI=%.3f FLOPs/byte, achieved=%.4f TFLOP/s",
                result.name, result.intensity, result.tflops
            )
            ax.scatter(result.intensity, result.tflops, s=65, label=result.name)

    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Throughput (TFLOP/s)")
    ax.set_title("Theoretical Roofline and optional estimated CUDA points")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    LOGGER.info("Saved %s; ridge intensity = %.2f FLOPs/byte", args.output, ridge)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        LOGGER.error("Roofline experiment failed: %s", error)
        raise SystemExit(1) from error
```

Install plotting dependencies:

```bash
python -m pip install numpy matplotlib
```

Create a theoretical Roofline graph without requiring a GPU:

```bash
python roofline_demo.py --peak-tflops 100 --bandwidth-tbps 1
```

To add local CUDA benchmark estimates, install a CUDA-enabled PyTorch build appropriate for your GPU and operating system, then run:

```bash
python roofline_demo.py --peak-tflops 100 --bandwidth-tbps 1 --cuda-benchmark
```

**Important caveat:** The default 100 TFLOP/s and 1 TB/s peaks describe an illustrative accelerator, not necessarily the GPU running the optional benchmarks. To interpret the position of the CUDA points relative to the roof meaningfully, replace these with consistent, correctly measured or specified ceilings for your device and FP32 execution path. Even then, the benchmark's `bytes_moved` values are analytical assumptions, not true DRAM traffic. For measured DRAM arithmetic intensity, use hardware counters in Nsight Compute.

## 18. Optimization decisions: what to change in each regime

| Observation | Likely direction | Useful techniques | Measure after change |
|---|---|---|---|
| Low intensity, near bandwidth roof | Reduce bytes transferred | Tiling, fusion, caching, compression | DRAM bytes, effective bandwidth, runtime |
| Low intensity, far below bandwidth roof | Improve memory efficiency or latency hiding | Coalescing, occupancy, parallelism, fewer launches | Cache misses, transactions, stalled cycles |
| High intensity, near compute roof | Improve math efficiency or reduce work | Better algorithms, tensor-core usage, precision choice | FLOP/s, tensor-core utilization, accuracy |
| High intensity, far below compute roof | Investigate utilization | Kernel shapes, scheduling, occupancy, instruction mix | Achieved occupancy, issue utilization, runtime |
| LLM decode, small batch | Improve weight reuse or reduce bytes | Continuous batching, quantization, efficient attention | Tokens/s, TPOT, tail latency, KV traffic |
| LLM prefill, large batch | Improve high-throughput execution | Fused attention, efficient GEMMs, tensor-core tiles | TTFT, FLOP/s, memory usage |

Do not select a technique purely because the model labels a workload “memory-bound.” Profile actual runtime, ensure correctness and numerical accuracy, and compare end-to-end service quality. A kernel speedup can disappear when network overhead, request queueing, CPU tokenization, or host-device transfers dominate total response time.

## 19. Production engineering considerations

Security, isolation, scalability, and observability matter when applying performance changes in deployed inference systems. Profiling can expose model shapes and runtime activity, so store performance reports with restricted access and avoid including user prompts or sensitive tensor contents. Apply least-privilege access to GPU profiling facilities. Quantization artifacts should be versioned and integrity-checked, with quality regression tests before promotion.

Scaling decisions should consider batching policies, throughput saturation, GPU-memory admission control, tenant fairness, prefill/decode separation, and autoscaling triggers tied to queue depth and tail latency rather than GPU utilization alone. Monitor the full system: sustained tokens per second, time to first token, output-token latency, OOM events, preemption frequency, DRAM utilization, KV-cache occupancy, and energy consumption. Profile representative traffic distributions instead of a single synthetic matrix size.

When comparing models or implementations, keep dimensions, hardware, numerical precision, clock and power settings, warm-up behavior, and measurement intervals controlled. The Roofline model is most useful when integrated into this broader measurement process.

## 20. Frequent misconceptions

**“High arithmetic intensity always means compute-bound.”** Not necessarily. It only means that the ideal compute roof may be lower than the chosen memory roof. Real performance can be limited by latency, occupancy, synchronization, a different memory level, or instruction inefficiencies.

**“Peak TFLOP/s is what an application will achieve.”** No. Peak arithmetic throughput is a theoretical upper limit for a particular precision and execution pathway; application throughput is frequently lower.

**“The ridge point is a fixed value for a GPU.”** Not universally. It changes with the precision-specific compute ceiling and the memory interface being considered, as well as the selected sustained/peak assumptions.

**“Memory allocated equals memory bytes moved.”** No. A 50 MB tensor accessed 100 times can generate far more than 50 MB of traffic, while cached accesses may generate less traffic at DRAM than their logical access count suggests.

**“Quantization produces a speedup proportional to compression.”** Not guaranteed. Dequantization work, metadata, kernel quality, bottleneck migration, and changed hardware compute pathways all matter.

**“Batching always improves latency.”** Batching can increase throughput by improving weight reuse, but queuing, memory pressure, and scheduling can worsen per-request latency, especially for long-tail requests.

## 21. Interview-level exercises

**Exercise A.** A GPU can execute 60 TFLOP/s and transfer 1.5 TB/s from DRAM. Derive its ridge point. **Answer:** `60/1.5 = 40 FLOPs/byte`.

**Exercise B.** A kernel performs 12 GFLOPs and transfers 3 GB. What is its intensity and theoretical bandwidth-limited throughput on that GPU? **Answer:** `4 FLOPs/byte`; bandwidth ceiling `4 × 1.5 = 6 TFLOP/s`, below the 60 TFLOP/s compute ceiling.

**Exercise C.** A kernel performs 300 GFLOPs, transfers 1 GB, and runs on the GPU from Exercise A. What is its theoretical upper bound? **Answer:** Intensity `300 FLOPs/byte`, bandwidth ceiling `450 TFLOP/s`, compute ceiling `60 TFLOP/s`; Roofline upper bound `60 TFLOP/s`.

**Exercise D.** Why might FP16 GEMM and FP32 GEMM have different ridge points on the same device? **Answer:** They can use different execution units and throughput ceilings; FP16 also changes operand traffic, and actual tensor-core eligibility depends on dimensions and kernels.

**Exercise E.** You double model decode batch size but throughput barely changes. What would you investigate? **Answer:** Compare weight and KV-cache bytes, kernel execution mode, GPU occupancy, available memory, batch padding, scheduler overhead, and achieved memory bandwidth. The simple projection intensity model may no longer be the service-level limiting factor.

## 22. Final mental model

The Roofline model answers three questions in sequence: **How much arithmetic can my hardware perform? How quickly can it deliver the bytes? How much useful work does each delivered byte support?** Peak compute provides the horizontal ceiling. Peak bandwidth multiplied by arithmetic intensity provides the sloped ceiling. Their intersection, the ridge point, identifies the intensity at which the ideal limiting resource changes.

For AI inference, this leads to a practical rule: low-batch decode often benefits from less memory traffic and more weight reuse; large prefill GEMMs often benefit from efficient matrix computation. Neither claim should be accepted without measurements on the real workload and chosen precision.

## References

The following documentation provides hardware-grounded definitions, profiler methods, and extensions beyond the simple two-roof model.

1. [NVIDIA Nsight Compute — Roofline Charts and Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)
2. [NVIDIA Nsight Compute — Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/)
3. [NVIDIA CUDA C++ Best Practices Guide — Bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#bandwidth)
4. [NERSC — Roofline Performance Model](https://docs.nersc.gov/tools/performance/roofline/)
5. [AMD ROCm — Understanding GPU Performance](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html)
6. [NVIDIA TensorRT-LLM — Roofline Analysis Reference](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-nsight-compute-analysis/references/roofline-analysis.md)

---

*The purpose of the Roofline model is not to replace profiling. It is to connect measured bottlenecks to the physical limits that explain them.*
