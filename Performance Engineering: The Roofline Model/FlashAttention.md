# FlashAttention: Tiling, Online Softmax, IO Awareness, and the Roofline Model

FlashAttention is an IO-aware implementation of exact dense attention. Its central idea is to avoid materializing the full N x N attention score and probability matrices in GPU HBM. Instead, it tiles Q, K, and V into fast on-chip memory, computes score blocks, performs numerically stable softmax online, accumulates the weighted V output, and writes only the final output back to HBM.

```text
Standard attention:
Q @ K^T -> N x N scores -> HBM -> softmax -> N x N probabilities -> HBM -> @ V

FlashAttention:
Q/K/V tiles -> on-chip scores -> online softmax -> on-chip output accumulation -> final O
```

The dense mathematical operation is still approximately O(N^2 * d) compute. The primary optimization is drastically lower expensive memory traffic, which raises effective arithmetic intensity and can move the kernel toward or across the Roofline ridge point.

## 1. Standard scaled dot-product attention

For one head:

```text
Q shape = [N, d]
K shape = [N, d]
V shape = [N, d]

S = Q @ K^T / sqrt(d)
P = softmax(S)
O = P @ V
```

The dominant FLOP count is approximately:

```text
QK^T: 2 * N^2 * d FLOPs
PV:   2 * N^2 * d FLOPs

Total dominant FLOPs ≈ 4 * N^2 * d
```

The problematic intermediates are `S` and `P`, each logically `[N, N]`. At N=4096 there are 16,777,216 elements in one such matrix. At two bytes per element, that is roughly 33.6 MB for one conceptual matrix. Real kernels differ in fusion and accumulation precision, but the quadratic intermediate is the fundamental IO problem.

## 2. Why FLOPs alone do not explain runtime

Two implementations can execute roughly the same mathematical FLOPs but have very different runtimes if one repeatedly transfers large intermediates through HBM.

```text
Arithmetic intensity = FLOPs / bytes moved
```

If useful FLOPs remain similar while expensive HBM traffic falls substantially, arithmetic intensity rises. FlashAttention is a canonical example of optimizing data movement rather than merely reducing arithmetic.

## 3. GPU memory hierarchy

A simplified hierarchy is:

```text
Registers
   |
Shared memory / on-chip SRAM
   |
L1 / L2
   |
HBM
```

On-chip storage is fast but limited. HBM is much larger but moving data to and from it is expensive relative to register/shared-memory reuse. FlashAttention deliberately chooses tiles whose working state can reside in fast on-chip storage.

## 4. Why ordinary softmax seems to require the whole row

For a score row `x`:

```text
m = max(x)

softmax(x_i) =
exp(x_i - m) / sum_j exp(x_j - m)
```

This seems to require all scores before normalization because the maximum and denominator depend on the entire row. Online softmax removes that requirement by maintaining sufficient statistics incrementally.

## 5. Online softmax derivation

Maintain:

```text
m = running maximum
l = running exponential sum
```

For a new score block:

```text
m_block = max(block)
m_new   = max(m, m_block)

l_new =
    exp(m - m_new) * l
    +
    sum(exp(block - m_new))
```

Why does rescaling work?

```text
exp(x_i - m_new)
=
exp(x_i - m) * exp(m - m_new)
```

Therefore all previously accumulated exponential terms can be converted from the old reference maximum `m` to the new maximum `m_new` using one scale factor. This preserves numerically stable global normalization without retaining every previous score.

## 6. Worked online-softmax example

Process:

```text
[1, 2, 3, 6]
```

as:

```text
block 1 = [1, 2]
block 2 = [3, 6]
```

First block:

```text
m = 2
l = exp(-1) + 1
  ≈ 1.3679
```

Second block introduces a new maximum:

```text
m_new = 6

old contribution =
exp(2 - 6) * 1.3679
≈ 0.0251

new contribution =
exp(3 - 6) + exp(6 - 6)
≈ 1.0498

l_new ≈ 1.0749
```

Direct stable normalization relative to maximum 6 gives the same denominator, modulo floating-point ordering.

## 7. Online accumulation of P @ V

Softmax alone is insufficient because attention ultimately needs `P @ V`. Maintain an unnormalized output accumulator `a` in addition to `m` and `l`.

For a new score/value block:

```text
m_new = max(m_old, max(scores))
alpha = exp(m_old - m_new)

p_block = exp(scores - m_new)

l_new =
alpha * l_old
+
sum(p_block)

a_new =
alpha * a_old
+
p_block @ V_block
```

After every K/V block has been processed:

```text
output = a_final / l_final
```

Thus both normalization and weighted-value accumulation can happen block by block.

## 8. Tiling

For each Q tile, FlashAttention iterates over K/V tiles:

```text
load Q_tile

initialize:
running max
running denominator
output accumulator

for each K_tile, V_tile:
    load K_tile and V_tile
    scores = Q_tile @ K_tile^T
    scale / mask scores
    update online softmax
    accumulate probabilities @ V_tile
    discard temporary score/probability tile

normalize accumulator
write final output tile
```

The score tile exists temporarily on-chip. The full N x N score/probability tensor never needs to become a persistent HBM intermediate.

## 9. Why kernel fusion matters

Tiling alone is not enough if every stage writes its tile back to HBM. The benefit comes from fusing:

```text
QK^T
 -> scale/mask
 -> online softmax
 -> P @ V
 -> output accumulation
```

while temporary state remains on-chip. A design that writes score tiles to HBM between separate kernels gives away much of the IO advantage.

## 10. Formal arithmetic-intensity shift

Let baseline attention perform `F` FLOPs and move `B_standard` bytes through the relevant expensive memory level:

```text
AI_standard = F / B_standard
```

Suppose the tiled implementation performs `F + delta_F` because of rescaling/recomputation but moves only `B_flash` bytes:

```text
AI_flash = (F + delta_F) / B_flash
```

When:

```text
B_flash << B_standard
```

it is possible that:

```text
AI_flash >> AI_standard
```

even if `delta_F` is positive. More FLOPs can therefore produce lower runtime when they replace much more expensive memory movement.

## 11. Roofline interpretation

The Roofline model is:

```text
Achievable performance =
min(
    peak compute,
    arithmetic intensity * peak memory bandwidth
)
```

The ridge point is:

```text
ridge point =
peak compute / peak memory bandwidth
```

Reducing HBM bytes moves the attention kernel rightward on the Roofline graph:

```text
Performance
^
|                         ---------------- peak compute
|                       /
|                     /
|                   /
|                 /
|       A ------> F
|      /
|_____/
+----------------------------------------------> Arithmetic intensity

A = baseline
F = FlashAttention-style implementation
```

If the new arithmetic intensity exceeds the ridge point, compute becomes the modeled limiting resource. If it remains below the ridge point, the kernel remains bandwidth-bound but can still be substantially faster.

## 12. Numerical Roofline example

Assume an illustrative accelerator:

```text
Peak compute       = 300 TFLOP/s
Peak HBM bandwidth = 2 TB/s

Ridge point = 300 / 2
            = 150 FLOPs/byte
```

Baseline:

```text
AI = 40 FLOPs/byte

Bandwidth roof = 40 * 2
               = 80 TFLOP/s

min(300, 80) = 80 TFLOP/s
```

The simplified model says bandwidth-bound.

After IO-aware optimization:

```text
AI = 200 FLOPs/byte

Bandwidth roof = 200 * 2
               = 400 TFLOP/s

min(300, 400) = 300 TFLOP/s
```

The modeled bottleneck has moved to compute. These are illustrative values; actual arithmetic intensity must be measured for the concrete workload and GPU.

## 13. The critical qualification

It is too absolute to say:

```text
standard attention = memory-bound
FlashAttention     = compute-bound
```

A more precise statement is:

```text
FlashAttention reduces expensive HBM traffic and raises effective arithmetic
intensity. This moves attention toward the compute roof and can cross the
ridge point for favorable workloads.
```

The final bottleneck depends on sequence length, head dimension, batch size, number of heads, precision, masking, GPU architecture, tile dimensions, occupancy, implementation, and whether the workload is forward, backward, prefill, or decode.

## 14. Why FlashAttention can do extra arithmetic and still win

Online softmax introduces running maxima, exponentials, rescaling, reductions, and accumulator updates. Training implementations can also recompute some local quantities in backward rather than storing them.

This is often profitable because:

```text
cost of extra arithmetic
<
cost of storing + rereading huge intermediates
```

The principle is:

```text
recompute instead of reload
```

when compute is relatively abundant and memory traffic is the scarce resource.

## 15. Memory footprint

FlashAttention also addresses memory capacity. Dense attention logically has an N x N score matrix.

For:

```text
N = 16384

N^2 = 268,435,456 elements
```

At two bytes per element, one conceptual N x N tensor is about 536.9 MB. Avoiding persistent materialization of these quadratic intermediates makes much longer contexts practical.

Do not confuse:

```text
memory capacity = how many bytes fit
memory bandwidth = how quickly bytes move
```

FlashAttention helps both, but for different reasons.

## 16. It does not remove quadratic dense-attention compute

For fixed head dimension:

```text
QK^T ~ O(N^2 * d)
PV   ~ O(N^2 * d)
```

FlashAttention still computes exact dense pairwise attention. Its major innovation is IO complexity and memory scheduling, not elimination of the fundamental quadratic dense-attention arithmetic.

## 17. Causal attention

A causal matrix is logically triangular:

```text
X . . . .
X X . . .
X X X . .
X X X X .
X X X X X
```

Tiling allows blocks to be classified as fully valid, partially valid, or fully masked. Fully masked blocks can be skipped and boundary blocks can apply masking locally.

## 18. Training and backward recomputation

Training needs gradients. Instead of retaining huge forward intermediates for backward, an IO-aware implementation can retain compact normalization state and recompute local quantities when needed.

```text
store approach:
compute -> store huge intermediate -> reload during backward

recompute approach:
compute -> retain compact state -> recompute local intermediate during backward
```

This exchanges additional FLOPs for lower activation storage and memory traffic.

## 19. Prefill versus decode

FlashAttention is especially natural during prefill:

```text
many Q tokens
many K/V tokens
large matrix-matrix work
large logical N x N attention
```

Avoiding N x N HBM materialization provides a large IO benefit.

Decode is different:

```text
Q       ≈ [1, d]
K_cache ≈ [N, d]
V_cache ≈ [N, d]
```

The query is tiny while a large KV cache may need to be streamed. There is less query-side reuse, so decode attention can remain strongly memory-bandwidth-sensitive. This is another reason not to claim that FlashAttention universally makes attention compute-bound.

## 20. GQA and MQA

Grouped-query and multi-query attention reduce distinct K/V heads and therefore KV-cache capacity and bandwidth requirements.

```text
FlashAttention:
reduces attention-intermediate IO

GQA / MQA:
reduces K/V representation and KV-cache traffic
```

They solve related but different memory problems and are particularly complementary in inference.

## 21. FlashAttention-2

FlashAttention-2 improved on the original algorithm through better work partitioning and parallelism. Its engineering directions include reducing non-matmul FLOPs, increasing parallelism, improving thread-block and warp work partitioning, and reducing unnecessary shared-memory communication.

This illustrates a general optimization pattern:

```text
remove HBM bottleneck
        |
        v
on-chip communication / occupancy / compute become more important
```

Solving one bottleneck exposes the next.

## 22. Tile-size trade-offs

Larger tiles can improve reuse and amortize overhead, but consume more registers/shared memory and may reduce occupancy. Smaller tiles use fewer resources but can reduce reuse and matrix-multiply efficiency.

A high-performance kernel balances:

```text
data reuse
register pressure
shared-memory usage
occupancy
warp efficiency
Tensor Core utilization
synchronization
```

This is why the algorithmic idea is simple compared with the engineering of a production kernel.

## 23. Educational pseudocode

```python
for each Q_block:

    load Q_block

    running_max = -infinity
    running_sum = 0
    output_accumulator = 0

    for each K_block, V_block:

        load K_block
        load V_block

        scores = Q_block @ K_block.T
        scores *= scale
        apply_mask_if_needed(scores)

        block_max = row_max(scores)
        new_max = max(running_max, block_max)

        old_scale = exp(running_max - new_max)
        probabilities = exp(scores - new_max)

        running_sum = (
            old_scale * running_sum
            + row_sum(probabilities)
        )

        output_accumulator = (
            old_scale * output_accumulator
            + probabilities @ V_block
        )

        running_max = new_max

    output_block = output_accumulator / running_sum
    write output_block
```

This omits many real GPU details but captures the tiling and online-softmax mechanism.

## 24. NumPy experiment for online softmax

```python
"""Demonstrate numerically stable online softmax."""

from __future__ import annotations

import math
import numpy as np


def stable_softmax(values: np.ndarray) -> np.ndarray:
    """Compute ordinary numerically stable softmax."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array.")

    maximum = float(np.max(values))
    exponentials = np.exp(values - maximum)
    denominator = float(np.sum(exponentials))

    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("Invalid softmax denominator.")

    return exponentials / denominator


def online_softmax(values: np.ndarray, block_size: int) -> np.ndarray:
    """Compute softmax using blockwise running max/sum statistics."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    running_max = -math.inf
    running_sum = 0.0

    for start in range(0, values.size, block_size):
        block = values[start : start + block_size]
        block_max = float(np.max(block))
        new_max = max(running_max, block_max)

        old_part = (
            0.0
            if running_max == -math.inf
            else math.exp(running_max - new_max) * running_sum
        )
        new_part = float(np.sum(np.exp(block - new_max)))

        running_sum = old_part + new_part
        running_max = new_max

    if not math.isfinite(running_sum) or running_sum <= 0.0:
        raise FloatingPointError("Invalid online denominator.")

    return np.exp(values - running_max) / running_sum


def main() -> None:
    """Compare conventional and online softmax."""
    values = np.array(
        [1.0, 2.0, 3.0, 6.0, -2.0, 8.0, 4.0],
        dtype=np.float64,
    )

    try:
        expected = stable_softmax(values)
        actual = online_softmax(values, block_size=2)

        print("Standard:", expected)
        print("Online:  ", actual)
        print("Max error:", float(np.max(np.abs(expected - actual))))

        np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
        print("Result: PASS")
    except (ValueError, FloatingPointError, AssertionError) as exc:
        print(f"Experiment failed: {exc}")


if __name__ == "__main__":
    main()
```

This intentionally materializes the final probability vector so correctness can be compared. A production FlashAttention kernel instead fuses normalization with V accumulation and does not materialize the complete attention probability matrix.

## 25. What to profile

Do not benchmark only wall-clock latency. Useful measurements include:

```text
kernel latency
achieved FLOP/s
HBM read/write bytes
DRAM bandwidth utilization
L2 traffic
shared-memory traffic
Tensor Core utilization
occupancy
register pressure
shared-memory usage
```

Then estimate:

```text
Arithmetic intensity =
estimated or measured FLOPs / measured HBM bytes
```

and compare it with the GPU's ridge point. After HBM traffic is reduced, another level of the memory hierarchy can become limiting, which motivates hierarchical Roofline analysis.

## 26. Common misconceptions

```text
Misconception:
FlashAttention is approximate.

Reality:
The core algorithm computes exact dense attention at the algorithmic level;
floating-point ordering can produce small numerical differences.

Misconception:
FlashAttention removes O(N^2) compute.

Reality:
Dense pairwise attention remains quadratic in sequence length.

Misconception:
FlashAttention is just fused softmax.

Reality:
Its core combination is IO-aware tiling + online softmax + fused output
accumulation + careful on-chip memory management.

Misconception:
Memory-bound means out of VRAM.

Reality:
Capacity and bandwidth are separate constraints.

Misconception:
FlashAttention always makes attention compute-bound.

Reality:
It raises arithmetic intensity; whether the workload crosses the hardware
ridge point must be measured.
```

## 27. Relationship to the inference-engineering sequence

```text
Bits / bytes
    |
FLOPs and bandwidth
    |
Arithmetic intensity
    |
Roofline model
    |
Prefill vs decode
    |
Batching
    |
TTFT / TPOT / throughput
    |
FlashAttention
    |
    +-- tiling
    +-- online softmax
    +-- fusion
    +-- less HBM traffic
    +-- higher arithmetic intensity
    |
KV-cache optimization
    |
Paged attention
    |
Continuous batching
    |
Chunked prefill
    |
Production LLM serving
```

FlashAttention is where the earlier theoretical concepts become a concrete GPU algorithm.

## 28. Production considerations

Validate the actual dispatch path for your model, framework, GPU, precision, head dimension, causal mode, GQA/MQA configuration, and sequence lengths. Unsupported shapes may fall back to a different kernel.

Correctness testing should cover masking, padding, variable sequence lengths, mixed precision, gradients when training, dropout semantics, GQA/MQA mapping, and numerical tolerances.

Performance optimizations must also preserve request and tenant isolation. KV caches, temporary buffers, logs, and shared caching mechanisms must not leak data across authorization boundaries.

## 29. Interview-ready explanation

```text
Standard attention can materialize large N x N score/probability intermediates
in HBM. FlashAttention tiles Q, K, and V into fast on-chip memory and uses an
online numerically stable softmax so each score tile can be consumed
immediately while accumulating the final output. This avoids writing and
rereading the full N x N attention matrix, reduces HBM traffic, raises
effective arithmetic intensity, and moves execution toward the compute roof.
The dense attention FLOP complexity remains quadratic; the main innovation is
IO-aware execution and fusion.
```

## 30. Key takeaways

```text
1. Attention logically contains an N x N score matrix.
2. Moving/materializing that intermediate through HBM is expensive.
3. FlashAttention tiles Q/K/V into fast on-chip storage.
4. Online softmax tracks a running maximum and normalization sum.
5. Old state is rescaled exactly when the running maximum changes.
6. P @ V can be accumulated online too.
7. Full N x N probabilities need not be written to HBM.
8. Dense-attention compute remains approximately O(N^2 * d).
9. HBM traffic and persistent intermediate memory fall dramatically.
10. Arithmetic intensity therefore rises.
11. Roofline position moves toward the compute ceiling.
12. Crossing the ridge point is workload- and hardware-dependent.
13. Decode can remain bandwidth-sensitive because of KV-cache traffic.
14. The deeper lesson is to optimize data movement, not FLOPs alone.
```

## Conclusion

FlashAttention is fundamentally an IO-aware execution strategy for exact dense attention.

```text
Instead of:

compute
-> write huge intermediate
-> read huge intermediate
-> compute
-> write huge intermediate
-> read huge intermediate
-> compute

FlashAttention does:

load tile
-> compute
-> normalize online
-> accumulate
-> reuse on-chip
-> discard temporary tile
-> load next tile
-> write final output
```

Online softmax is the mathematical mechanism that makes this schedule possible while preserving global row-wise normalization. The resulting reduction in HBM traffic raises arithmetic intensity. Under the Roofline model, that shifts the kernel away from the bandwidth ceiling and toward the compute ceiling.

The broader engineering principle is:

```text
The fastest GPU algorithm is not necessarily the one with the fewest FLOPs.

It is often the one that performs the most useful computation
for every expensive byte moved.
```

## References

Primary papers:

```text
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
2022

Tri Dao
FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
2023

Barna Saha, Christopher Ye
The I/O Complexity of Attention, or How Optimal is Flash Attention?
2024
```
