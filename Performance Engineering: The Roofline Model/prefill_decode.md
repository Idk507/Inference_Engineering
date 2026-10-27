# Why LLM Prefill Is Usually Compute-Bound and Decode Is Usually Memory-Bound

> A formal Roofline derivation, from FLOPs and bytes to batching, KV-cache traffic, attention, and production GPU profiling. All equations use plain text suitable for GitHub README rendering; no LaTeX is required.

## 1. The precise claim

During transformer inference, **prefill** evaluates a batch of prompt tokens concurrently, whereas **decode** evaluates typically one new token per active sequence per iteration. Prefill often forms large matrix–matrix multiplications (GEMMs) with substantial reuse of model weights. Decode often forms thin GEMMs or matrix–vector products, requiring the engine to stream large model weights and cached keys/values to perform comparatively little arithmetic.

But **prefill is not always compute-bound, and decode is not always memory-bound**. These are common *operating regimes*, not mathematical identities. The regime depends on actual arithmetic intensity, precision-dependent compute throughput, memory bandwidth, batch size, sequence length, cache behavior, and kernel efficiency.

We will derive a conditional statement:

```text
Let R = peak compute FLOP/s / peak memory bandwidth byte/s.
Let I = kernel FLOPs / bytes actually moved at the selected memory level.

If I < R, the memory-bandwidth roof is lower than the compute roof.
If I > R, the compute roof is lower than the memory-bandwidth roof.
If I = R, the two ideal roofs intersect (the ridge point).
```

A kernel below the ridge is *bandwidth-limited in the two-ceiling Roofline model*. It can still run below that bandwidth roof because of latency, occupancy, instruction overhead, communication, and other bottlenecks.

## 2. Define everything before deriving anything

| Symbol | Meaning | Units |
|---|---|---|
| `F` | Floating-point work | FLOPs |
| `M` | Memory traffic at a specified memory interface | bytes |
| `I = F / M` | Arithmetic intensity | FLOPs/byte |
| `P` | Peak compute throughput for the relevant precision/path | FLOPs/s |
| `BW` | Peak bandwidth at the specified memory interface | bytes/s |
| `R = P / BW` | Ridge point | FLOPs/byte |
| `T_compute = F / P` | Ideal compute lower bound | seconds |
| `T_memory = M / BW` | Ideal transfer lower bound | seconds |
| `s` | Bytes per stored tensor element | bytes/element |
| `H` | Hidden dimension of a dense projection | elements |
| `S` | Prompt tokens processed together | tokens |
| `B` | Active decode sequences evaluated together | sequences |
| `L` | Cached context length per decode sequence | tokens |
| `G` | Query heads sharing one KV head | dimensionless |

**Conventions.** A multiply-add is counted as two floating-point operations. Unless stated otherwise, model examples use `s = 2` for BF16/FP16 storage, no bias, no parameter sharing across layers, and a single read of each required weight matrix per GEMM. Tensor allocation size is **not** the same quantity as DRAM bytes transferred.

## 3. Derive the Roofline criterion from elapsed-time lower bounds

The compute hardware cannot execute `F` FLOPs faster than `F / P` seconds. The memory system cannot move `M` bytes faster than `M / BW` seconds. Even when computation and transfers overlap perfectly:

```text
T_actual >= max(F / P, M / BW)

Achieved FLOP/s = F / T_actual
                 <= min(P, F * BW / M)
                 = min(P, I * BW)
```

The equality between the two ideal time bounds occurs when:

```text
F / P = M / BW
F / M = P / BW
I = R
```

Thus the formal roof comparison is:

```text
I < R  =>  M / BW > F / P  =>  bandwidth roof is restrictive
I > R  =>  F / P > M / BW  =>  compute roof is restrictive
```

**Example hypothetical accelerator:**

```text
Peak compute P = 100 TFLOP/s
Peak DRAM bandwidth BW = 1 TB/s
Ridge R = 100 FLOPs/byte

At I =   1 FLOP/byte => roof =   1 TFLOP/s (bandwidth side)
At I =  32 FLOPs/byte => roof =  32 TFLOP/s (bandwidth side)
At I = 100 FLOPs/byte => roof = 100 TFLOP/s (ridge)
At I = 500 FLOPs/byte => roof = 100 TFLOP/s (compute side)
```

These figures are illustrative, not specifications of a particular GPU. For a real accelerator, select the *achievable relevant* compute path (e.g., BF16 Tensor Core) and consistent DRAM/HBM bandwidth numbers. Changing precision can change **both** the numerator workload and the compute roof.

## 4. One dense transformer layer: exact accounting for a simplified GEMM

Consider a linear layer `Y = X @ W` with:

```text
X shape: [N, H]     // N tokens processed together
W shape: [H, H]     // model projection weights
Y shape: [N, H]
```

This is representative of a transformer projection. A real block has several projections with different dimensions, but the same reasoning applies to each.

Conventional GEMM FLOP count:

```text
F(N,H) = 2 * N * H * H
```

Under *ideal compulsory traffic*, read `X` and `W` once and write `Y` once:

```text
M(N,H,s) = s * (N*H + H*H + N*H)
         = s * (H*H + 2*N*H)

I(N,H,s) = 2*N*H*H / [s*(H*H + 2*N*H)]
         = 2*N*H / [s*(H + 2*N)]
```

This formula is exact **for the stated idealized traffic model**, not a guarantee that a deployed GEMM reads all of its operands from HBM only once. The actual kernel may move additional bytes or benefit from weights already present in cache.

When `H` is large and `N` is small relative to `H`, the denominator is dominated by `s*H*H`, so:

```text
I(N,H,s) ≈ 2*N/s

With BF16/FP16 (s = 2): I ≈ N FLOPs/byte.
```

**The critical observation:** the arithmetic intensity grows roughly linearly with the number of tokens sharing each weight fetch. The weight matrix does not get larger when we process more tokens in the same GEMM, yet each loaded weight participates in more multiply-add operations.

For enormous `N`, compulsory input/output traffic becomes significant and the ideal intensity does not grow without limit:

```text
lim N -> infinity I(N,H,s) = H/s
```

## 5. Formal proof for prefill

For a prompt of `S` tokens (single request, full prompt processed jointly):

```text
X: [S, H]
W: [H, H]
F_prefill = 2*S*H*H
M_prefill = s*(H*H + 2*S*H)
I_prefill = 2*S*H / [s*(H + 2*S)]
```

To be on the compute side of a hypothetical device's ridge, require:

```text
I_prefill > R

2*S*H / [s*(H + 2*S)] > R
2*S*H > R*s*H + 2*R*s*S
2*S*(H - R*s) > R*s*H

If H > R*s:
  S > R*s*H / [2*(H - R*s)]

If H <= R*s:
  the projection's ideal intensity never exceeds R for any finite S.
```

This last condition is frequently omitted in casual explanations: **sufficiently long prefill is not guaranteed to become compute-bound on every hardware/shape combination.** Also, the *full transformer* includes attention and non-GEMM operations with other performance characteristics.

### Numerical prefill example

Set:

```text
H = 4096
s = 2 bytes (BF16)
R = 100 FLOPs/byte
S = 1024 prompt tokens
```

Then:

```text
F_prefill = 2 * 1024 * 4096^2
          = 34,359,738,368 FLOPs = 34.36 GFLOPs

M_prefill = 2 * (4096^2 + 2*1024*4096)
          = 50,331,648 bytes = 50.33 MB

I_prefill = 34,359,738,368 / 50,331,648
          = 682.67 FLOPs/byte
```

Since `682.67 > 100`, **the projection is on the compute-limited side of the ideal Roofline model**.

The model's prefill threshold for this projection is:

```text
S > (100*2*4096) / [2*(4096 - 100*2)]
S > 105.13
```

Thus an integer prompt-token dimension of **106 or larger** crosses the *idealized* ridge for this projection. Practical boundaries can be very different due to kernel sizes, hardware utilization, attention, masking, and caching.

## 6. Formal proof for decode: same weights, far fewer token rows

In standard autoregressive decoding, **each active sequence contributes one new token per iteration**. With `B` sequences batched together, the dense projection sees:

```text
X: [B, H]
W: [H, H]
Y: [B, H]

F_decode = 2*B*H*H
M_decode = s*(H*H + 2*B*H)
I_decode = 2*B*H / [s*(H + 2*B)]
```

At batch size 1:

```text
I_decode = 2*H / [s*(H + 2)]
```

For `H = 4096, s = 2`:

```text
I_decode ≈ 0.9995 FLOPs/byte.
```

This lies far below the example ridge of 100 FLOPs/byte.

The *same formal inequality* predicts the batch size required to cross the ideal ridge:

```text
If H > R*s:
  B > R*s*H / [2*(H - R*s)]
```

For our values, `B > 105.13`, so an ideal projection GEMM at `B = 106` crosses the modeled ridge, **even though it is decoding**. In a real serving system, KV-cache reads, batching overhead, attention, memory capacity, padding, and kernel efficiency alter the result.

### Direct comparison at the same hidden dimension

| Phase / active rows `N` | Approx. intensity, BF16 projection | Ideal Roofline regime (`R=100`) |
|---|---:|---|
| Decode, `B=1` | 1.00 FLOPs/byte | Bandwidth side |
| Decode, `B=8` | 7.97 FLOPs/byte | Bandwidth side |
| Decode, `B=32` | 31.51 FLOPs/byte | Bandwidth side |
| Decode, `B=128` | 120.47 FLOPs/byte | Compute side |
| Prefill, `S=16` | 15.88 FLOPs/byte | Bandwidth side |
| Prefill, `S=128` | 120.47 FLOPs/byte | Compute side |
| Prefill, `S=1024` | 682.67 FLOPs/byte | Compute side |

**This is the formal reason:** the relevant GEMM dimension is the number of token rows processed while the weights are reused. Typical prefill has a large `S`; typical latency-sensitive decode has a small `B`. A large decode batch can cross into the compute regime, and a tiny prefill can remain on the bandwidth side.

## 7. A second formal perspective: operational time per token

For a decoder-only model with `P_model` weight parameters, assume most operations are dense projections, weights are stored using `s_w` bytes/parameter, and inference processes `N` token rows against each fetched weight tensor.

A rough model (excluding attention, embeddings, activation traffic, and logits) is:

```text
F ≈ 2 * N * P_model FLOPs
M_weights ≈ s_w * P_model bytes
I_weights ≈ 2*N / s_w
```

For BF16/FP16 storage (`s_w = 2`):

```text
I_weights ≈ N FLOPs/byte.
```

Consider an 8-billion-parameter *illustrative* model:

```text
Weight storage ≈ 8e9 * 2 bytes = 16 GB
Compute per token ≈ 2 * 8e9 = 16 GFLOPs

Single sequence decode:
  F ≈ 16 GFLOPs
  M_weights ≈ 16 GB
  I_weights ≈ 1 FLOP/byte

1024-token prefill:
  F ≈ 16.384 TFLOPs
  M_weights ≈ 16 GB (weights alone)
  I_weights-only ≈ 1024 FLOPs/byte
```

The prefill `1024` figure is a **weight-only upper estimate**, not the full execution's arithmetic intensity: activation traffic, outputs, attention intermediates, and other memory operations add to its denominator. Both these simplified figures assume the weights are fetched from the chosen memory level for the forward pass. Repeated decode steps may reuse some weights via cache, depending on the model and cache capacity.

### Time comparison on illustrative hardware

Using the `100 TFLOP/s`, `1 TB/s` accelerator:

```text
Decode B=1, weights only:
  Compute lower bound = 16 GFLOPs / 100 TFLOP/s = 0.16 ms
  Weight-read lower bound = 16 GB / 1 TB/s = 16 ms
  Memory-read lower bound is 100x larger.

Prefill S=1024, weights-only comparison:
  Compute lower bound = 16.384 TFLOPs / 100 TFLOP/s = 163.84 ms
  Weight-read lower bound = 16 GB / 1 TB/s = 16 ms
  Compute lower bound is 10.24x larger than the weight-read bound.
```

Do not interpret these as real measured model latencies: they omit attention, KV traffic, activation traffic, communication, sampling, and many other costs.

## 8. Decode has another major bandwidth consumer: the KV cache

The dense-weight derivation is only half the story. During autoregressive decoding, attention must access previous tokens' cached keys and values.

For each active sequence, define:

```text
L = number of cached context tokens
h_q = number of query heads
h_kv = number of KV heads
G = h_q / h_kv = query heads per KV head
D = head dimension
s_kv = bytes per stored KV element
```

For one new query token per sequence, the leading attention dot products and weighted value sums require approximately:

```text
F_attention ≈ 4 * L * h_q * D
```

If the KV tensors are read once per step and reused across their grouped query heads, the key and value traffic is approximately:

```text
M_KV ≈ 2 * L * h_kv * D * s_kv
```

Therefore, for the **KV read contribution alone**:

```text
I_attention_KV ≈ (4*L*h_q*D) / (2*L*h_kv*D*s_kv)
               = 2*G / s_kv
```

For BF16 KV (`s_kv = 2`):

```text
I_attention_KV ≈ G FLOPs/byte
```

Thus conventional multi-head attention with `G=1` has roughly 1 FLOP/byte of KV traffic, while grouped-query attention with `G=8` has roughly 8 FLOPs/byte, **if the kernel really reuses the loaded KV data across those query heads**. For a batch of sequences with independent contexts, KV caches generally cannot be shared across sequences simply by batching.

This derivation ignores queries, outputs, scores, softmax, paging metadata, and possible redundant loads. It explains a structural limit, not a full kernel-performance forecast.

### Why context length doesn't automatically raise decode attention intensity

Both decode attention compute and cache traffic scale linearly with `L`:

```text
F_attention ∝ L
M_KV ∝ L
F_attention / M_KV ≈ constant, for fixed G and precision
```

As context grows, **latency and traffic can increase without moving the operation across the ridge**. NVIDIA's detailed attention analysis makes the same qualitative distinction between full-prompt attention and single-query KV-cache attention. [NVIDIA: Co-Designing AI Model Attention](https://developer.nvidia.com/blog/co-designing-ai-model-attention-for-fast-interactive-long-context-inference)

## 9. Why long-prefill attention can also be compute-intensive

For full causal self-attention over `S` prompt tokens, attention evaluates approximately half of the possible token pairs. Ignoring diagonal corrections, its leading computation is:

```text
F_causal_attention ≈ 2 * S^2 * h_q * D
```

The factor differs from full *bidirectional* attention (approximately `4*S^2*h_q*D`) because causal masking removes roughly half the pairwise interactions. This is an algorithmic FLOP approximation: a kernel that explicitly computes masked-out tiles may execute more operations.

Efficient tiled attention implementations avoid storing the full `S x S` attention score matrix in DRAM. When Q, K, and V tiles are reused efficiently on-chip, arithmetic work grows quadratically with prompt length while compulsory tensor-storage sizes grow roughly linearly. That permits arithmetic intensity to increase with `S`.

**Important caveat:** the exact attention intensity depends on tiling and HBM loads. KV blocks can be reloaded for different query tiles. It would be incorrect to infer actual DRAM traffic solely from the combined logical size of Q, K, and V. FlashAttention improves I/O complexity compared with attention implementations that materialize large score/probability tensors, but actual traffic remains implementation-dependent.

Prefill also performs the dense projection and MLP GEMMs derived above. The aggregate phase is usually compute-heavy when prompts and effective batches are sufficiently large and GPU kernels reach high compute utilization.

## 10. Why the two-regime claim has real exceptions

| Scenario | What changes | Possible regime |
|---|---|---|
| Tiny prefill (e.g., 8 tokens) | Very few rows reuse the weights | Memory/launch limited |
| Large batched decode | Hundreds of requests reuse weights in a GEMM | Dense layers may become compute limited |
| Prefix-cache hit and tiny uncached suffix | Very short new-query prefill, long cached context | Decode-like attention behavior |
| Extremely long contexts | KV bytes and attention costs dominate portions of execution | KV-bandwidth or attention-compute limited by kernel |
| GQA / MQA | More query heads reuse fewer KV heads | Higher KV-read intensity |
| Weight quantization | Fewer stored bytes, extra dequantization cost | Moves weight roof, not necessarily attention roof |
| KV quantization | Fewer KV bytes, conversion overhead | Improves KV bandwidth pressure |
| Speculative decoding | Multiple proposed tokens evaluated together | May move verification compute toward GEMM-like behavior |
| Chunked prefill | Fewer prefill rows evaluated in a chunk | Individual chunks can have lower intensity |
| Small GPU occupancy | Insufficient parallelism or overhead dominates | Neither ideal roof may be achieved |
| Tensor/pipeline parallelism | Communication and sharded kernel dimensions change | Network and launch costs can dominate |

These are not mere academic edge cases. Production inference engines mix prefill and decode, regulate active batch sizes, and sometimes place phases on separate worker pools because their resource profiles differ. See [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm).

## 11. The distinction between a kernel-level proof and a model-level claim

The preceding derivation **formally classifies an idealized GEMM** relative to a stated Roofline ridge point. It does not prove that an entire transformer forward pass is always limited by one resource.

A model forward pass contains multiple dense GEMMs, QK and PV attention, normalization, nonlinearities, rotary position encoding, embedding lookups, possible mixture-of-experts routing, and communication. Their intensities can differ substantially. A scheduler also adds overhead and may co-batch different work types.

To classify the **whole inference phase**, measure either:

```text
I_phase = total measured floating-point work / total measured DRAM traffic
```

or classify the dominant constituent kernels and use their measured runtimes to find the true bottleneck. Beware that a single aggregate intensity can hide individual kernels that sit on opposite sides of the ridge.

The correct wording is therefore:

> Under ordinary serving batch sizes, sufficiently long prefill tends to have high arithmetic intensity because it reuses weights across many prompt-token rows and can be dominated by large GEMMs. Autoregressive decoding tends to have low arithmetic intensity because each iteration adds few query rows while reading large weight tensors and per-sequence KV-cache data. The hardware-specific Roofline ridge determines the formal regime.

## 12. A reproducible Python proof: compute the ridge crossing

Save this as `prefill_decode_roofline.py` next to the README. It uses only the Python standard library and prints idealized FLOPs, bytes, arithmetic intensity, Roofline ceilings, and the theoretical prefill/decode crossing threshold.

```python
"""Analyze ideal GEMM arithmetic intensity for LLM prefill and decoding.

This educational model does not measure hardware memory traffic.
It assumes each operand is read once and the output is written once.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    """Hold arithmetic and memory roofs in SI units."""

    peak_tflops: float
    bandwidth_tbps: float

    def __post_init__(self) -> None:
        """Reject nonphysical or nonfinite roof specifications."""
        if not all(
            math.isfinite(v) and v > 0
            for v in (self.peak_tflops, self.bandwidth_tbps)
        ):
            raise ValueError("Compute and bandwidth peaks must be positive and finite.")

    @property
    def ridge(self) -> float:
        """Return the precision-specific ridge in FLOPs per byte."""
        return self.peak_tflops / self.bandwidth_tbps


@dataclass(frozen=True)
class GemmEstimate:
    """Describe ideal compulsory memory traffic for [rows,H] @ [H,H]."""

    rows: int
    hidden: int
    element_bytes: int

    def __post_init__(self) -> None:
        """Validate integer matrix dimensions and storage size."""
        if min(self.rows, self.hidden, self.element_bytes) <= 0:
            raise ValueError("All GEMM dimensions and element bytes must be > 0.")

    @property
    def flops(self) -> int:
        """Count matrix multiply-add arithmetic at two FLOPs per pair."""
        return 2 * self.rows * self.hidden**2

    @property
    def bytes_moved(self) -> int:
        """Count one input read, one weight read, and one output write."""
        return self.element_bytes * (
            self.hidden**2 + 2 * self.rows * self.hidden
        )

    @property
    def intensity(self) -> float:
        """Return FLOPs divided by assumed bytes moved."""
        return self.flops / self.bytes_moved


def crossing_threshold(device: Device, hidden: int, element_bytes: int) -> int | None:
    """Return smallest integer row count strictly above the ideal ridge.

    Return None if no finite row count can exceed the ridge.
    """
    if hidden <= 0 or element_bytes <= 0:
        raise ValueError("Hidden dimension and element bytes must be > 0.")

    r = device.ridge
    if hidden <= r * element_bytes:
        return None

    threshold = (r * element_bytes * hidden) / (
        2 * (hidden - r * element_bytes)
    )
    candidate = math.floor(threshold) + 1

    # Correct possible floating-point rounding at the boundary.
    while GemmEstimate(candidate, hidden, element_bytes).intensity <= r:
        candidate += 1
    return candidate


def describe(label: str, estimate: GemmEstimate, device: Device) -> None:
    """Print work, traffic, Roofline limits, and the predicted regime."""
    intensity = estimate.intensity
    memory_roof = intensity * device.bandwidth_tbps
    attainable_roof = min(device.peak_tflops, memory_roof)
    compute_time_ms = estimate.flops / (device.peak_tflops * 1e12) * 1e3
    memory_time_ms = estimate.bytes_moved / (
        device.bandwidth_tbps * 1e12
    ) * 1e3

    regime = "compute-side" if intensity > device.ridge else "bandwidth-side"
    if math.isclose(intensity, device.ridge, rel_tol=1e-12):
        regime = "ridge"

    print(f"\n{label} ({estimate.rows} token rows)")
    print(f"  FLOPs             : {estimate.flops / 1e9:.5f} GFLOPs")
    print(f"  Assumed traffic   : {estimate.bytes_moved / 1e6:.5f} MB")
    print(f"  Arithmetic intensity: {intensity:.5f} FLOPs/byte")
    print(f"  Memory roof       : {memory_roof:.5f} TFLOP/s")
    print(f"  Attainable roof   : {attainable_roof:.5f} TFLOP/s")
    print(f"  Compute lower bound: {compute_time_ms:.5f} ms")
    print(f"  Memory lower bound : {memory_time_ms:.5f} ms")
    print(f"  Model classification: {regime}")


def main() -> None:
    """Parse arguments and compare prefill with multiple decode batch sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--element-bytes", type=int, default=2)
    parser.add_argument("--peak-tflops", type=float, default=100.0)
    parser.add_argument("--bandwidth-tbps", type=float, default=1.0)
    args = parser.parse_args()

    if args.hidden <= 0 or args.prompt_tokens <= 0 or args.element_bytes <= 0:
        parser.error("Dimensions and element bytes must be positive integers.")

    try:
        device = Device(args.peak_tflops, args.bandwidth_tbps)
        print(f"Theoretical ridge: {device.ridge:.3f} FLOPs/byte")

        threshold = crossing_threshold(device, args.hidden, args.element_bytes)
        print(f"Minimum compute-side token rows: {threshold or 'unreachable'}")

        describe(
            "PREFILL",
            GemmEstimate(args.prompt_tokens, args.hidden, args.element_bytes),
            device,
        )
        for batch in (1, 8, 32, 128):
            describe(
                f"DECODE batch={batch}",
                GemmEstimate(batch, args.hidden, args.element_bytes),
                device,
            )
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
```

Run:

```bash
python prefill_decode_roofline.py
python prefill_decode_roofline.py --prompt-tokens 16
python prefill_decode_roofline.py --peak-tflops 200 --bandwidth-tbps 1
python prefill_decode_roofline.py --element-bytes 4
```

The third command doubles the hypothetical compute roof without changing bandwidth, increasing the ridge from 100 to 200 FLOPs/byte. That can move **the same operation** from the compute side to the bandwidth side. This establishes that compute-/memory-bound classification is a joint property of **workload and hardware**.

## 13. Going from analytical estimates to actual profiling

A defensible production investigation proceeds in three stages.

**Stage A — analytical model.** Compute `F`, expected input/output bytes, the chosen compute peak `P`, and bandwidth peak `BW`. Classify each dominant GEMM and attention kernel by `I` relative to `R`. Keep weights, activations, and KV-cache traffic separate to identify what can be optimized.

**Stage B — actual GPU measurements.** Use NVIDIA Nsight Compute for kernel-level memory counters, achieved throughput, and Roofline analysis. Use PyTorch Profiler for operator attribution and execution timeline; PyTorch allocated memory is **not** cumulative DRAM bytes transferred. Use warm-up iterations, synchronize GPU timing, and repeat experiments to estimate variance.

**Stage C — system-level serving metrics.** Capture time to first token (TTFT), inter-token latency (ITL), tokens per second, requests per second, active batch size, prompt/context length distributions, KV-cache occupancy, preemption/recomputation, and queueing delays. A low-level kernel optimization that increases throughput may harm tail ITL or TTFT under realistic concurrency.

When profiling, ensure device frequency and power limits, thermals, actual precision, Tensor Core eligibility, and the number of occupied SMs are considered. On a multi-GPU deployment, also track interconnect traffic and synchronization. A DRAM-only Roofline cannot explain network-bound tensor parallelism.

## 14. How these results guide inference-system architecture

### Continuous batching

Decode intensity of dense layers increases with active batch size because weight reads can be shared across more token rows. A scheduler can admit new requests as old requests finish rather than waiting for static batches. It must also protect KV-cache capacity and respect service-level latency objectives.

### Chunked prefill

Very long prompts can be divided into smaller token chunks so prefill work can coexist with decode requests. This improves scheduling flexibility and can help latency, but a chunk with fewer effective token rows may have lower GEMM intensity than a large monolithic prefill. Chunk size is therefore a scheduling and efficiency trade-off.

### Disaggregated prefill and decode

Compute-oriented prefill workers and bandwidth-oriented decode workers can be scaled and tuned independently. The architecture introduces KV-cache transfers between pools; those transfers have their own bandwidth and latency costs. Co-location can be better at low volume or when transfer overhead exceeds the benefit of disaggregation.

### Quantization

Weight-only INT4/INT8 reduces weight bytes moved and can improve bandwidth-sensitive decode, provided dequantization overhead does not erase the gain. KV quantization reduces a distinct bottleneck: attention-cache traffic and capacity. Neither should be evaluated by nominal bit width alone; measure quality impact, throughput, per-token latency, and actual kernel behavior.

### GQA and MQA

Grouped-query and multi-query attention reduce the number of distinct KV heads and often reduce memory footprint and KV bandwidth. Their advantage depends on efficient KV reuse and the desired model-quality trade-offs.

### Speculative decoding

Processing or verifying several candidate tokens in one target-model forward pass can create larger GEMMs and improve weight reuse. The total benefit depends on draft quality, acceptance rate, verification cost, and latency.

## 15. A compact decision procedure

```text
1. Choose the workload unit (one kernel, one decoder block, or one inference phase).
2. Count relevant arithmetic operations.
3. Choose one memory boundary (HBM/DRAM, L2, etc.) and count bytes crossing it.
4. Calculate I = FLOPs / bytes.
5. Choose peak compute for the *actual* precision and execution mechanism.
6. Choose peak bandwidth for the *same* memory boundary.
7. Calculate ridge R = peak compute / peak bandwidth.
8. Compare I with R; compute F/P and M/BW lower bounds.
9. Profile real execution and identify overhead below the ideal roofs.
10. Optimize the measured bottleneck, then repeat.
```

## 16. Final derivation to remember

```text
Dense projection:
  X[N,H] @ W[H,H] -> Y[N,H]

Compute:
  F = 2*N*H^2

Ideal compulsory traffic:
  M = s*(H^2 + 2*N*H)

Arithmetic intensity:
  I = 2*N*H / [s*(H + 2*N)]

With H >> N:
  I ≈ 2*N/s

Prefill:
  N = prompt tokens processed concurrently (S)
  S is often large -> high weight reuse -> high arithmetic intensity

Decode:
  N = active sequences (B), one new token each
  B is often small -> low weight reuse -> low arithmetic intensity

Roofline criterion:
  R = peak compute FLOP/s / peak memory bandwidth byte/s

  I < R => bandwidth roof is lower
  I > R => compute roof is lower

Attention decode KV traffic:
  I_attention_KV ≈ 2*G/s_kv
  (assuming efficient reuse of each KV head across G query heads)
```

**Conclusion:** Prefill and decode have different bottlenecks primarily because they expose *different amounts of reusable work per model-weight and KV-cache byte*. Their conventional classifications follow from comparing their arithmetic intensities against the hardware ridge point, not from the names of the phases themselves.

## References

1. [NVIDIA — Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization)
2. [NVIDIA — Co-Designing AI Model Attention for Fast, Interactive Long-Context Inference](https://developer.nvidia.com/blog/co-designing-ai-model-attention-for-fast-interactive-long-context-inference)
3. [vLLM — Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
4. [NVIDIA — Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
5. [PyTorch — Profiler Documentation](https://docs.pytorch.org/docs/stable/profiler.html)
6. [FlashAttention — Official Repository](https://github.com/Dao-AILab/flash-attention)

---

*Scope note:* Calculations in this guide are analytical illustrations. They should be validated against kernel-level hardware counters and actual workload distributions before being used for GPU procurement, capacity planning, or production latency commitments.
