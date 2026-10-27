# How Batching Shifts Arithmetic Intensity

## Why Increasing Batch Size Raises GPU and LLM Inference Throughput

Batching is one of the most important performance techniques in GPU computing and large-language-model inference. The usual explanation is that "a larger batch uses the GPU better." That statement is correct, but incomplete. The deeper reason is that batching changes the ratio between useful computation and data movement.

That ratio is **arithmetic intensity**.

```text
Arithmetic Intensity = FLOPs performed / Bytes moved
```

For many neural-network operations, especially the large linear projections inside transformers, increasing batch size allows weights fetched from GPU memory to be reused across multiple input vectors. The amount of computation therefore increases much faster than the amount of weight traffic. Arithmetic intensity rises, the workload moves to the right on the Roofline model, and the GPU can achieve more useful FLOPs per second.

This chapter derives that behavior formally rather than treating batching as a vague GPU-utilization trick.

---

## 1. The Performance Problem Batching Solves

Consider a GPU running an LLM. The model contains billions of parameters stored in GPU HBM. During inference, those parameters must participate in matrix multiplications.

A GPU has two fundamentally different resources:

```text
Compute system
    Tensor Cores
    CUDA cores
    matrix-multiplication pipelines

Memory system
    HBM / VRAM
    memory controllers
    caches
    shared memory
    registers
```

The compute system might theoretically execute hundreds of trillions of operations per second, while HBM can only deliver a few terabytes per second.

If every small piece of computation requires another expensive trip to HBM, the compute units spend much of their time waiting for data.

Batching helps because it creates opportunities to reuse data after it has been fetched.

---

## 2. Start With One Linear Layer

A transformer contains many linear layers. Consider:

```text
Y = XW
```

For a single token:

```text
X shape = [1, K]
W shape = [K, N]
Y shape = [1, N]
```

For every output element, approximately `K` multiplications and `K` additions are performed.

A multiply-add conventionally counts as two FLOPs.

Therefore:

```text
FLOPs ≈ 2 * K * N
```

Suppose the weights are FP16 or BF16.

```text
bytes_per_weight = 2
```

The weight matrix occupies:

```text
Weight bytes = 2 * K * N
```

If weight traffic dominates, arithmetic intensity is approximately:

```text
AI ≈ FLOPs / Weight bytes

AI ≈ (2 * K * N) / (2 * K * N)

AI ≈ 1 FLOP/byte
```

This is extremely low arithmetic intensity.

The GPU performs roughly one floating-point operation for every byte of model weight delivered from memory.

---

## 3. Why Batch Size Changes the Equation

Now process `B` input vectors simultaneously.

Instead of:

```text
X shape = [1, K]
```

we have:

```text
X shape = [B, K]
```

The operation becomes:

```text
Y = XW

X = [B, K]
W = [K, N]
Y = [B, N]
```

The number of FLOPs becomes:

```text
FLOPs ≈ 2 * B * K * N
```

The critical observation is that the weight matrix does not become `B` times larger.

```text
Weight size = K * N
```

A high-performance GEMM implementation loads tiles of `W` and reuses those values while computing outputs for multiple rows of `X`.

In the idealized weight-dominated case:

```text
Weight traffic ≈ s * K * N
```

where `s` is bytes per element.

Therefore:

```text
AI ≈ (2 * B * K * N) / (s * K * N)

AI ≈ 2B / s
```

For FP16/BF16:

```text
s = 2 bytes

AI ≈ B FLOPs/byte
```

This gives the central batching result:

```text
Batch size 1    -> approximately   1 FLOP/byte
Batch size 2    -> approximately   2 FLOPs/byte
Batch size 8    -> approximately   8 FLOPs/byte
Batch size 32   -> approximately  32 FLOPs/byte
Batch size 64   -> approximately  64 FLOPs/byte
Batch size 128  -> approximately 128 FLOPs/byte
```

This approximation assumes weight traffic dominates and that the implementation successfully reuses weights rather than repeatedly fetching them from HBM.

---

## 4. The Important Insight: FLOPs Increase, Weight Bytes Barely Do

Suppose a particular matrix contains 100 MB of weights.

For batch size 1:

```text
Read approximately 100 MB of weights.
Perform computation for 1 input.
```

For batch size 32, an efficient matrix multiplication conceptually does:

```text
Read approximately the same weight matrix.
Reuse weight tiles.
Perform computation for 32 inputs.
```

The amount of useful arithmetic has increased roughly 32 times while the dominant weight matrix has not increased in size.

That is why batching can dramatically increase arithmetic intensity.

The important concept is **reuse**.

```text
Without batching:

HBM -> W -> compute for token 1

With batching:

             -> compute for token 1
             -> compute for token 2
HBM -> W ----> compute for token 3
             -> ...
             -> compute for token B
```

Real GPU kernels achieve this reuse through tiling across registers, shared memory and caches rather than literally loading an entire matrix once.

---

## 5. The More Complete Arithmetic-Intensity Equation

Ignoring activation traffic is useful for intuition, but it is not the complete model.

For:

```text
Y = XW
```

with:

```text
X = [B, K]
W = [K, N]
Y = [B, N]
```

approximately:

```text
FLOPs = 2BKN
```

The minimum conceptual tensor traffic includes:

```text
Read X  = BK elements
Read W  = KN elements
Write Y = BN elements
```

If each element occupies `s` bytes:

```text
Bytes ≈ s * (BK + KN + BN)
```

Therefore:

```text
AI ≈ 2BKN / [s * (BK + KN + BN)]
```

This equation is much more useful than simply saying:

```text
AI ≈ B
```

because it shows that arithmetic intensity cannot increase linearly forever.

---

## 6. Why Arithmetic Intensity Eventually Saturates

Start with:

```text
AI(B) = 2BKN / [s * (BK + KN + BN)]
```

Group the terms involving `B`:

```text
AI(B) = 2BKN / [s * (KN + B(K + N))]
```

When `B` is small, the `KN` weight term can dominate:

```text
KN >> B(K + N)
```

Then:

```text
AI(B) ≈ 2BKN / (sKN)

AI(B) ≈ 2B/s
```

For FP16:

```text
AI(B) ≈ B
```

But as `B` becomes very large, activation traffic becomes increasingly important.

For extremely large `B`:

```text
B(K + N) >> KN
```

Then:

```text
AI(B) ≈ 2BKN / [sB(K + N)]
```

Cancel `B`:

```text
AI_max ≈ 2KN / [s(K + N)]
```

Notice what disappeared:

```text
B
```

Arithmetic intensity approaches an upper limit.

Therefore:

```text
Batch size increases AI strongly at first.

Then the improvement becomes smaller.

Eventually AI approaches a saturation region.
```

---

## 7. Worked Example With a Transformer Projection

Suppose:

```text
K = 4096
N = 4096
precision = FP16
s = 2 bytes
```

The weight matrix contains:

```text
4096 * 4096
= 16,777,216 parameters
```

Weight storage is approximately:

```text
16,777,216 * 2
= 33,554,432 bytes
≈ 32 MiB
```

For batch size 1:

```text
FLOPs = 2 * 1 * 4096 * 4096
      = 33,554,432 FLOPs
```

Conceptual traffic is:

```text
Bytes = 2 * [(1 * 4096) + (4096 * 4096) + (1 * 4096)]
      = 33,570,816 bytes
```

So:

```text
AI ≈ 33,554,432 / 33,570,816
   ≈ 1.00 FLOP/byte
```

For batch size 32:

```text
FLOPs = 2 * 32 * 4096 * 4096
      = 1,073,741,824 FLOPs
```

Traffic becomes:

```text
Bytes = 2 * [(32 * 4096) + (4096 * 4096) + (32 * 4096)]
      = 34,078,720 bytes
```

Therefore:

```text
AI ≈ 31.51 FLOPs/byte
```

The computation increased by 32x, while conceptual tensor traffic increased only slightly because the weight matrix still dominates the bytes.

That is the mathematical heart of batching.

---

## 8. Connecting Batching to the Roofline Model

The Roofline model says:

```text
Achievable Performance = min(
    Peak Compute,
    Arithmetic Intensity * Peak Memory Bandwidth
)
```

Suppose a hypothetical GPU provides:

```text
Peak compute = 300 TFLOP/s
Peak bandwidth = 3 TB/s
```

The ridge point is:

```text
Ridge AI = Peak Compute / Peak Bandwidth

Ridge AI = 300 TFLOP/s / 3 TB/s

Ridge AI = 100 FLOPs/byte
```

A kernel below approximately 100 FLOPs/byte tends to sit on the bandwidth side of this simplified Roofline model.

A kernel above approximately 100 FLOPs/byte can become compute-limited if it is otherwise efficient.

Now imagine FP16 arithmetic intensity approximately follows batch size in the weight-dominated region.

```text
B = 1     -> AI ≈ 1
B = 8     -> AI ≈ 8
B = 32    -> AI ≈ 32
B = 64    -> AI ≈ 64
B = 128   -> AI ≈ 128
```

The workload moves horizontally across the Roofline graph:

```text
Performance
    ^
    |
    |                             ---------------- Peak compute
    |                           /
    |                         /
    |                       /
    |                     /
    |                   /
    |                 /
    |               /
    |             /
    |           /
    +-------------------------------------------------> Arithmetic intensity
        B=1    B=8    B=32     B=64     B=128
                                         |
                                   around ridge region
```

Batching does not increase the GPU's hardware peak bandwidth or peak compute.

Instead, it changes the **workload's position relative to those hardware ceilings**.

---

## 9. Why Throughput Rises

Consider a memory-bound kernel.

From the Roofline model:

```text
Performance ≈ AI * Memory Bandwidth
```

Suppose memory bandwidth is fixed at:

```text
3 TB/s
```

If arithmetic intensity is:

```text
1 FLOP/byte
```

then the bandwidth roof is approximately:

```text
3 TFLOP/s
```

If batching raises AI to:

```text
32 FLOPs/byte
```

then:

```text
Performance ceiling ≈ 32 * 3
                    ≈ 96 TFLOP/s
```

If batching raises AI to 64:

```text
Performance ceiling ≈ 192 TFLOP/s
```

At AI 100:

```text
Performance ceiling ≈ 300 TFLOP/s
```

which reaches the hypothetical compute roof.

Beyond that point, the simple Roofline prediction becomes:

```text
Performance ≈ Peak Compute
```

and further arithmetic-intensity improvements cannot raise FLOP/s above the compute ceiling.

This explains the common throughput curve:

```text
Throughput
    ^
    |                         _____________
    |                    ____/
    |                ___/
    |             __/
    |          __/
    |       __/
    |    __/
    |___/
    +---------------------------------------> Batch size

       strong gains      diminishing returns
```

---

## 10. Throughput Is Not the Same as Latency

This distinction is critical in inference systems.

**Latency** asks:

```text
How long did one request take?
```

**Throughput** asks:

```text
How many tokens or requests can the system process per second?
```

Suppose one request takes:

```text
10 ms
```

and produces one result.

Throughput is approximately:

```text
1 / 0.010 = 100 requests/s
```

Now suppose a batch of 16 takes 25 ms.

Individual batch latency is higher:

```text
25 ms > 10 ms
```

but throughput becomes:

```text
16 / 0.025 = 640 requests/s
```

So:

```text
Latency worsened:   10 ms -> 25 ms
Throughput improved: 100 -> 640 requests/s
```

Batching is fundamentally a throughput optimization. Whether it also improves latency depends on the workload, scheduler, kernel efficiency and whether requests had to wait to form the batch.

---

## 11. Why This Matters So Much During LLM Decode

Autoregressive generation repeatedly performs something conceptually like:

```text
one current token
    -> transformer layers
    -> next-token logits
    -> sample next token
    -> repeat
```

For one sequence, many large weight matrices must participate in computation for only a small amount of token work.

A simplified projection at batch 1 resembles:

```text
[1, K] @ [K, N]
```

This is close to matrix-vector multiplication.

Its arithmetic intensity is low because a huge matrix is accessed to perform relatively little arithmetic.

With multiple active sequences:

```text
[B, K] @ [K, N]
```

we move toward matrix-matrix multiplication.

The same weights can support computation for multiple sequences.

Therefore:

```text
More concurrent sequences
        ↓
More reuse of model weights
        ↓
More FLOPs per byte fetched
        ↓
Higher arithmetic intensity
        ↓
Better use of available compute
        ↓
Higher token throughput
```

This is one of the central performance reasons high-throughput LLM servers batch decode requests.

---

## 12. Why Prefill Already Has Natural Batching Along the Token Dimension

Suppose a prompt contains `S` tokens.

A projection during prefill looks approximately like:

```text
X = [S, K]
W = [K, N]

Y = XW
```

Even for one user request, `S` can be hundreds or thousands.

The weight matrix can therefore be reused across many prompt-token rows.

For example:

```text
S = 2048

[2048, K] @ [K, N]
```

This already resembles a large GEMM.

During decode for one sequence:

```text
[1, K] @ [K, N]
```

Hence the usual distinction:

```text
Prefill:
large token dimension
high weight reuse
higher arithmetic intensity
frequently compute-bound

Decode at low concurrency:
small token dimension
low weight reuse
low arithmetic intensity
frequently bandwidth-bound
```

Decode batching increases the effective number of rows being processed together and partially restores the reuse that prefill naturally obtains from its sequence dimension.

---

## 13. Batch Size and Sequence Length Are Closely Related for Linear Layers

For many transformer projections, what matters to the GEMM is the number of token vectors processed together.

During prefill:

```text
M ≈ batch_size * prompt_tokens
```

During ordinary decode:

```text
M ≈ number_of_active_sequences
```

The matrix multiplication can often be viewed as:

```text
[M, K] @ [K, N]
```

Therefore increasing either prompt-token parallelism or request concurrency can increase reuse of the `[K, N]` weights.

This is why the underlying mathematical concept is broader than "batch size" alone.

It is about how much independent work can reuse the same data before that data must be fetched again from expensive memory.

---

## 14. Why Batch Size Does Not Give Perfect Linear Scaling

The idealized argument says:

```text
AI ≈ B
```

for FP16 in a weight-dominated projection.

Real systems deviate from this for several reasons.

First, activation traffic grows with batch size:

```text
X traffic ∝ B
Y traffic ∝ B
```

Second, cache capacity is finite. Tiles may be evicted and fetched again.

Third, Tensor Core utilization has discrete tile shapes. Some dimensions map more efficiently to hardware than others.

Fourth, kernel launch overhead matters more for tiny workloads but becomes amortized as batches grow.

Fifth, large batches increase memory capacity requirements.

Sixth, transformer decoding also moves KV-cache data, not only model weights.

Seventh, synchronization, scheduling and sampling operations introduce additional overheads.

Therefore the real throughput curve typically exhibits diminishing returns rather than perfect linear scaling.

---

## 15. KV Cache Changes the Decode Equation

For attention during autoregressive decoding, the system must read historical keys and values.

For a sequence length `L`, KV-cache traffic grows approximately with the amount of cached context that must be accessed.

A simplified decode traffic model can be written as:

```text
Total bytes ≈
    model-weight traffic
  + KV-cache traffic
  + activation traffic
  + output traffic
```

Batching is excellent at amortizing shared model-weight traffic because every sequence uses the same model parameters.

But different sequences generally have different KV-cache contents.

Therefore KV traffic does not enjoy the same cross-request reuse as model weights.

Conceptually:

```text
Model weights
    shared by all requests
    -> excellent batching reuse

KV cache
    request-specific
    -> grows with active sequences and context lengths
```

At high batch sizes or long contexts, KV-cache bandwidth can therefore become an increasingly important bottleneck even after model-weight arithmetic intensity has improved substantially.

---

## 16. Quantization Changes the Arithmetic-Intensity Picture

Suppose model weights are FP16.

```text
Weight bytes per parameter ≈ 2
```

With INT8:

```text
Weight bytes per parameter ≈ 1
```

With packed INT4:

```text
Weight bytes per parameter ≈ 0.5
```

Ignoring metadata and dequantization overhead, reducing weight bytes increases effective arithmetic intensity because fewer bytes need to be moved for similar useful computation.

The simplified weight-dominated expression is:

```text
AI ≈ 2B / s
```

So conceptually:

```text
FP16: s = 2
AI ≈ B

INT8: s = 1
AI ≈ 2B

INT4: s = 0.5
AI ≈ 4B
```

Real quantized kernels also incur scale/zero-point traffic and conversion work, so this is an analytical approximation rather than an exact measured value.

Still, it explains why quantization and batching complement each other particularly well for bandwidth-bound inference.

---

## 17. Continuous Batching

Traditional static batching might wait for a group of requests and process them together until every sequence finishes.

The problem is that generated sequence lengths differ.

Suppose:

```text
Request A -> 20 output tokens
Request B -> 200 output tokens
Request C -> 50 output tokens
Request D -> 500 output tokens
```

When A and C finish, their batch slots become useless if the system cannot admit new work.

The effective batch shrinks.

That reduces:

```text
concurrent token work
weight reuse
arithmetic intensity
GPU utilization
throughput
```

Continuous batching instead maintains a dynamic set of active sequences.

Conceptually:

```text
Iteration 1:
[A B C D]

A finishes.

Iteration 2:
[E B C D]

C finishes.

Iteration 3:
[E B F D]
```

The scheduler attempts to keep enough work active to use the GPU efficiently.

This is why continuous batching is not merely a request-management feature. It is directly connected to hardware efficiency.

---

## 18. Batching and GPU Occupancy Are Related but Not Identical

Arithmetic intensity and occupancy are often confused.

**Arithmetic intensity** is:

```text
FLOPs / bytes moved
```

**Occupancy** concerns how many warps can reside on a streaming multiprocessor relative to the hardware maximum.

Increasing batch size can improve both, but through different mechanisms.

Batching may increase arithmetic intensity by enabling data reuse. It may also provide more parallel work, allowing more thread blocks and warps to execute.

But high occupancy does not automatically imply high arithmetic intensity, and high arithmetic intensity does not automatically guarantee high occupancy.

Performance analysis should treat them separately.

---

## 19. A More Accurate Mental Model

Do not memorize:

```text
larger batch = faster
```

Instead think:

```text
larger batch
    ↓
more independent work available simultaneously
    ↓
shared weights can be reused across more work
    ↓
FLOPs increase faster than dominant weight traffic
    ↓
higher arithmetic intensity
    ↓
workload moves right on Roofline model
    ↓
more memory bandwidth can be converted into useful FLOP/s
    ↓
GPU compute utilization rises
    ↓
throughput increases
```

Until eventually:

```text
compute ceiling
or
memory capacity
or
KV-cache bandwidth
or
scheduler overhead
or
latency SLO
```

becomes the next bottleneck.

---

## 20. Practical Python Experiment

The following script computes the theoretical arithmetic intensity of a matrix multiplication as batch size changes.

```python
"""Explore how batch size changes arithmetic intensity for a linear layer."""

from __future__ import annotations


def arithmetic_intensity(
    batch_size: int,
    k: int,
    n: int,
    bytes_per_element: int = 2,
) -> float:
    """Calculate idealized arithmetic intensity for Y = XW.

    Args:
        batch_size: Number of input rows processed together.
        k: Input feature dimension.
        n: Output feature dimension.
        bytes_per_element: Storage size of each tensor element.

    Returns:
        Arithmetic intensity in FLOPs per byte.

    Raises:
        ValueError: If any supplied dimension or element size is non-positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if k <= 0 or n <= 0:
        raise ValueError("Matrix dimensions must be positive")
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")

    flops = 2 * batch_size * k * n

    input_elements = batch_size * k
    weight_elements = k * n
    output_elements = batch_size * n

    bytes_moved = bytes_per_element * (
        input_elements + weight_elements + output_elements
    )

    return flops / bytes_moved


def main() -> None:
    """Print arithmetic intensity across several batch sizes."""
    k = 4096
    n = 4096

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    print("Batch Size | Arithmetic Intensity (FLOPs/byte)")
    print("-" * 52)

    for batch_size in batch_sizes:
        try:
            ai = arithmetic_intensity(
                batch_size=batch_size,
                k=k,
                n=n,
                bytes_per_element=2,
            )
        except ValueError as exc:
            print(f"Unable to evaluate batch {batch_size}: {exc}")
            continue

        print(f"{batch_size:10d} | {ai:10.2f}")


if __name__ == "__main__":
    main()
```

Expected values are approximately:

```text
Batch Size | Arithmetic Intensity
----------------------------------
1          | ~1
2          | ~2
4          | ~4
8          | ~8
16         | ~16
32         | ~31.5
64         | ~62
128        | ~120
256        | ~228
512        | ~410
```

The deviation from perfect `AI = B` scaling becomes increasingly visible because activation traffic grows with batch size.

---

## 21. Finding the Batch Size Needed to Cross the Ridge Point

Suppose the hardware ridge point is `R` FLOPs/byte.

We have:

```text
AI(B) = 2BKN / [s(KN + B(K + N))]
```

To reach the ridge point:

```text
2BKN / [s(KN + B(K + N))] >= R
```

Multiply both sides by the denominator:

```text
2BKN >= Rs(KN + B(K + N))
```

Expand:

```text
2BKN >= RsKN + RsB(K + N)
```

Move the terms involving `B` together:

```text
B[2KN - Rs(K + N)] >= RsKN
```

Therefore, when the denominator is positive:

```text
B >= RsKN / [2KN - Rs(K + N)]
```

This is a formal estimate for the batch size required for this idealized GEMM to reach a given Roofline ridge point.

There is also an important edge case.

If:

```text
2KN - Rs(K + N) <= 0
```

then even an infinitely large batch cannot reach that ridge point under this traffic model, because the operation's maximum arithmetic intensity is too low.

That maximum is:

```text
AI_max = 2KN / [s(K + N)]
```

This gives a powerful engineering test:

```text
If AI_max < hardware ridge point:
    batching alone can never make this operation compute-bound.
```

---

## 22. Why Real Throughput Can Improve Even Before Arithmetic Intensity Changes Dramatically

Arithmetic intensity is a major reason batching helps, but it is not the only one.

Larger workloads can also improve:

```text
Tensor Core tile utilization
SM occupancy
instruction-level parallelism
kernel-launch amortization
scheduler efficiency
memory-request coalescing
collective efficiency in distributed inference
```

Therefore a measured throughput improvement should not automatically be attributed entirely to arithmetic intensity.

The Roofline model answers one specific question:

```text
How does the relationship between computation and memory traffic constrain performance?
```

A complete profiler investigation must examine other ceilings as well.

---

## 23. Production LLM Serving Trade-Off

A serving system generally wants a batch large enough to obtain strong GPU efficiency without violating user-facing latency requirements.

Conceptually:

```text
Very small batch
    low queueing delay
    poor weight reuse
    low arithmetic intensity
    low throughput

Moderate batch
    good reuse
    high GPU utilization
    strong throughput
    acceptable latency

Very large batch
    diminishing throughput gains
    higher queueing delay
    larger KV-cache footprint
    possible memory pressure
    possible latency-SLO violations
```

Therefore production systems do not simply maximize batch size.

They optimize under constraints such as:

```text
TTFT  = time to first token
TPOT  = time per output token
ITL   = inter-token latency
TPS   = tokens per second
request throughput
GPU memory capacity
KV-cache utilization
p50/p95/p99 latency
```

Dynamic schedulers continuously trade these objectives against one another.

---

## 24. Observability: What to Measure

To validate the theory on real hardware, measure more than wall-clock latency.

Useful metrics include:

```text
achieved FLOP/s
HBM bandwidth utilization
L2-cache hit rate
Tensor Core utilization
SM utilization
kernel duration
kernel launch count
batch size
active sequence count
KV-cache occupancy
tokens/s
TTFT
TPOT
p50/p95/p99 latency
```

Useful tools include NVIDIA Nsight Compute, NVIDIA Nsight Systems, PyTorch Profiler and framework-specific serving metrics.

For a Roofline-style investigation, estimate or measure:

```text
FLOPs executed
DRAM bytes transferred
Arithmetic intensity = FLOPs / DRAM bytes
Achieved performance = FLOPs / execution time
```

Then compare the measured point with the hardware's relevant compute and bandwidth ceilings.

---

## 25. Common Misconceptions

### "Batching makes the GPU faster"

The GPU's physical peak compute and bandwidth do not increase. Batching changes how efficiently the workload uses those resources.

### "Batch size B means exactly B times higher arithmetic intensity"

Only approximately in a weight-dominated regime. Activation, KV-cache and other traffic eventually prevent linear scaling.

### "Larger batches always reduce latency"

No. Batching primarily improves throughput. Queueing and larger execution groups can increase request latency.

### "Once decode is batched, it is always compute-bound"

No. Whether it crosses the ridge point depends on hardware, precision, model dimensions, batch size, KV traffic and kernel implementation.

### "High GPU utilization proves the kernel is compute-bound"

No. Utilization metrics alone do not identify the limiting resource. Roofline analysis requires FLOP and byte relationships.

---

## 26. Final Derivation to Remember

For a linear layer:

```text
Y = XW

X = [B, K]
W = [K, N]
Y = [B, N]
```

Compute:

```text
FLOPs ≈ 2BKN
```

Conceptual tensor traffic:

```text
Bytes ≈ s(BK + KN + BN)
```

Arithmetic intensity:

```text
AI(B) ≈ 2BKN / [s(BK + KN + BN)]
```

When weights dominate traffic:

```text
AI(B) ≈ 2B/s
```

For FP16/BF16:

```text
AI(B) ≈ B FLOPs/byte
```

As batch becomes extremely large:

```text
AI_max ≈ 2KN / [s(K + N)]
```

Roofline performance is bounded by:

```text
Performance <= min(
    Peak Compute,
    AI(B) * Peak Memory Bandwidth
)
```

Therefore the exact performance argument is:

```text
Batching
    -> increases reusable work per weight fetch
    -> increases FLOPs faster than dominant weight traffic
    -> raises arithmetic intensity
    -> raises the bandwidth-limited Roofline ceiling
    -> improves achieved compute utilization
    -> increases throughput
    -> until another bottleneck becomes dominant
```

---

## 27. Connection to the Larger Inference-Engineering Picture

The concepts in this learning sequence now fit together naturally:

```text
Arithmetic intensity
        |
        v
FLOPs performed per byte moved
        |
        v
Roofline model
        |
        +---- low AI  -> bandwidth-bound
        |
        +---- high AI -> compute-bound
        |
        v
Prefill
        |
        +---- many tokens reuse weights
        +---- high AI
        +---- often compute-bound
        |
        v
Decode at low concurrency
        |
        +---- little work per weight fetch
        +---- low AI
        +---- often memory-bound
        |
        v
Batching
        |
        +---- more active tokens reuse weights
        +---- AI increases
        +---- throughput increases
        |
        v
Continuous batching
        |
        +---- maintains concurrency dynamically
        +---- avoids losing reuse as requests finish
        |
        v
Production scheduler
        |
        +---- maximize throughput
        +---- respect TTFT/TPOT latency SLOs
        +---- respect KV-cache capacity
        +---- avoid memory overcommitment
```

This is the hardware-level reasoning behind one of the most important design decisions in modern LLM inference engines.

---

## 28. Summary

Batching increases throughput not because the GPU somehow becomes more powerful, but because the workload becomes more efficient at converting memory traffic into useful computation.

For weight-dominated transformer projections, a single decode token may require streaming large model matrices while performing relatively little work. When multiple token vectors are processed together, those same weights can participate in many more multiply-add operations. FLOPs therefore grow roughly with batch size while dominant model-weight traffic grows much more slowly.

That raises arithmetic intensity.

The Roofline model then explains the throughput consequence. In the memory-bound region, achievable FLOP/s is approximately arithmetic intensity multiplied by memory bandwidth. Raising arithmetic intensity therefore raises the performance ceiling until the workload approaches the ridge point. Beyond that region, compute capacity or another subsystem becomes the bottleneck and batching exhibits diminishing returns.

For LLM serving, the complete optimization problem is consequently not "use the largest possible batch." It is to maintain enough concurrent token work to obtain strong weight reuse and GPU efficiency while respecting KV-cache capacity, memory pressure, time-to-first-token, inter-token latency and tail-latency requirements.

That is why modern inference engines combine batching with continuous scheduling, paged KV-cache management, admission control, chunked prefill and increasingly disaggregated prefill/decode architectures.
