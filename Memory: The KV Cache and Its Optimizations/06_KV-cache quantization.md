# KV-Cache Quantization — From First Principles

KV-cache quantization is the next natural step after **GQA, PagedAttention, prefix caching, and sliding-window eviction**.

All of those techniques attack KV-cache cost in different ways. GQA reduces the number of KV heads. PagedAttention manages where KV tensors live. Prefix caching avoids recomputing shared prefixes. Sliding-window attention limits how many tokens need to remain active.

**KV-cache quantization attacks the size of each individual K and V value.**

The central idea is very simple:

> Instead of storing every K and V value using 16-bit floating point, store them using fewer bits such as 8-bit or 4-bit representations, while retaining enough numerical information to keep model quality acceptable.

The interesting part is understanding exactly how this works, why it saves so much memory, where the scaling factors come from, and why quantizing KV cache is not quite the same thing as quantizing model weights.

---

# 1. Start with the problem

From our previous example, consider a Llama-3-8B-style configuration:

```text
Layers       = 32
KV heads     = 8
Head dim     = 128
Precision    = BF16
```

The KV cache stores both:

```text
K = keys
V = values
```

Therefore the raw memory per token is:

```text
2 × 32 × 8 × 128 × 2 bytes
```

which is:

```text
131,072 bytes
```

or:

```text
128 KiB per token
```

So:

```text
8K tokens    ≈ 1 GiB
32K tokens   ≈ 4 GiB
64K tokens   ≈ 8 GiB
128K tokens  ≈ 16 GiB
```

Now imagine that instead of using BF16, which requires 16 bits per value, we store each KV value using 8 bits.

We have immediately cut the payload approximately in half.

With 4-bit storage, we theoretically cut it to one quarter.

That is the fundamental attraction.

---

# 2. What exactly are we quantizing?

It is important not to confuse three different things:

```text
Model weights
KV cache
Activations
```

Weight quantization means:

```text
W
```

is stored using fewer bits.

KV-cache quantization means:

```text
K
V
```

inside the inference cache are stored using fewer bits.

For example:

```text
BF16 KV cache:

K = [0.182, -0.731, 1.024, ...]
V = [0.052,  0.881, -0.312, ...]
```

Instead of storing these directly as BF16, we transform them into compact integer-like representations.

The model can later reconstruct approximate floating-point values when computing attention.

So the conceptual pipeline becomes:

```text
Original K/V
    ↓
Quantization
    ↓
Compact K/V
    ↓
GPU memory
    ↓
Dequantization
    ↓
Attention computation
```

The key word is **approximate**.

Quantization introduces numerical error.

The entire engineering problem is therefore:

> How much memory can we save without introducing enough error to noticeably hurt generation quality?

---

# 3. Why can't we simply convert BF16 to INT8?

Suppose we have:

```text
K = [-1.2, -0.5, 0.0, 0.8, 1.4]
```

An INT8 value can represent integers approximately between:

```text
-128 and +127
```

But our original numbers are floating-point values.

We need a mapping.

A very simple idea is:

```text
float value
      ↓
scale
      ↓
integer
```

Then later:

```text
integer
   ↓
scale
   ↓
approximate float
```

This is quantization.

---

# 4. The simplest symmetric quantization

Suppose we have:

```text
xmax = maximum absolute value
```

For example:

```text
values:

[-1.2, -0.5, 0.0, 0.8, 1.4]
```

The maximum absolute value is:

```text
1.4
```

For signed INT8, the largest positive integer magnitude is approximately:

```text
127
```

So we can define:

```text
scale = xmax / 127
```

Therefore:

```text
scale = 1.4 / 127
      ≈ 0.01102
```

Then each value is approximately represented as:

```text
q = round(x / scale)
```

So:

```text
1.4 / 0.01102 ≈ 127
0.8 / 0.01102 ≈ 73
-0.5 / 0.01102 ≈ -45
```

We store:

```text
[ -109, -45, 0, 73, 127 ]
```

instead of the original floating-point values.

---

# 5. Dequantization

When the attention kernel needs the original values, it can approximately reconstruct them.

The equation is simply:

```text
x ≈ q × scale
```

For example:

```text
q = 73
scale = 0.01102
```

Then:

```text
x ≈ 73 × 0.01102
  ≈ 0.805
```

The original value was:

```text
0.8
```

So we introduced a small error:

```text
0.805 - 0.8
= 0.005
```

That is the fundamental quantization tradeoff.

---

# 6. Where the memory savings come from

BF16:

```text
16 bits/value
```

INT8:

```text
8 bits/value
```

INT4:

```text
4 bits/value
```

Ignoring metadata for a moment:

```text
BF16 → 16 bits
INT8 → 8 bits
INT4 → 4 bits
```

Therefore:

```text
INT8 ≈ 2× smaller
INT4 ≈ 4× smaller
```

than BF16.

Return to our 128 KiB/token example.

BF16:

```text
128 KiB/token
```

INT8:

```text
≈ 64 KiB/token
```

INT4:

```text
≈ 32 KiB/token
```

Therefore at 128K tokens:

```text
BF16:
128K × 128 KiB
≈ 16 GiB
```

INT8:

```text
≈ 8 GiB
```

INT4:

```text
≈ 4 GiB
```

Again, these are raw payload estimates. Real implementations require scale metadata, alignment, block padding, temporary buffers, and kernel-specific overhead.

---

# 7. Why quantizing KV cache is particularly attractive

There is a very important asymmetry between weights and KV cache.

Model weights are relatively static.

KV cache is generated dynamically for every request.

Consider a model with:

```text
100 GB of weights
```

and:

```text
20 GB of KV cache
```

If you quantize the weights from FP16 to INT8, you may save a large amount of persistent model memory.

But if your inference server has many concurrent long-context requests, KV cache can become the dominant dynamic memory consumer.

For example:

```text
Request A → 8 GB KV
Request B → 6 GB KV
Request C → 10 GB KV
Request D → 5 GB KV
```

Suddenly:

```text
KV cache = 29 GB
```

Quantizing those caches can significantly increase concurrency.

---

# 8. But there is a major difference from weight quantization

Weight quantization changes parameters used repeatedly throughout inference.

KV-cache quantization affects the intermediate state used by attention.

The attention operation is approximately:

```text
Attention(Q, K, V)
```

with:

```text
scores = Q × Kᵀ
```

and then:

```text
output = softmax(scores) × V
```

If we quantize K, we affect:

```text
Q × Kᵀ
```

If we quantize V, we affect:

```text
softmax(QKᵀ) × V
```

Therefore K and V quantization can introduce errors directly into attention.

This makes the problem more sensitive than simply storing a generic tensor in fewer bits.

---

# 9. Why K can be especially sensitive

Let's examine:

```text
scores = Q × Kᵀ
```

Suppose:

```text
K = K_original + error
```

Then:

```text
Q × Kᵀ
```

becomes:

```text
Q × (K_original + error)ᵀ
```

which is:

```text
Q × K_originalᵀ
+
Q × errorᵀ
```

So quantization error in K directly changes the attention logits.

Then we apply:

```text
softmax()
```

which can amplify the consequences when attention scores are close to important decision boundaries.

For V, the error enters later:

```text
attention_weights × V
```

So K and V are not necessarily equally sensitive to quantization.

This is one reason practical KV-cache quantization schemes can treat K and V differently.

---

# 10. Per-tensor quantization is usually too crude

Imagine an entire KV tensor contains:

```text
mostly values between -0.1 and +0.1
```

but one value happens to be:

```text
8.0
```

If we calculate the scale using the global maximum:

```text
xmax = 8.0
```

then most of the tensor's numerical range is being represented very inefficiently.

We might instead divide the tensor into smaller groups.

For example:

```text
Tensor
 |
 +---- Group 0
 +---- Group 1
 +---- Group 2
 +---- Group 3
 ...
```

Each group gets its own scale.

This is called **group-wise quantization**.

---

# 11. Group-wise quantization

Suppose we have:

```text
values:

[-0.10, 0.04, 0.08, -0.07,
 1.20, 0.80, 1.10, 0.90]
```

If we use one scale for all eight values, the first four values occupy only a small fraction of the available integer range because the second group contains values around 1.2.

Instead:

```text
Group 0:

[-0.10, 0.04, 0.08, -0.07]

Group 1:

[1.20, 0.80, 1.10, 0.90]
```

we calculate separate scales.

That gives us much better numerical resolution.

The tradeoff is that we now need to store multiple scales.

So there is a classic tradeoff:

```text
smaller group
    ↓
better accuracy
    ↓
more scale metadata
```

versus:

```text
larger group
    ↓
less metadata
    ↓
potentially more quantization error
```

---

# 12. The scale itself consumes memory

This is an important detail that is often ignored in simple explanations.

Suppose:

```text
4096 KV values
```

are quantized to INT4.

The raw values require:

```text
4096 × 4 bits
= 16384 bits
= 2048 bytes
```

So:

```text
2 KiB
```

But suppose we use one FP16 scale per 64 values.

Then:

```text
4096 / 64
= 64 groups
```

and each scale requires:

```text
2 bytes
```

so scale metadata requires:

```text
64 × 2
= 128 bytes
```

Total:

```text
2048 + 128
= 2176 bytes
```

instead of 2048 bytes.

The metadata is small compared with the raw values, but it is not zero.

---

# 13. Quantization granularity

There are several possible granularities.

You could have:

```text
one scale for the entire tensor
```

or:

```text
one scale per layer
```

or:

```text
one scale per KV head
```

or:

```text
one scale per channel
```

or:

```text
one scale per group of values
```

or:

```text
one scale per token/group
```

Each choice changes the accuracy, metadata, and kernel complexity.

The general principle is:

```text
finer granularity
        ↓
better adaptation to value distributions
        ↓
more metadata and computation
```

---

# 14. INT8 versus INT4

INT8 is relatively comfortable.

You have:

```text
8 bits
```

per value and therefore:

```text
256
```

possible integer states.

For signed INT8:

```text
-128 ... +127
```

INT4 has only:

```text
16
```

possible states.

For signed 4-bit representation, the effective range is tiny compared with FP16/BF16.

This means INT4 has much coarser quantization.

So:

```text
BF16
   ↓
INT8
```

usually introduces less numerical distortion than:

```text
BF16
   ↓
INT4
```

but INT4 provides much larger memory savings.

---

# 15. FP8 is another important option

Not all low-precision KV caches have to use integer formats.

Modern accelerators increasingly support low-precision floating-point formats such as:

```text
FP8
```

The advantage is that floating-point representations preserve a notion of dynamic range.

Conceptually:

```text
BF16
16 bits

FP8
8 bits
```

So FP8 can approximately halve KV storage while retaining floating-point-style behavior.

There are different FP8 formats and implementation details, but the important conceptual distinction is:

```text
INT8
```

typically uses an explicit scale to map floating values into an integer range.

Whereas:

```text
FP8
```

itself encodes a floating-point-like value with a smaller exponent/mantissa representation.

---

# 16. Why FP8 can be attractive for KV cache

Attention involves operations such as:

```text
Q × Kᵀ
```

and:

```text
attention × V
```

which are highly optimized on modern GPUs.

If the hardware provides efficient FP8 load/compute paths, storing KV in FP8 can reduce:

```text
memory footprint
memory bandwidth
GPU memory traffic
```

while avoiding some of the complications of integer dequantization.

However, the actual benefit depends heavily on:

```text
GPU architecture
attention kernel
framework
quantization scheme
scale handling
batch size
sequence length
```

So "FP8 is faster" is not universally true. The hardware and kernel implementation matter.

---

# 17. Why KV quantization can improve throughput

This is not just about fitting more sequences into VRAM.

During decode, the model repeatedly reads the KV cache.

Suppose:

```text
sequence length = 64K
```

Every newly generated token needs to interact with a large amount of previous K/V data.

This means the decoder can become heavily dependent on memory bandwidth.

If KV cache size is reduced:

```text
BF16
   ↓
FP8 / INT8
```

the GPU has less data to read from memory.

So quantization can potentially improve:

```text
memory bandwidth utilization
```

as well as:

```text
memory capacity
```

This is particularly relevant during long-context decoding.

---

# 18. Prefill versus decode

This distinction matters a lot.

During **prefill**, the model processes many input tokens together.

During **decode**, the model typically generates one new token at a time while reading the accumulated KV cache.

For a long sequence:

```text
Decode step:

Q_new
   |
   +---- K0
   +---- K1
   +---- K2
   +---- ...
   +---- K65000
```

The model is reading a large KV cache for each generated token.

So reducing the KV representation can reduce memory traffic during decoding.

This is one reason KV-cache quantization is particularly interesting for:

```text
long-context
low-batch
decode-heavy
```

workloads.

---

# 19. Quantization happens after K/V generation

Conceptually, consider one transformer layer.

Normally:

```text
Hidden state
    ↓
K projection
    ↓
K
    ↓
KV cache
```

and:

```text
Hidden state
    ↓
V projection
    ↓
V
    ↓
KV cache
```

With KV quantization:

```text
Hidden state
    ↓
K projection
    ↓
K in higher precision
    ↓
Quantize
    ↓
Compressed K cache
```

and:

```text
Hidden state
    ↓
V projection
    ↓
V in higher precision
    ↓
Quantize
    ↓
Compressed V cache
```

During attention:

```text
Compressed K
      ↓
Dequantize / low-precision processing
      ↓
Attention

Compressed V
      ↓
Dequantize / low-precision processing
      ↓
Attention output
```

The exact implementation varies by framework and kernel.

---

# 20. A concrete INT8 calculation

Let's take one small vector:

```text
K = [0.10, -0.40, 0.70, 1.00]
```

Maximum absolute value:

```text
1.00
```

Using symmetric INT8:

```text
scale = 1 / 127
      ≈ 0.007874
```

Quantization:

```text
0.10 / 0.007874 ≈ 13
-0.40 / 0.007874 ≈ -51
0.70 / 0.007874 ≈ 89
1.00 / 0.007874 = 127
```

So we store:

```text
[13, -51, 89, 127]
```

Dequantization:

```text
13 × 0.007874 ≈ 0.1024

-51 × 0.007874 ≈ -0.4016

89 × 0.007874 ≈ 0.7008

127 × 0.007874 = 1.0
```

Original:

```text
[0.10, -0.40, 0.70, 1.00]
```

Reconstructed:

```text
[0.1024, -0.4016, 0.7008, 1.0000]
```

The values are close, but not identical.

That difference is the price we pay for compression.

---

# 21. What happens inside attention?

Suppose:

```text
Q = [0.2, 0.5, -0.1, 0.3]
```

and the original:

```text
K = [0.10, -0.40, 0.70, 1.00]
```

The attention score component is:

```text
Q · K
```

which is:

```text
0.2×0.10
+
0.5×(-0.40)
+
(-0.1)×0.70
+
0.3×1.00
```

giving:

```text
0.02 - 0.20 - 0.07 + 0.30
= 0.05
```

Now use the reconstructed quantized K:

```text
[0.1024, -0.4016, 0.7008, 1.0]
```

The score becomes approximately:

```text
0.02048
- 0.2008
- 0.07008
+ 0.30
```

which is:

```text
≈ 0.0496
```

instead of:

```text
0.05
```

So the attention score changed slightly.

In a real transformer, this happens across:

```text
many layers
many heads
many tokens
```

which is why quantization error must be carefully controlled.

---

# 22. Quantization error does not necessarily accumulate catastrophically

It might seem that if every layer introduces a small error, the final answer must become terrible.

That isn't necessarily true.

Neural networks often have considerable tolerance to small numerical perturbations.

The question is not:

> Is the quantized KV exactly equal to the original KV?

It isn't.

The practical question is:

> Does the resulting model behavior remain sufficiently close to the original model?

This is evaluated using:

```text
perplexity
long-context benchmarks
generation quality
task accuracy
retrieval accuracy
reasoning benchmarks
human evaluation
```

The acceptable error depends on the application.

---

# 23. Why 4-bit KV is much harder

Suppose values are:

```text
[-0.95, -0.30, 0.10, 0.45, 0.92]
```

With 8-bit quantization, we have many possible representational states.

With 4-bit quantization, we have only a small number.

So the distance between representable values becomes much larger.

That means:

```text
BF16
 ↓
INT8
```

can often preserve the distribution relatively well.

But:

```text
BF16
 ↓
INT4
```

requires more careful choices of:

```text
group size
scaling
outlier handling
K/V treatment
calibration
kernel implementation
```

This is why 4-bit KV-cache quantization is considerably more technically interesting than simply "divide the memory by four."

---

# 24. Outliers are a major problem

Suppose a group contains:

```text
[0.01, 0.02, 0.03, 8.0]
```

The value:

```text
8.0
```

is an outlier.

If we calculate one scale based on 8.0, the small values:

```text
0.01
0.02
0.03
```

may receive very poor resolution.

This leads to an important design problem:

```text
How should we handle outliers?
```

Possible approaches include:

```text
smaller quantization groups
different scaling strategies
separate treatment of outliers
higher precision for selected dimensions
mixed-precision KV storage
```

This is one reason real KV quantization implementations are substantially more complicated than the basic INT8 equation.

---

# 25. K and V may use different strategies

Remember:

```text
K → affects QKᵀ
V → affects attention output
```

Because their errors enter different parts of the computation, an implementation may choose different precision or scaling strategies.

For example, conceptually:

```text
K → FP8
V → INT8
```

or:

```text
K → INT8
V → INT4
```

or:

```text
K and V → same precision
```

The appropriate choice depends on the model and implementation.

There is no universal rule saying:

> "Always quantize K and V identically."

---

# 26. Quantization and GQA multiply their benefits

Now let's combine what we have learned.

Suppose MHA uses:

```text
32 KV heads
```

and GQA uses:

```text
8 KV heads
```

GQA gives:

```text
32 / 8
= 4×
```

less KV payload.

Now suppose we additionally move:

```text
BF16 → INT8
```

which approximately gives:

```text
2×
```

compression.

Together:

```text
4 × 2
= 8×
```

reduction relative to the original MHA BF16 KV cache, ignoring metadata and implementation overhead.

Starting with:

```text
512 KiB/token
```

for MHA BF16:

```text
GQA:
512 / 4
= 128 KiB/token
```

Then INT8:

```text
128 / 2
= 64 KiB/token
```

Then with a 4096-token window:

```text
4096 × 64 KiB
= 256 MiB
```

So we have transformed:

```text
MHA + BF16 + 128K full context
≈ 64 GiB
```

into something conceptually closer to:

```text
GQA + INT8 + 4096 active window
≈ 256 MiB
```

for the raw active KV payload.

That's a dramatic difference.

But notice what happened: **three different optimizations attacked three different dimensions of the problem.**

---

# 27. The full optimization stack

You can now think of KV-cache optimization as a stack:

```text
                    KV CACHE
                       |
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
     GQA          Quantization       Eviction
       |               |                |
 fewer KV heads    fewer bits       fewer tokens
       |               |                |
       └───────────────┼────────────────┘
                       ↓
                smaller KV memory
                       |
                       ↓
                 PagedAttention
                       |
                       ↓
             efficient block management
                       |
                       ↓
                Prefix Caching
                       |
                       ↓
             reuse shared KV blocks
```

This is the bigger picture of inference memory engineering.

---

# 28. Quantization versus sliding window

These two techniques solve different problems.

Suppose:

```text
KV cache = 16 GiB
```

and you use INT8.

You get approximately:

```text
8 GiB
```

But the cache still grows with context length.

With sliding window:

```text
16 GiB → bounded amount
```

but each retained value still consumes BF16.

So:

```text
Quantization:
reduce bytes per token
```

while:

```text
Sliding window:
reduce number of retained tokens
```

Together:

```text
memory
=
bytes per KV value
×
number of retained KV values
```

Both terms can be optimized independently.

---

# 29. Quantization versus PagedAttention

These are also fundamentally different.

Quantization answers:

> How many bytes does each KV value require?

PagedAttention answers:

> How should those KV values be physically allocated and accessed?

For example:

```text
BF16 + PagedAttention
```

still stores BF16 values.

And:

```text
INT8 without efficient memory management
```

can still suffer from allocation and fragmentation problems.

Therefore:

```text
Quantization
+
PagedAttention
```

are complementary.

---

# 30. Quantization versus prefix caching

Prefix caching reduces computation.

Suppose ten users share:

```text
8K-token system prompt
```

Prefix caching allows the server to reuse the KV state.

Quantization determines how much memory those cached KV blocks consume.

So:

```text
Prefix caching
    ↓
reuse KV
```

while:

```text
KV quantization
    ↓
compress KV
```

A system can use both.

---

# 31. Quantization and prefix-cache correctness

There is an additional subtle issue.

Suppose a prefix cache contains quantized KV blocks.

The cache key cannot simply mean:

```text
"these tokens exist"
```

The reusable state also depends on the configuration under which it was generated.

Conceptually, a robust cache namespace may depend on:

```text
model identity
model version
tokenizer/version
attention configuration
KV precision
quantization scheme
quantization parameters
relevant runtime configuration
tenant/security boundary
```

Why?

Because a KV representation generated under one model/configuration should not accidentally be reused under an incompatible configuration.

This is particularly important in multi-tenant inference infrastructure.

---

# 32. Quantization and PagedAttention blocks

Imagine:

```text
block size = 16 tokens
```

and:

```text
BF16 KV block = 2 MiB
```

from our earlier example.

With INT8:

```text
≈ 1 MiB
```

per block.

With INT4:

```text
≈ 512 KiB
```

ignoring scale metadata and alignment.

So the PagedAttention block allocator can conceptually operate on much smaller physical blocks.

That means the GPU memory pool can hold more active logical KV blocks.

---

# 33. But smaller KV doesn't automatically mean faster inference

This is an important engineering caveat.

Suppose you compress KV to INT4.

You now have:

```text
4× less data
```

but the GPU must perform:

```text
dequantization
scaling
possibly extra memory operations
```

If the attention kernel is poorly implemented, the additional computation may cancel out some of the memory-bandwidth benefit.

Therefore the real performance equation is more like:

```text
performance benefit
=
memory traffic saved
-
quantization/dequantization overhead
-
kernel inefficiency
```

This is why optimized CUDA/Triton attention kernels matter so much.

A theoretically smaller representation is not automatically a faster production implementation.

---

# 34. The decode bottleneck

For long-context decoding, consider:

```text
one new token
```

The model generates:

```text
Q_new
```

and then needs to read:

```text
K0 ... KN
V0 ... VN
```

from the cache.

If N is very large, this can become memory-bandwidth dominated.

Therefore:

```text
BF16 KV
```

might require reading a large amount of data.

With:

```text
INT8 KV
```

the amount of data transferred can approximately halve.

With:

```text
INT4 KV
```

it can theoretically become approximately one quarter.

This is why KV quantization is particularly interesting for long-context decode workloads.

---

# 35. A useful mental model

Think of the KV cache as a giant table.

BF16:

```text
┌──────────────┐
│ 16-bit value │
├──────────────┤
│ 16-bit value │
├──────────────┤
│ 16-bit value │
└──────────────┘
```

INT8:

```text
┌─────────────┐
│ 8-bit value │
├─────────────┤
│ 8-bit value │
├─────────────┤
│ 8-bit value │
└─────────────┘
```

INT4:

```text
┌─────────────┐
│ 4-bit value │
├─────────────┤
│ 4-bit value │
├─────────────┤
│ 4-bit value │
└─────────────┘
```

But each compressed representation needs enough metadata to understand how the integer maps back to the original floating-point range.

So the real representation is closer to:

```text
quantized values
+
scales
+
possibly zero points
+
alignment/metadata
```

---

# 36. A complete inference path

Let's put everything together.

Suppose a user sends:

```text
"Explain PagedAttention."
```

The transformer generates K and V for the prompt.

Normally:

```text
Prompt
  ↓
Transformer
  ↓
K/V
  ↓
BF16 KV cache
```

With quantization:

```text
Prompt
  ↓
Transformer
  ↓
K/V
  ↓
Quantization
  ↓
INT8/FP8 KV cache
```

With PagedAttention:

```text
Quantized K/V
      ↓
KV blocks
      ↓
Physical GPU block pool
```

With prefix caching:

```text
Existing matching prefix
      ↓
Reuse existing KV blocks
```

With sliding window:

```text
Only active window remains resident
      ↓
Old blocks become evictable
```

So a modern inference engine can conceptually look like:

```text
                       User Request
                            |
                            ↓
                         Tokens
                            |
                            ↓
                         Prefill
                            |
                            ↓
                       K/V tensors
                            |
                            ↓
                    KV Quantization
                            |
                            ↓
                    Paged KV Blocks
                            |
               ┌────────────┴────────────┐
               ↓                         ↓
       Prefix-cache reuse         Active window
               ↓                         ↓
               └────────────┬────────────┘
                            ↓
                       KV Scheduler
                            |
                     ┌──────┴──────┐
                     ↓             ↓
                  retain          evict
                     |
                     ↓
                    Decode
                     |
                     ↓
                  New token
                     |
                     ↓
                 repeat
```

That is the complete conceptual lifecycle.

---

# 37. The four major KV-cache compression dimensions

At this point, the KV-cache problem can be decomposed very cleanly.

### Head dimension

GQA/MQA:

```text
Reduce number of KV heads
```

### Precision dimension

KV quantization:

```text
Reduce bits per KV value
```

### Sequence dimension

Sliding-window/eviction:

```text
Reduce number of retained tokens
```

### Allocation dimension

PagedAttention:

```text
Reduce memory-management waste
```

And prefix caching attacks computation:

```text
Reuse previously computed KV state
```

This is a much more useful way to understand modern inference optimization than memorizing individual techniques.

---

# 38. The master equation

For raw KV memory, you can mentally reduce almost everything to:

```text
KV memory
≈
number of retained tokens
×
number of layers
×
number of KV heads
×
head dimension
×
2
×
bytes per value
```

Now every optimization has a clear place.

**GQA** changes:

```text
number of KV heads
```

**Sliding window** changes:

```text
number of retained tokens
```

**KV quantization** changes:

```text
bytes per value
```

**PagedAttention** reduces the overhead around allocation and fragmentation.

**Prefix caching** reduces the amount of KV computation that has to be repeated.

This single equation is one of the most useful mental models for inference engineering.

---

# 39. What happens when everything is combined?

Using our earlier example:

```text
Model:
32 layers
8 KV heads
128 head dimension
```

BF16:

```text
2 bytes/value
```

Raw KV cost:

```text
128 KiB/token
```

Now add INT8:

```text
≈ 64 KiB/token
```

Now add a 4096-token sliding window:

```text
4096 × 64 KiB
≈ 256 MiB
```

Now add PagedAttention:

```text
256 MiB worth of logical KV
→ managed as physical blocks
```

Now add prefix caching:

```text
shared prefix
→ existing blocks reused
```

Now add continuous batching:

```text
many requests
→ scheduler dynamically shares GPU compute
```

You end up with a serving architecture that can support substantially more concurrent long-context workloads than a naive:

```text
full-context
BF16
MHA
contiguous KV
no caching
```

implementation.

---

# 40. The most important tradeoff

The entire topic can ultimately be summarized as one engineering triangle:

```text
                Quality
                  /\
                 /  \
                /    \
               /      \
              /        \
             /          \
            /____________\
       Memory            Speed
```

Aggressive KV quantization:

```text
↓ memory
potentially ↑ throughput
but potentially ↓ numerical fidelity
```

Aggressive eviction:

```text
↓ memory
↑ concurrency
but ↓ direct long-range context
```

Larger precision:

```text
↑ memory
potentially ↑ numerical fidelity
```

Larger context:

```text
↑ memory
potentially ↑ long-range capability
```

There is no magic compression technique that simultaneously makes all dimensions better.

The engineering task is to find the point that matches the workload.

---

# 41. Where KV-cache quantization fits in your inference-engineering roadmap

You have now gone through a very logical progression:

```text
1. Inference Lifecycle
          ↓
2. Attention Math
          ↓
3. KV Cache
          ↓
4. MHA / MQA / GQA
          ↓
5. PagedAttention
          ↓
6. Prefix Caching
          ↓
7. Sliding Window / Cache Eviction
          ↓
8. KV-Cache Quantization
```

The next major pieces naturally become:

```text
Continuous Batching
        ↓
Chunked Prefill
        ↓
Prefill vs Decode Scheduling
        ↓
Speculative Decoding
        ↓
KV Cache Offloading
        ↓
Disaggregated Prefill/Decode
        ↓
Tensor Parallelism
        ↓
Pipeline Parallelism
        ↓
Expert Parallelism
        ↓
Inference Serving Architecture
```

The important connection is that **KV-cache quantization doesn't replace the previous techniques**. It joins them.

A modern LLM inference engine is essentially trying to answer four questions continuously:

```text
What KV data do I need?
        ↓
Sliding window / eviction

How many KV values do I need?
        ↓
GQA / MQA

How many bytes should each value consume?
        ↓
KV quantization

Where should those values physically live?
        ↓
PagedAttention
```

And prefix caching answers a fifth question:

```text
Have I already computed this KV state?
        ↓
Prefix caching
```

That is the core mental model behind modern KV-cache engineering.
