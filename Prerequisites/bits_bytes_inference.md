
The key is to understand the chain:

**bits → numbers → floating point → memory → tensors → matrix multiplication → GPU → Transformer → attention/KV cache → inference phases → quantization → memory/compute bottlenecks → optimization techniques.**

Below is the consolidated foundation you need. Once you understand this, papers and systems such as FlashAttention, FP8, INT4, KV-cache compression, speculative decoding, GQA/MQA, CUDA kernels, and modern inference engines become much easier to reason about.

---

# 1. Start at the absolute bottom: Bits

A **bit** is the smallest unit of digital information.

It has two possible values:

```text
0
1
```

You can think of it as an electrical switch being in one of two states.

With multiple bits, the number of possible combinations grows exponentially:

```text
1 bit  → 2¹ = 2 values
2 bits → 2² = 4 values
4 bits → 2⁴ = 16 values
8 bits → 2⁸ = 256 values
16 bits → 2¹⁶ = 65,536 values
32 bits → 2³² possible bit patterns
```

This equation is one of the most important equations in the entire subject:

```text
number of possible representations = 2^bits
```

It is the foundation of quantization.

If I give you 4 bits, you physically have only 16 possible patterns.

If I give you 8 bits, you have 256.

If I give you 16 bits, you have 65,536.

The computer cannot magically represent more distinct bit patterns than this.

---

# 2. Bytes

Eight bits make one byte.

```text
1 byte = 8 bits
```

Therefore:

```text
1 KB ≈ 1,024 bytes
1 MB ≈ 1,024 KB
1 GB ≈ 1,024 MB
```

More technically, binary units are:

```text
1 KiB = 1,024 bytes
1 MiB = 1,024 KiB
1 GiB = 1,024 MiB
```

Why should you care?

Because **model size is fundamentally a memory calculation**.

If a model has:

```text
7 billion parameters
```

and every parameter takes 4 bytes:

```text
7,000,000,000 × 4
= 28,000,000,000 bytes
≈ 28 GB
```

That's already enough to understand why a 7B FP32 model requires roughly 28 GB just for weights.

---

# 3. Binary numbers

Computers naturally operate using binary.

For example:

```text
1011₂
```

means:

```text
1×2³ + 0×2² + 1×2¹ + 1×2⁰
```

which is:

```text
8 + 0 + 2 + 1 = 11
```

Therefore:

```text
1011₂ = 11₁₀
```

The same idea works for fractions.

For example:

```text
0.101₂
```

means:

```text
1×2⁻¹ + 0×2⁻² + 1×2⁻³
```

which is:

```text
0.5 + 0 + 0.125
= 0.625
```

So:

```text
0.101₂ = 0.625₁₀
```

This becomes important because floating-point numbers are fundamentally **binary scientific notation**.

---

# 4. Hexadecimal

You will constantly encounter hexadecimal when working close to hardware.

Hexadecimal has 16 symbols:

```text
0 1 2 3 4 5 6 7 8 9 A B C D E F
```

One hexadecimal digit represents exactly four bits.

For example:

```text
1111₂ = F₁₆
1010₂ = A₁₆
```

Therefore:

```text
8 bits = 2 hexadecimal digits
```

For example:

```text
10110101
```

can be grouped:

```text
1011 0101
```

and converted to:

```text
B5
```

So:

```text
10110101₂ = B5₁₆
```

Hexadecimal becomes useful when looking at memory addresses, binary representations, CUDA debugging, low-level systems, and numerical formats.

---

# 5. Integers

An unsigned 8-bit integer is called:

```text
uint8
```

It can represent:

```text
0 → 255
```

because:

```text
2⁸ = 256
```

possible values exist.

A signed 8-bit integer is usually:

```text
int8
```

and commonly uses two's complement:

```text
-128 → 127
```

Again, there are 256 total representations.

This is important for quantization because:

```text
INT8
```

does not mean "a smaller floating-point number."

It means:

> An integer represented using eight bits.

That distinction is critical.

---

# 6. Two's complement

Computers need a way to represent negative integers.

For an 8-bit signed integer:

```text
00000000 = 0
00000001 = 1
...
01111111 = 127
```

The negative numbers occupy the remaining representations:

```text
10000000 = -128
11111111 = -1
```

This is called **two's complement**.

You don't need to memorize every conversion, but you should understand why:

```text
int8 = -128 ... 127
```

while:

```text
uint8 = 0 ... 255
```

This becomes important when we compare INT8 with FP8.

---

# 7. Why integers aren't enough

Neural networks contain values such as:

```text
0.132
-0.827
1.392
0.000023
-14.72
```

Integers aren't sufficient.

We therefore need a representation capable of efficiently representing both very large and very small values.

That's where floating point comes in.

---

# 8. Scientific notation

Consider:

```text
602000000000000000000000
```

We can write:

```text
6.02 × 10²³
```

We have separated the number into:

```text
sign
significand
exponent
```

Floating point uses the same fundamental concept but with base 2:

```text
value ≈ sign × significand × 2^exponent
```

That is the fundamental mental model for FP32, FP16, BF16 and FP8.

---

# 9. Floating-point representation

A floating-point number conceptually looks like:

```text
┌────────┬──────────┬───────────────┐
│ Sign   │ Exponent │ Fraction      │
└────────┴──────────┴───────────────┘
```

The sign answers:

```text
positive or negative?
```

The exponent answers:

```text
how large/small is the number?
```

The fraction answers:

```text
how precisely can we represent it?
```

This gives us the most important trade-off in floating point:

```text
Exponent bits → range

Fraction bits → precision
```

Remember this.

It explains almost everything about numerical formats.

---

# 10. IEEE 754

IEEE 754 is the major standard governing floating-point representations and arithmetic.

For FP32:

```text
1 sign bit
8 exponent bits
23 fraction bits
```

Total:

```text
1 + 8 + 23 = 32 bits
```

Conceptually:

```text
┌─────┬──────────┬───────────────────────┐
│  S  │    E     │           F           │
│ 1   │    8     │          23           │
└─────┴──────────┴───────────────────────┘
```

For normal numbers, the conceptual formula is:

```text
value =
(-1)^S × (1 + fraction) × 2^(E-bias)
```

For FP32:

```text
bias = 127
```

So:

```text
actual exponent = stored exponent - 127
```

---

# 11. FP32

FP32 means:

```text
32-bit floating point
```

Each value consumes:

```text
32 bits = 4 bytes
```

FP32 has approximately:

```text
24 bits of significand precision
```

and an enormous numerical range.

It is the traditional high-precision floating-point format used extensively in numerical computing.

For a model with 70B parameters:

```text
70B × 4 bytes
≈ 280 GB
```

This is why storing very large models entirely in FP32 is expensive.

---

# 12. FP16

FP16 means:

```text
16-bit floating point
```

Each value requires:

```text
16 bits = 2 bytes
```

So compared with FP32:

```text
FP32 = 4 bytes
FP16 = 2 bytes
```

A 70B model would therefore require approximately:

```text
70B × 2
≈ 140 GB
```

for the weights.

You immediately get approximately:

```text
50% memory reduction
```

compared with FP32.

---

# 13. BF16

BF16 means:

```text
Brain Floating Point 16
```

It also uses:

```text
16 bits
```

but allocates the bits differently.

The simplified comparison is:

```text
             Sign   Exponent   Fraction

FP32           1        8         23
FP16           1        5         10
BF16           1        8          7
```

This is extremely important.

FP16 has:

```text
5 exponent bits
```

while BF16 has:

```text
8 exponent bits
```

Therefore BF16 has a numerical range much closer to FP32.

But BF16 sacrifices fraction bits.

So:

```text
FP16 → more precision, smaller range

BF16 → less precision, much larger range
```

For many deep-learning workloads, BF16 is attractive because neural-network values can span a wide numerical range.

---

# 14. FP8

FP8 reduces the representation even further.

It uses:

```text
8 bits
```

Common formats include:

```text
E4M3
E5M2
```

The notation means:

```text
E4M3:

1 sign
4 exponent
3 fraction
```

while:

```text
E5M2:

1 sign
5 exponent
2 fraction
```

Again:

```text
more exponent bits → range

more fraction bits → precision
```

FP8 is particularly important in modern AI accelerators because neural-network computation can often tolerate lower precision when the computation is carefully designed.

---

# 15. The format hierarchy

You can now understand this:

```text
FP32
  ↓
FP16 / BF16
  ↓
FP8
  ↓
INT8
  ↓
INT4
```

But don't interpret this as:

```text
lower = universally better
```

It is a trade-off.

Reducing precision can provide:

```text
less memory
less memory bandwidth
higher throughput
potentially better hardware utilization
```

but can introduce:

```text
rounding error
quantization error
overflow/underflow issues
accuracy degradation
```

Inference engineering is largely about managing these trade-offs.

---

# 16. Precision vs range

This deserves special attention.

Suppose I tell you that a format can represent:

```text
10^-30 → 10^30
```

That tells you about **range**.

It doesn't tell you how accurately it can distinguish:

```text
1.00000
1.00001
1.00002
```

That's **precision**.

Therefore:

```text
Range
=
how far can I go?

Precision
=
how finely can I distinguish values?
```

Exponent bits primarily control range.

Fraction bits primarily control precision.

---

# 17. Rounding error

Floating-point numbers have finite precision.

Therefore many decimal numbers cannot be represented exactly.

For example:

```python
0.1 + 0.2
```

may produce:

```text
0.30000000000000004
```

rather than exactly:

```text
0.3
```

because `0.1` and `0.2` don't have finite binary representations.

This is normal.

Machine learning systems perform enormous numbers of numerical operations, so understanding numerical error is essential.

---

# 18. Machine epsilon

Another important concept is **machine epsilon**.

It roughly describes the spacing around 1 between representable floating-point numbers.

For a format with `p` bits of significand precision, the spacing behaves roughly like:

```text
2^(-p)
```

The exact definition depends on convention, but the important intuition is:

```text
more precision bits
→ smaller representational gaps
→ smaller rounding error
```

This explains why FP32 can distinguish values more finely than FP16 or BF16.

---

# 19. NaN and Infinity

Floating-point formats also have special representations.

You will encounter:

```text
+∞
-∞
NaN
```

where:

```text
NaN = Not a Number
```

For example:

```python
x = float("inf")
y = float("nan")
```

In deep-learning systems, NaNs can appear due to numerical instability such as:

```text
overflow
division by zero
invalid operations
unstable exponentials
bad normalization
```

This becomes especially relevant when using lower-precision arithmetic.

---

# 20. Overflow and underflow

Overflow happens when:

```text
result > maximum representable value
```

Underflow happens when:

```text
result is too small to represent normally
```

This is why numerical range matters.

If you move:

```text
FP32 → FP16
```

you aren't simply making the number smaller in storage.

You are changing its numerical behavior.

---

# 21. Now move from numbers to tensors

A neural network doesn't usually manipulate individual numbers one at a time.

It manipulates **tensors**.

A tensor is essentially a multidimensional array.

For example:

```text
scalar
```

is a:

```text
0-dimensional tensor
```

A vector:

```text
[1, 2, 3]
```

is a:

```text
1D tensor
```

A matrix:

```text
[1 2 3]
[4 5 6]
```

is a:

```text
2D tensor
```

A batch of images could be:

```text
4D tensor
```

For example:

```text
[B, C, H, W]
```

where:

```text
B = batch
C = channels
H = height
W = width
```

For a Transformer, we commonly see:

```text
[batch, sequence_length, hidden_dimension]
```

---

# 22. Matrix multiplication

The fundamental operation inside neural networks is often matrix multiplication.

Suppose:

```text
X = [1 2]
```

and:

```text
W =
[3]
[4]
```

Then:

```text
XW
=
1×3 + 2×4
=
11
```

For larger matrices, the same operation is repeated many times.

This is called:

```text
GEMM
```

which means:

```text
General Matrix-Matrix Multiplication
```

A huge amount of neural-network computation ultimately reduces to variants of GEMM.

---

# 23. Dot product

A dot product is:

```text
a · b
=
a₁b₁ + a₂b₂ + ... + aₙbₙ
```

For example:

```text
[1, 2, 3] · [4, 5, 6]
```

becomes:

```text
1×4 + 2×5 + 3×6
```

which is:

```text
4 + 10 + 18 = 32
```

Attention mechanisms use enormous numbers of dot products.

---

# 24. FLOPs

You will frequently encounter:

```text
FLOP
```

which means:

```text
Floating Point Operation
```

and:

```text
FLOPS
```

which means:

```text
Floating Point Operations Per Second
```

If a system performs:

```text
10¹² operations/second
```

that's approximately:

```text
1 TFLOPS
```

Modern GPUs can provide enormous theoretical compute throughput.

But here's the important part:

> **Peak FLOPS does not necessarily equal real-world inference performance.**

Why?

Because computation is only one part of the problem.

---

# 25. Memory is often the real bottleneck

Imagine a GPU has extremely powerful compute units.

You still need to get the data into those compute units.

The simplified process is:

```text
Memory
   ↓
Load weights
   ↓
Compute
   ↓
Store results
```

If moving data takes longer than performing the arithmetic, the GPU spends time waiting for memory.

This creates the fundamental distinction:

```text
compute-bound
```

versus:

```text
memory-bound
```

---

# 26. Memory hierarchy

Modern systems have multiple levels of storage.

Conceptually:

```text
Registers
    ↓
L1 cache
    ↓
L2 cache
    ↓
GPU SRAM / shared memory
    ↓
HBM / VRAM
    ↓
System RAM
    ↓
SSD
```

Generally:

```text
closer to compute
→ faster
→ smaller

farther from compute
→ slower
→ larger
```

Therefore inference optimization is heavily concerned with **data movement**.

---

# 27. Memory bandwidth

Memory bandwidth describes how much data can be moved per unit time.

For example:

```text
1 TB/s
```

means roughly:

```text
1 trillion bytes/second
```

If your workload repeatedly needs to read enormous model weights from HBM, memory bandwidth becomes critical.

This explains an important phenomenon:

> Making the arithmetic faster doesn't help much if the GPU is waiting for data.

This is why quantization can improve performance even when the mathematical operation itself isn't fundamentally cheaper.

Smaller numbers mean:

```text
less data
↓
less memory traffic
↓
better bandwidth utilization
↓
potentially faster inference
```

---

# 28. Arithmetic intensity

A useful concept is:

```text
arithmetic intensity
=
operations performed / bytes moved
```

If you perform lots of computation while moving relatively little data:

```text
high arithmetic intensity
```

If you move enormous amounts of data for relatively little computation:

```text
low arithmetic intensity
```

This helps us understand whether a workload is likely to be:

```text
compute-bound
```

or:

```text
memory-bound
```

This eventually leads to the **roofline model**, which is one of the important concepts in performance engineering.

---

# 29. GPU fundamentals

A GPU contains many parallel execution resources.

A simplified NVIDIA-style mental model is:

```text
GPU
│
├── Streaming Multiprocessors
│
│   ├── CUDA cores
│   ├── Tensor cores
│   ├── Registers
│   └── Shared memory
│
├── L2 cache
│
└── HBM / VRAM
```

You don't need to memorize hardware diagrams yet.

Understand the fundamental idea:

```text
CPU:
few powerful general-purpose cores

GPU:
many highly parallel compute resources
```

Neural networks are highly parallel, so GPUs are exceptionally suitable for them.

---

# 30. CUDA threads and warps

On NVIDIA GPUs, threads execute in groups called:

```text
warps
```

A typical NVIDIA warp contains:

```text
32 threads
```

These threads execute instructions in a coordinated fashion.

This is part of the reason GPU programs are written differently from ordinary CPU programs.

Eventually, you'll encounter:

```text
CUDA kernels
warps
blocks
threads
SMs
occupancy
shared memory
register pressure
memory coalescing
```

These become important when optimizing custom inference kernels.

---

# 31. Tensor Cores

Modern NVIDIA GPUs include specialized hardware called:

```text
Tensor Cores
```

These are designed for high-throughput matrix operations.

Conceptually:

```text
matrix multiplication
        ↓
Tensor Core
        ↓
very high throughput
```

They support various numerical formats depending on the GPU architecture, including formats such as:

```text
FP16
BF16
TF32
FP8
INT8
```

This is one reason numerical format selection matters at the hardware level.

---

# 32. Now enter the Transformer

Now everything we've learned starts connecting.

An LLM is generally based on a Transformer architecture.

The simplified flow is:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer layers
 ↓
Logits
 ↓
Next-token selection
```

Each Transformer layer contains major components such as:

```text
Self-attention
MLP
Normalization
Residual connections
```

And these components perform huge amounts of tensor computation.

---

# 33. Tokenization

Suppose we have:

```text
"Hello world"
```

The tokenizer converts it into token IDs.

Conceptually:

```text
Hello → 15496
world → 995
```

The actual IDs depend on the tokenizer.

The important point is:

```text
text
↓
integers
```

Then:

```text
integers
↓
embedding vectors
```

---

# 34. Embeddings

Suppose the model has:

```text
vocabulary = 100,000 tokens
hidden dimension = 4,096
```

Then its embedding matrix might conceptually have shape:

```text
[100000, 4096]
```

Each token ID selects one row.

That row becomes the vector representation of the token.

So:

```text
token ID
↓
embedding lookup
↓
vector
```

---

# 35. Attention

The Transformer creates:

```text
Q = XWQ
K = XWK
V = XWV
```

where:

```text
Q = Query
K = Key
V = Value
```

Then attention is conceptually:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d) V
```

This equation is fundamental.

The steps are:

```text
QKᵀ
↓
similarity scores
↓
divide by √d
↓
softmax
↓
attention weights
↓
multiply by V
```

Every operation here involves tensors.

---

# 36. Why Q, K and V?

Think conceptually.

A query asks:

```text
"What information am I looking for?"
```

A key represents:

```text
"What information do I contain?"
```

A value contains:

```text
"What information should I actually provide?"
```

The query and key determine relevance.

The values provide the information.

You don't need to treat this analogy as mathematically exact, but it provides useful intuition.

---

# 37. The KV cache

This is one of the most important concepts for inference engineering.

Suppose an LLM generates:

```text
The
```

then:

```text
The cat
```

then:

```text
The cat sat
```

then:

```text
The cat sat on
```

At each generation step, the model needs attention over previous tokens.

The keys and values for previous tokens can be reused.

Instead of recomputing them, we store them:

```text
K → cache
V → cache
```

This is the:

```text
KV cache
```

---

# 38. Why KV cache becomes huge

Suppose we have:

```text
sequence length = 32,000
layers = 80
```

and many attention heads.

Every previous token contributes K/V tensors.

As sequence length grows:

```text
KV cache grows
```

This can consume enormous GPU memory.

Therefore:

```text
model weights
+
KV cache
+
activations
+
temporary buffers
```

all compete for GPU memory.

This is why inference engineering cannot focus only on model weights.

---

# 39. GQA and MQA

One method of reducing KV-cache memory is changing the attention architecture.

Traditional multi-head attention might have:

```text
many Q heads
many K heads
many V heads
```

With:

```text
GQA
```

multiple query heads share fewer K/V heads.

With:

```text
MQA
```

many query heads share a single K/V head.

Conceptually:

```text
MHA
Q Q Q Q Q Q Q Q
K K K K K K K K
V V V V V V V V

GQA
Q Q Q Q Q Q Q Q
K   K   K   K
V   V   V   V

MQA
Q Q Q Q Q Q Q Q
K
V
```

The result is substantially less K/V memory.

---

# 40. Prefill and decode

LLM inference has two fundamentally different phases.

The first is:

```text
Prefill
```

The prompt is processed.

Suppose the user gives:

```text
2,000 tokens
```

The model processes those tokens and constructs the KV cache.

The second phase is:

```text
Decode
```

The model generates tokens one by one.

So:

```text
Prompt
 ↓
PREFILL
 ↓
KV cache
 ↓
DECODE
 ↓
token
 ↓
DECODE
 ↓
token
 ↓
...
```

These two phases have different performance characteristics.

---

# 41. TTFT and TPOT

Two important inference metrics are:

```text
TTFT
```

meaning:

```text
Time To First Token
```

and:

```text
TPOT
```

meaning:

```text
Time Per Output Token
```

TTFT is heavily influenced by:

```text
prompt length
prefill computation
model size
system load
```

Decode performance is influenced strongly by:

```text
KV cache
memory bandwidth
batch size
attention implementation
model architecture
```

Therefore optimizing prefill isn't necessarily the same as optimizing decode.

---

# 42. Quantization

Now we're finally ready for quantization.

Suppose the original values are:

```text
-1.00
-0.75
-0.40
0.10
0.60
1.00
```

We want to represent them using INT8.

We map a floating-point range to an integer range.

A simplified symmetric quantization formula is:

```text
scale = max(|x|) / Qmax
```

and:

```text
q = round(x / scale)
```

Then to recover approximately:

```text
x ≈ q × scale
```

For signed INT8:

```text
Qmax = 127
```

Suppose:

```text
max(|x|) = 1.0
```

Then:

```text
scale = 1 / 127
```

A value such as:

```text
x = 0.5
```

becomes approximately:

```text
q = round(0.5 / (1/127))
```

which is:

```text
q ≈ 64
```

Dequantization gives approximately:

```text
64 × (1/127)
≈ 0.504
```

The original:

```text
0.5
```

has been approximated.

That difference is **quantization error**.

---

# 43. Why INT4 is difficult

INT4 has:

```text
2⁴ = 16
```

possible values.

That's incredibly small compared with FP32.

So the challenge becomes:

> How can a model with billions of continuous-valued parameters survive being represented using only 16 levels per quantization group?

This leads to techniques such as:

```text
GPTQ
AWQ
SmoothQuant
group-wise quantization
per-channel quantization
per-token quantization
activation-aware quantization
```

The exact algorithms differ, but the fundamental problem is always:

```text
continuous values
↓
limited discrete representation
↓
minimize accuracy loss
```

---

# 44. Weight quantization vs activation quantization

A crucial distinction:

```text
weight quantization
```

means reducing the precision of:

```text
model parameters
```

while:

```text
activation quantization
```

means reducing the precision of:

```text
intermediate values
```

Weights are relatively static.

Activations depend on the input.

Therefore activation quantization can be more difficult.

---

# 45. KV-cache quantization

We can also quantize:

```text
K
V
```

instead of storing them entirely in FP16/BF16.

This can significantly reduce:

```text
KV cache memory
```

which can allow:

```text
longer context
larger batches
more concurrent users
```

But again, quantization introduces numerical error, so the engineering challenge is balancing:

```text
memory savings
vs
quality
vs
latency
```

---

# 46. FlashAttention

Now we can understand why FlashAttention exists.

The naive attention calculation conceptually performs:

```text
QKᵀ
↓
attention matrix
↓
softmax
↓
multiply V
```

The problem isn't merely arithmetic.

The intermediate attention matrix can be enormous.

If sequence length is:

```text
N
```

the attention matrix has approximately:

```text
N × N
```

elements.

Therefore its memory requirement grows quadratically with sequence length.

FlashAttention changes the implementation so that the computation is performed in tiles/blocks while minimizing expensive memory traffic.

The central idea is:

```text
Don't materialize huge intermediate matrices unnecessarily.
```

This is an extremely important inference-engineering principle.

---

# 47. Kernel fusion

Suppose you perform:

```text
operation A
↓
write to memory
↓
operation B
↓
write to memory
↓
operation C
```

You may instead combine them:

```text
A → B → C
```

inside one kernel.

This reduces:

```text
memory reads
memory writes
kernel-launch overhead
```

This is called:

```text
kernel fusion
```

Again, the recurring theme is:

> **Move less data.**

---

# 48. Tiling

Tiling means breaking a large computation into smaller blocks.

Instead of:

```text
huge matrix
```

you process:

```text
┌─────┬─────┬─────┐
│tile │tile │tile │
├─────┼─────┼─────┤
│tile │tile │tile │
├─────┼─────┼─────┤
│tile │tile │tile │
└─────┴─────┴─────┘
```

The goal is to keep frequently used data in faster memory.

This improves:

```text
cache locality
```

and:

```text
memory reuse
```

---

# 49. Speculative decoding

Another inference optimization attacks a different bottleneck.

Normally:

```text
LLM
 ↓
token 1
 ↓
token 2
 ↓
token 3
 ↓
token 4
```

The decode process is sequential.

Speculative decoding uses a smaller model to propose several tokens:

```text
small model
 ↓
token 1
token 2
token 3
token 4
```

Then the larger model verifies them.

Conceptually:

```text
Draft model
     ↓
candidate tokens
     ↓
Large model verification
     ↓
accept/reject
```

If many tokens can be accepted, the expensive model can generate multiple tokens' worth of progress per verification cycle.

This addresses **sequential decoding latency**, rather than simply reducing model memory.

---

# 50. Continuous batching

Suppose user A requests:

```text
request A
```

and user B arrives later:

```text
request B
```

A naive serving system might process requests independently.

Modern LLM serving systems can dynamically combine active requests into batches.

This is:

```text
continuous batching
```

The batch changes as requests enter and leave.

This is important for:

```text
throughput
GPU utilization
multi-user serving
```

---

# 51. Tensor parallelism

A large model might not fit on one GPU.

Suppose:

```text
model = 140 GB
```

and one GPU has:

```text
80 GB
```

We can distribute computation across GPUs.

For example:

```text
GPU 0
 ↓
part of model

GPU 1
 ↓
part of model
```

This is broadly called:

```text
tensor parallelism
```

The exact partitioning determines communication patterns.

Now networking and interconnect bandwidth become important.

---

# 52. Pipeline parallelism

Another approach divides layers.

For example:

```text
GPU 0:
layers 0–19

GPU 1:
layers 20–39

GPU 2:
layers 40–59

GPU 3:
layers 60–79
```

This is:

```text
pipeline parallelism
```

Now the challenge becomes keeping all GPUs busy while minimizing pipeline bubbles and communication overhead.

---

# 53. The complete inference-engineering stack

At this point you should be able to see the entire stack:

```text
                    APPLICATION
                         │
                         ▼
                    USER PROMPT
                         │
                         ▼
                     TOKENIZER
                         │
                         ▼
                     TOKEN IDs
                         │
                         ▼
                    EMBEDDINGS
                         │
                         ▼
              ┌────────────────────┐
              │    TRANSFORMER     │
              │                    │
              │ Attention          │
              │ MLP                │
              │ Normalization      │
              │ Residuals          │
              └────────────────────┘
                         │
                         ▼
                       LOGITS
                         │
                         ▼
                  NEXT TOKEN
                         │
                         ▼
                      DECODE
```

Underneath:

```text
Transformer
     │
     ├── Tensor operations
     │
     ├── Matrix multiplication
     │
     ├── Attention
     │
     └── MLP
             │
             ▼
          GPU kernels
             │
             ▼
       Tensor Cores / SMs
             │
             ▼
        GPU memory
             │
             ▼
       HBM bandwidth
```

And the numerical representation layer underneath everything:

```text
Bits
 ↓
Bytes
 ↓
INT / FLOAT
 ↓
FP32 / BF16 / FP16 / FP8 / INT8 / INT4
 ↓
Tensor storage
```

---

# 54. The five things that dominate inference engineering

If you want to understand the majority of modern inference optimization work, keep these five dimensions in your head.

### Numerical representation

Ask:

```text
How many bits per value?
```

Examples:

```text
FP32
BF16
FP16
FP8
INT8
INT4
```

### Memory

Ask:

```text
How much data must be stored?
```

This includes:

```text
weights
KV cache
activations
temporary buffers
```

### Memory bandwidth

Ask:

```text
How quickly can that data move?
```

This frequently dominates decode performance.

### Compute

Ask:

```text
How many operations must be performed?
```

and:

```text
How quickly can the hardware perform them?
```

### Communication

Ask:

```text
How much data must move between GPUs/CPUs/devices?
```

This becomes crucial in distributed inference.

---

# 55. The most important equations

You don't need hundreds of equations initially. These are the ones worth internalizing.

The number of representations from `b` bits:

```text
N = 2^b
```

Memory required for parameters:

```text
Memory = Number_of_parameters × Bytes_per_parameter
```

Floating-point conceptual representation:

```text
x = (-1)^sign × significand × 2^exponent
```

Dot product:

```text
a · b = Σ aᵢbᵢ
```

Matrix multiplication:

```text
C = AB
```

Attention:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d)V
```

Arithmetic intensity:

```text
Arithmetic Intensity
=
FLOPs / Bytes moved
```

Quantization:

```text
q = round(x / scale)
```

Dequantization:

```text
x̂ = q × scale
```

And the fundamental optimization principle:

```text
Inference performance
≈
computation
+
memory movement
+
communication
+
scheduling
```

---

# 56. The most important mental model

If you remember only one thing from this entire explanation, remember this hierarchy:

```text
BIT
 │
 │  0 / 1
 ▼
BYTE
 │
 │  8 bits
 ▼
NUMBER
 │
 ├───────────────┐
 ▼               ▼
INTEGER        FLOAT
 │               │
 │               ├── FP32
 │               ├── FP16
 │               ├── BF16
 │               └── FP8
 │
 ├── INT8
 └── INT4
         │
         ▼
       TENSOR
         │
         ▼
   MATRIX MULTIPLY
         │
         ▼
       GPU
         │
    ┌────┴────┐
    ▼         ▼
 COMPUTE    MEMORY
    │         │
    │         └── bandwidth
    │
    └── Tensor Cores
         │
         ▼
     TRANSFORMER
         │
    ┌────┴─────┐
    ▼          ▼
 ATTENTION     MLP
    │
    ▼
 KV CACHE
    │
    ▼
 LLM INFERENCE
    │
    ├── Prefill
    └── Decode
         │
         ▼
 OPTIMIZATION
    │
    ├── Quantization
    ├── FlashAttention
    ├── Kernel fusion
    ├── Tiling
    ├── KV optimization
    ├── Speculative decoding
    ├── Continuous batching
    └── Parallelism
```

That is the **single conceptual map** I would use for learning inference engineering.

---

# 57. What you actually need to master

You don't need to become an electrical engineer or memorize every detail of IEEE 754.

For the kind of work you're interested in, the essential depth is:

```text
Bits / bytes
        ↓
Binary / hexadecimal
        ↓
Signed / unsigned integers
        ↓
Floating point
        ↓
FP32 / FP16 / BF16 / FP8
        ↓
Range / precision / rounding
        ↓
Tensor memory
        ↓
Matrix multiplication
        ↓
FLOPs / FLOPS
        ↓
Memory bandwidth
        ↓
Arithmetic intensity
        ↓
GPU architecture
        ↓
CUDA execution model
        ↓
Tensor Cores
        ↓
Transformer mathematics
        ↓
Attention
        ↓
KV cache
        ↓
Prefill / Decode
        ↓
Quantization
        ↓
FlashAttention
        ↓
Kernel optimization
        ↓
Distributed inference
```

Everything else can be learned when it becomes relevant.

---

# 58. One final example tying everything together

Imagine we deploy a:

```text
70B parameter LLM
```

If stored in FP32:

```text
70B × 4 bytes
≈ 280 GB
```

That creates a huge memory requirement.

Move to BF16:

```text
70B × 2 bytes
≈ 140 GB
```

Now we need multiple GPUs.

Move to INT8:

```text
70B × 1 byte
≈ 70 GB
```

Now the weight memory becomes much smaller.

Move toward 4-bit weights:

```text
70B × 0.5 byte
≈ 35 GB
```

Now the model may fit into a smaller GPU configuration.

But then:

```text
↓
quantization error
```

must be managed.

During inference:

```text
prompt
 ↓
prefill
 ↓
KV cache
```

The KV cache grows with context.

Therefore:

```text
weights + KV cache
```

become the dominant memory consumers.

Then:

```text
FlashAttention
```

reduces unnecessary attention memory traffic.

Then:

```text
GQA/MQA
```

can reduce KV-cache size.

Then:

```text
KV quantization
```

can reduce it further.

Then:

```text
continuous batching
```

improves GPU utilization across users.

Then:

```text
speculative decoding
```

can reduce sequential decode latency.

Then:

```text
tensor/pipeline parallelism
```

allows the model to run across multiple GPUs.

And underneath all of those optimizations, the original problem is still the same:

```text
How many bits?
How many bytes?
Where are those bytes?
How fast can we move them?
How much computation do we perform?
Can the hardware perform that computation efficiently?
```

**That is inference engineering.**
