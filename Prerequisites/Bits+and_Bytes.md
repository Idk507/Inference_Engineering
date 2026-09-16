
# Part 1 — Before AI: What is a Bit?

Let's completely forget AI for a moment.

A computer ultimately works with **bits**.

A bit has only two possible states:

```text
0
1
```

You can think of a bit as a tiny switch:

```text
OFF → 0
ON  → 1
```

That's it.

A single bit cannot represent the number `2`, `5`, `100`, or `3.14`.

It can represent only:

```text
0
1
```

The power comes from putting many bits together.

For example, with two bits:

```text
00
01
10
11
```

There are four possible combinations.

Mathematically:

```text
Number of possible states = 2^number_of_bits
```

So:

```text
1 bit  → 2^1 = 2 states
2 bits → 2^2 = 4 states
3 bits → 2^3 = 8 states
4 bits → 2^4 = 16 states
8 bits → 2^8 = 256 states
```

This equation will become **extremely important later when we study quantization**.

For example, an 8-bit number has:

```text
256 possible representations
```

while a 4-bit number has:

```text
16 possible representations
```

This is the fundamental reason that moving from FP16 to INT8 or INT4 can dramatically reduce memory requirements.

---

# Part 2 — What is a Byte?

A **byte is 8 bits**.

```text
1 byte = 8 bits
```

Therefore:

```text
1 byte = 2^8 = 256 possible bit patterns
```

For example:

```text
10110101
```

is one byte.

Notice something important.

The computer doesn't inherently know that:

```text
10110101
```

means a number.

Those eight bits could represent many different things depending on the interpretation.

For example, they could represent an integer:

```text
10110101₂
```

They could represent part of a floating-point number.

They could represent part of an image.

They could represent text.

They could represent machine instructions.

**Bits have no meaning by themselves. The interpretation comes from the data type.**

This idea is absolutely fundamental to understanding AI hardware.

---

# Part 3 — Binary Numbers

Humans normally use decimal.

We have ten digits:

```text
0 1 2 3 4 5 6 7 8 9
```

Computers commonly use binary.

Binary has two digits:

```text
0 1
```

In decimal:

```text
123
```

means:

```text
1 × 100
+ 2 × 10
+ 3 × 1
```

or:

```text
1 × 10²
+ 2 × 10¹
+ 3 × 10⁰
```

Binary works similarly, except the base is 2.

Consider:

```text
1011
```

Its value is:

```text
1 × 2³
+ 0 × 2²
+ 1 × 2¹
+ 1 × 2⁰
```

Therefore:

```text
= 8 + 0 + 2 + 1
= 11
```

So:

```text
1011₂ = 11₁₀
```

---

# Part 4 — Why 8 Bits Gives 256 Values

Take:

```text
00000000
```

through:

```text
11111111
```

The smallest value is:

```text
0
```

The largest is:

```text
255
```

because:

```text
11111111₂
=
128 + 64 + 32 + 16 + 8 + 4 + 2 + 1
=
255
```

Therefore an unsigned 8-bit integer can represent:

```text
0 → 255
```

This is called:

```text
uint8
```

where:

```text
uint = unsigned integer
```

So:

```text
uint8
```

means:

> An integer represented using 8 bits without a sign.

---

# Part 5 — What About Negative Numbers?

Machine learning obviously needs negative numbers.

Weights can be:

```text
-2.3
-0.5
0.7
1.8
```

So we need a way to represent negative values.

One common representation is **two's complement**.

For an 8-bit signed integer, the range becomes:

```text
-128 → +127
```

Notice that we still have exactly:

```text
256 possible representations
```

but they're divided approximately between negative and positive values.

Therefore:

```text
uint8:

0 → 255
```

whereas:

```text
int8:

-128 → 127
```

This becomes very important when we eventually discuss:

```text
INT8 quantization
```

because LLM weights can be converted from floating-point representations into lower-bit integer representations.

---

# Part 6 — Kilobyte, Megabyte, Gigabyte

Now we can build memory units.

Conceptually:

```text
1 byte = 8 bits
```

Historically, computer memory calculations often use powers of two:

```text
1 KB ≈ 1,024 bytes
1 MB ≈ 1,024 KB
1 GB ≈ 1,024 MB
1 TB ≈ 1,024 GB
```

More precisely, the binary units are:

```text
1 KiB = 1,024 bytes
1 MiB = 1,024 KiB
1 GiB = 1,024 MiB
1 TiB = 1,024 GiB
```

This becomes relevant to LLM inference because model weights consume enormous amounts of memory.

Suppose we have:

```text
7 billion parameters
```

and each parameter requires:

```text
4 bytes
```

Then:

```text
7,000,000,000 × 4
=
28,000,000,000 bytes
```

Approximately:

```text
28 GB
```

So a 7B parameter model stored in FP32 requires roughly:

```text
28 GB
```

just for the weights.

That one calculation is the beginning of **LLM inference engineering**.

---

# Part 7 — But What Exactly Is a Float?

This is where things become interesting.

Consider:

```text
7
```

An integer is relatively straightforward to represent.

But what about:

```text
7.25
```

or:

```text
0.001273
```

or:

```text
-13.78291
```

We need a representation capable of handling fractional values and extremely large/small values.

That's where **floating-point numbers** come in.

A floating-point number is conceptually similar to scientific notation.

For example:

```text
6.02 × 10²³
```

Scientific notation has:

```text
sign
significand
exponent
```

Floating-point representation does essentially the same thing, but in **binary**.

Conceptually:

```text
number = sign × significand × 2^exponent
```

This is the foundation of:

```text
FP32
FP16
BF16
FP8
```

---

# Part 8 — FP32

FP32 means:

```text
Floating Point 32-bit
```

A 32-bit IEEE 754 floating-point number is divided into:

```text
1 bit  → sign
8 bits → exponent
23 bits → fraction
```

So:

```text
┌──────┬──────────┬───────────────────────┐
│ Sign │ Exponent │ Fraction              │
│ 1bit │ 8 bits   │ 23 bits               │
└──────┴──────────┴───────────────────────┘
              32 bits
```

The sign determines whether the number is positive or negative.

The exponent determines the approximate magnitude.

The fraction/significand determines the precision.

This gives FP32 an enormous dynamic range compared with integer representations.

---

# Part 9 — Why Do We Need Both Exponent and Fraction?

Imagine these numbers:

```text
0.000000001
```

and:

```text
1,000,000,000
```

An ordinary fixed-point representation has difficulty efficiently representing both.

Floating point solves this by effectively moving the binary point.

For example, conceptually:

```text
123000
```

can be represented approximately as:

```text
1.23 × 10^5
```

Similarly:

```text
0.000123
```

can be represented as:

```text
1.23 × 10^-4
```

The exponent allows the representation to "move" across different scales.

That is why neural networks heavily rely on floating-point arithmetic.

---

# Part 10 — FP16

Now comes one of the most important ideas in modern AI hardware.

FP32:

```text
32 bits
```

FP16:

```text
16 bits
```

So one FP16 number occupies half the storage of FP32.

Suppose you have:

```text
1 billion parameters
```

FP32:

```text
1,000,000,000 × 4 bytes
≈ 4 GB
```

FP16:

```text
1,000,000,000 × 2 bytes
≈ 2 GB
```

Immediately:

```text
50% memory reduction
```

And potentially substantially less memory traffic.

This is one reason modern GPUs are heavily optimized for reduced-precision arithmetic.

---

# Part 11 — BF16

Then we encounter:

```text
BF16
```

or:

```text
Brain Floating Point 16
```

BF16 also occupies:

```text
16 bits
```

but its bit allocation differs from FP16.

The key idea is:

```text
FP32:
1 sign + 8 exponent + 23 fraction

FP16:
1 sign + 5 exponent + 10 fraction

BF16:
1 sign + 8 exponent + 7 fraction
```

So BF16 sacrifices precision compared with FP32 but preserves the **same exponent width**.

That means BF16 has a range much closer to FP32 than FP16.

This is one reason BF16 became extremely useful for deep-learning training and inference.

---

# Part 12 — The Critical Difference: Range vs Precision

This distinction is one of the most important concepts you should understand before going deeper into inference engineering.

**Range** answers:

> How large or small can the number be?

**Precision** answers:

> How finely can I distinguish between nearby numbers?

Imagine a ruler.

A ruler that goes from:

```text
0 → 100 meters
```

has large range.

But if its smallest marking is:

```text
1 meter
```

it has relatively poor precision.

Another ruler might measure:

```text
0 → 1 meter
```

but have markings every:

```text
0.001 meter
```

That has smaller range but higher local precision.

Floating-point formats make trade-offs between these properties.

---

# Part 13 — Now Connect This to Neural Networks

Suppose a neural network contains:

```text
W = [0.21, -0.73, 1.42, 0.005]
```

These are floating-point numbers.

During inference, the GPU needs to perform operations such as:

```text
y = Wx
```

For a simple example:

```text
W = [0.2, 0.5, -0.3]

x = [2.0, 4.0, 1.0]
```

Then:

```text
y =
0.2 × 2.0
+
0.5 × 4.0
-
0.3 × 1.0
```

Therefore:

```text
y = 0.4 + 2.0 - 0.3
```

giving:

```text
y = 2.1
```

That tiny calculation is conceptually the same operation that occurs billions or trillions of times inside a modern neural network.

---

# Part 14 — Matrix Multiplication Is the Heart of LLM Inference

Suppose:

```text
X
```

is the input matrix and:

```text
W
```

is a neural-network weight matrix.

The fundamental operation is:

```text
Y = XW
```

For example:

```text
X =

[1  2]
[3  4]
```

and:

```text
W =

[5  6]
[7  8]
```

Then:

```text
Y[0,0] = 1×5 + 2×7
       = 19

Y[0,1] = 1×6 + 2×8
       = 22

Y[1,0] = 3×5 + 4×7
       = 43

Y[1,1] = 3×6 + 4×8
       = 50
```

So:

```text
Y =

[19 22]
[43 50]
```

Modern GPUs are essentially extraordinarily sophisticated machines for performing huge numbers of these operations extremely quickly.

---

# Part 15 — Why Inference Engineering Cares So Much About Bits

Now we can finally connect this back to the X article.

Imagine a model containing:

```text
70 billion parameters
```

At FP32:

```text
70B × 4 bytes
=
280 GB
```

That's enormous.

At FP16:

```text
70B × 2 bytes
=
140 GB
```

At INT8:

```text
70B × 1 byte
=
70 GB
```

At INT4:

```text
70B × 0.5 byte
=
35 GB
```

Ignoring metadata and implementation overhead, the difference is dramatic.

This is why concepts that initially sound like low-level computer architecture:

```text
bits
bytes
memory
bandwidth
floating point
integer arithmetic
cache
vectorization
```

become directly relevant to LLM inference.

---

# Part 16 — And This Is Where Quantization Appears

Suppose your neural network contains:

```text
0.237
-0.842
1.391
0.021
-0.113
```

Instead of storing every value using FP16, we might approximate them using fewer bits.

For example, an INT8 quantizer maps a continuous range of floating-point values into only:

```text
256
```

possible integer values.

INT4 gives only:

```text
16
```

possible values.

The general intuition is:

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

As we move downward, memory consumption and often compute/memory bandwidth requirements decrease, but numerical approximation becomes more significant.

That trade-off is one of the central themes of inference optimization.

---

# Part 17 — But There Is Another Problem: Moving the Data

Here's an important insight.

You might initially think:

> "The GPU is slow because multiplication is expensive."

Often, that's not the entire problem.

Modern hardware can perform enormous numbers of arithmetic operations.

The problem can instead become:

```text
Where is the data?
```

and:

```text
How quickly can we move it?
```

Imagine you have:

```text
100 GB of model weights
```

Even if your GPU can perform the mathematics extremely quickly, it still needs to retrieve those weights and feed them into compute units.

This introduces the concept of:

```text
memory bandwidth
```

---

# Part 18 — Memory Hierarchy

A modern computer doesn't have just one kind of memory.

Conceptually:

```text
Registers
   ↓
L1 Cache
   ↓
L2 Cache
   ↓
GPU HBM / VRAM
   ↓
System RAM
   ↓
SSD
```

As you go downward, memory generally becomes:

```text
larger
but slower
```

and moving data between these levels costs time and energy.

Therefore inference engineering isn't simply:

```text
"make matrix multiplication faster"
```

It is:

```text
"make the entire movement + computation pipeline efficient."
```

This is why you'll encounter terms such as:

```text
memory bandwidth
cache locality
tiling
blocking
coalesced memory access
kernel fusion
tensor cores
SIMD
CUDA kernels
FlashAttention
KV cache
quantization
speculative decoding
continuous batching
```

They are all connected.

---

# Part 19 — Where the LLM Fits

Now zoom out.

When you send:

```text
"What is machine learning?"
```

to an LLM, the text is first converted into tokens.

For example, conceptually:

```text
"What is machine learning?"
```

becomes something like:

```text
[What] [ is] [ machine] [ learning] [?]
```

Each token corresponds to an integer ID.

Then those IDs are converted into vectors.

Those vectors pass through transformer layers.

Inside those layers, enormous numbers of operations happen:

```text
embedding
↓
attention
↓
matrix multiplications
↓
normalization
↓
MLP
↓
attention
↓
...
↓
logits
↓
next-token probability
```

And almost everything ultimately reduces to numerical operations on tensors.

---

# Part 20 — The Complete Mental Model

You should eventually be able to visualize an LLM like this:

```text
                 HUMAN TEXT
                     │
                     ▼
                  TOKENS
                     │
                     ▼
              INTEGER TOKEN IDs
                     │
                     ▼
                EMBEDDINGS
                     │
                     ▼
               FLOATING VALUES
                     │
                     ▼
             ┌───────────────┐
             │  Transformer  │
             │               │
             │ Attention     │
             │ MatMul        │
             │ Softmax       │
             │ MLP           │
             │ Normalization │
             └───────────────┘
                     │
                     ▼
                  LOGITS
                     │
                     ▼
                PROBABILITIES
                     │
                     ▼
               NEXT TOKEN
```

And underneath all of this:

```text
bits
 ↓
bytes
 ↓
numbers
 ↓
floating point
 ↓
tensors
 ↓
matrix multiplication
 ↓
GPU kernels
 ↓
memory movement
 ↓
transformer computation
 ↓
LLM inference
```

That is the bridge between **computer architecture and AI**.

---

# The roadmap I recommend for you

Given that your goal is to understand the material behind this post rather than memorize terminology, I would **not** jump directly into FlashAttention, FP8, KV-cache compression, or speculative decoding.

We should build a proper course from the ground up:

```text
LEVEL 0
Bits
Bytes
Binary
Hexadecimal
ASCII / Unicode
Integer representation
Signed integers
Two's complement

        ↓

LEVEL 1
Floating point
Scientific notation
IEEE-754
FP32
FP16
BF16
TF32
FP8
Exponent
Mantissa
Range
Precision
Overflow
Underflow
NaN
Infinity

        ↓

LEVEL 2
Computer memory
Registers
Cache
RAM
VRAM
HBM
Memory bandwidth
Latency
Cache locality
Data movement

        ↓

LEVEL 3
Vectors
Matrices
Tensors
Dot products
Matrix multiplication
FLOPs
FLOPS
Arithmetic intensity

        ↓

LEVEL 4
CPU architecture
SIMD
AVX
NEON
GPU architecture
CUDA
Warps
Threads
SMs
Tensor Cores

        ↓

LEVEL 5
Deep-learning computation
Linear layers
Convolution
Attention
Softmax
Normalization
MLP

        ↓

LEVEL 6
Transformer internals
Q
K
V
Attention
Multi-head attention
GQA
MQA
RoPE
KV cache

        ↓

LEVEL 7
Inference
Prefill
Decode
TTFT
TPOT
Latency
Throughput
Batching

        ↓

LEVEL 8
Quantization
INT8
INT4
GPTQ
AWQ
SmoothQuant
FP8
Activation quantization
Weight quantization
KV-cache quantization

        ↓

LEVEL 9
Inference optimization
Kernel fusion
Tiling
FlashAttention
PagedAttention
Speculative decoding
Continuous batching
Tensor parallelism
Pipeline parallelism

        ↓

LEVEL 10
Frontier inference engineering
Memory-bound vs compute-bound
Roofline model
KV-cache optimization
Attention optimization
Quantized GEMM
Custom CUDA kernels
Compiler optimization
CUDA graphs
Serving architecture
```

 https://remaja.twstalker.com/TheVixhal/status/2097008871231672595?utm_source=chatgpt.com "vixhaℓ @TheVixhal, Twitter Profile | TwStalker"
