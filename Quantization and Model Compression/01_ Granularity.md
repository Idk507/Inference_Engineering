# Quantization Granularity: Per-Tensor vs. Per-Channel vs. Per-Group — From First Principles to LLM Inference

When we quantize a neural network from FP32 or FP16 into INT8 or INT4, the first question is how to represent floating-point values using a smaller number of bits. We already established that affine quantization uses two parameters, the scale and the zero-point, to map floating-point values onto a discrete integer grid.

But there is another equally important question: Should every value in a tensor share the same quantization scale, or should different parts of the tensor have their own scales?

This is the problem of quantization granularity.

It is one of the most consequential design decisions in low-precision inference because it determines how well we preserve small weights, how much quantization error accumulates during matrix multiplication, how much metadata we store, and how efficiently the GPU can execute the quantized computation.

The three central approaches are per-tensor, per-channel, and per-group quantization.

Conceptual comparison: one weight matrix, three ways to assign scales

Per-tensor

One scale for everything

Per-channel

One scale per row

Per-group

One scale per row group

Each color represents an independently calibrated quantization region. The matrix shown has 3 rows and 4 columns.

We will develop this from the mathematics, calculate actual quantization errors, implement the three approaches in PyTorch, and connect them to transformer inference, INT4 weight quantization, GPU memory bandwidth, tensor cores, and KV caches.

## 1. First principles: What does granularity actually mean?

Consider a transformer weight matrix with shape:

```
W.shape = [4096, 4096]
```

The matrix contains:

`4096 × 4096 = 16,777,216 parameters`

If every parameter is stored in FP16, each occupies two bytes.

Therefore:

`16,777,216 × 2 = 33,554,432 bytes = 32 MiB`

Suppose we convert the matrix to INT8. Each weight now requires one byte, so the raw storage becomes 16 MiB.

For INT4, each weight takes four bits, equivalent to half a byte, so the raw storage becomes 8 MiB.

But reducing the bits is only one part of quantization. We also need a mathematical rule for reconstructing each weight.

For symmetric quantization:

`q = clamp(round(w / s), -Q, Q)`

`w_reconstructed = q × s`

Here, `Q` is the maximum positive quantized magnitude. For conventional symmetric INT8, Q = 127. For a symmetric signed INT4 scheme that uses the levels -7 through +7, Q = 7.

For asymmetric quantization:

`q = clamp(round(w / s) + z, q_min, q_max)`

`w_reconstructed = s × (q - z)`

The scale `s` determines the spacing between representable floating-point values, while `z` determines which integer corresponds to floating-point zero.

Granularity determines the collection of values that must share `s` and, for asymmetric quantization, `z`.

If 16 million weights share one scale, that is per-tensor quantization. If each of the 4,096 output channels has its own scale, that is per-channel quantization. If each channel is further divided into groups of 128 weights, every such group receives its own scale, giving per-group quantization.

This changes precision even when all three approaches use the same number of bits per weight.

## 2. Why sharing one scale introduces quantization error

Imagine three groups of neural-network weights.

```
Group A:
[-0.02, -0.01, 0.00, 0.01, 0.02]

Group B:
[-0.20, -0.10, 0.00, 0.10, 0.20]

Group C:
[-10.0, -5.0, 0.0, 5.0, 10.0]
```

These groups have dramatically different numerical ranges.

Group A contains very small weights, Group B contains medium-sized weights, and Group C contains comparatively large weights.

If every group shares the same INT8 symmetric scale, we must accommodate the largest absolute magnitude, 10.

Therefore:

`scale = 10 / 127 ≈ 0.07874`

One integer step represents approximately 0.07874 in floating-point space.

But notice that every nonzero weight in Group A is smaller than half of that step.

For example:

`q(0.02) = round(0.02 / 0.07874) = round(0.254) = 0`

Consequently:

`reconstructed_weight = 0 × 0.07874 = 0`

The original weight 0.02 has disappeared.

The same happens to -0.02, 0.01, and -0.01.

Our quantized neural network effectively loses all the nonzero weights in Group A.

This phenomenon is especially important because a small weight does not necessarily mean an unimportant weight. A small parameter may participate in a sensitive computation, and the errors of many small parameters can accumulate across layers.

Now imagine that Group A has its own scale:

`scale_A = 0.02 / 127 ≈ 0.00015748`

Then:

`q(0.02) = 127`

and:

`q(0.01) ≈ 64`

Their reconstructed values are approximately 0.02000 and 0.01008.

Instead of being eliminated, the small weights are preserved with considerably better precision.

This is the central motivation for finer-grained quantization.

Interactive example: See how a large outlier destroys small values

Largest magnitude in shared tensor

10.0

Shared INT8 scale

## 0.07874

Values rounded to zero

## 2 / 5

| Original | INT8 code | Reconstructed | Absolute error |
| -------- | --------- | ------------- | -------------- |
| 0.010    | 0         | 0.0000        | 0.0100         |
| 0.020    | 0         | 0.0000        | 0.0200         |
| 0.050    | 1         | 0.0787        | 0.0287         |
| 0.100    | 1         | 0.0787        | 0.0213         |
| 0.200    | 3         | 0.2362        | 0.0362         |

Symmetric INT8, rounding to nearest for the positive examples. Increase the outlier magnitude to observe the scale grow and smaller weights collapse toward zero.

## 3. Per-tensor quantization: One scale for the entire tensor

Per-tensor quantization is the simplest granularity.

A complete tensor, regardless of its number of dimensions, shares one scale and one zero-point.

Suppose we have this weight matrix:

```
W = [
  [ 0.01, -0.02,  0.03, -0.04 ],
  [ 0.10, -0.20,  0.30, -0.40 ],
  [ 1.00, -2.00,  3.00, -4.00 ]
]
```

This matrix has three rows and four columns, with twelve weights in total.

For symmetric INT8, the largest absolute value is 4.

Therefore:

`scale_tensor = 4 / 127 ≈ 0.031496`

We use the same scale for every single weight.

Consider the first row.

`0.01 / 0.031496 ≈ 0.3175`, rounded to zero.

`-0.02 / 0.031496 ≈ -0.635`, rounded to -1.

`0.03 / 0.031496 ≈ 0.9525`, rounded to 1.

`-0.04 / 0.031496 ≈ -1.27`, rounded to -1.

The INT8 representation becomes:

```
q_row_1 = [0, -1, 1, -1]
```

After multiplying these integers by the shared scale, the reconstructed row becomes:

```
W_hat_row_1 = [
   0.000000,
  -0.031496,
   0.031496,
  -0.031496
]
```

We can see that the first weight has become zero, while the others are only approximations.

The problem is not that INT8 is inherently incapable of accurately representing values such as 0.01. The problem is that the integer grid must simultaneously represent values near 0.01 and values as large as 4.0.

Per-tensor quantization is computationally attractive because there is very little metadata. The inference kernel can use a single scale across the entire tensor, broadcasting it without complicated scale indexing.

However, when different channels have very different ranges, the single scale is dominated by the largest magnitudes.

This is often called the outlier-dominance problem.

## 4. Per-channel quantization: Each channel gets its own scale

Per-channel quantization addresses this problem by assigning a separate scale to each channel.

First, we need to understand what a channel means in a transformer.

Consider a linear layer:

`Y = X × W.T + bias`

In PyTorch, a conventional linear-layer weight matrix has shape:

```
W.shape = [out_features, in_features]
```

For example:

```
W.shape = [4096, 4096]
```

Each row contains the weights used to compute one output feature.

Therefore, when we apply per-output-channel quantization, each row gets its own scale.

For a weight matrix with 4,096 output features, we have 4,096 scales.

Returning to the earlier matrix:

```
W = [
  [ 0.01, -0.02,  0.03, -0.04 ],
  [ 0.10, -0.20,  0.30, -0.40 ],
  [ 1.00, -2.00,  3.00, -4.00 ]
]
```

The maximum absolute magnitude of each row is different.

```
Row 1: max_abs = 0.04
Row 2: max_abs = 0.40
Row 3: max_abs = 4.00
```

Instead of computing one scale, we compute three:

```
scale_1 = 0.04 / 127 ≈ 0.00031496

scale_2 = 0.40 / 127 ≈ 0.00314961

scale_3 = 4.00 / 127 ≈ 0.03149606
```

Every row now uses a quantization grid suited to its own numerical range.

Look at the first weight, 0.01.

Under per-tensor quantization:

```
q = round(0.01 / 0.031496)
  = 0

reconstructed = 0
```

Under per-channel quantization:

```
q = round(0.01 / 0.00031496)
  = 32

reconstructed ≈ 0.010079
```

This is a dramatic improvement.

The same weight has not changed, the datatype is still INT8, and the quantization equation is identical. Only the scale's scope has changed.

The first row is now approximately reconstructed as:

```
[ 0.010079, -0.020157, 0.029921, -0.040000 ]
```

The reconstruction errors are much smaller than before.

This is why per-channel quantization is frequently preferred for weight matrices with substantial differences in their channel ranges.

PyTorch explicitly supports per-channel quantization with a scale and zero-point vector aligned with a selected tensor axis.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

PyTorch 2.14 documentation



## 5. Per-group quantization: Divide channels into smaller regions

Per-channel quantization is significantly better than per-tensor quantization when different output channels have different ranges.

However, it does not solve every problem.

Imagine a single output channel that contains 4,096 weights.

Most of those weights lie between -0.1 and 0.1, but a small number have magnitudes near 5.0.

Even though the channel has its own scale, the outlier with magnitude 5 still determines that scale.

For INT4 symmetric quantization:

`scale = 5 / 7 ≈ 0.7143`

This scale is far too coarse for weights near 0.01.

Almost all the small values will become zero.

Per-group quantization divides the weights inside the channel into smaller groups.

For example, group size = 128 means that consecutive blocks of 128 weights get independent scales.

For a channel containing 4,096 weights:

`4096 / 128 = 32 groups`

Each group has its own scale.

Instead of representing all 4,096 weights using one quantization grid, we now use 32 locally calibrated grids.

Consider this simplified matrix:

```
W = [
  [ 0.01, -0.02 |  0.50, -1.00 ],
  [ 0.10, -0.20 |  2.00, -4.00 ],
  [ 1.00, -2.00 |  3.00, -4.00 ]
]
```

The vertical separator marks the boundary between groups of two weights.

For INT4 symmetric quantization using the range -7 through +7, the first row receives two scales:

```
Group 1: [0.01, -0.02]

max_abs = 0.02
scale = 0.02 / 7
```

```
Group 2: [0.50, -1.00]

max_abs = 1.00
scale = 1.00 / 7
```

The first group can preserve very small values accurately because it no longer needs to represent the large value -1.0.

This is the central idea of group-wise quantization: make the quantization range locally representative of the values being quantized.

Group-wise quantization is common in low-bit LLM inference because 4-bit integer grids have far fewer representable values than INT8. NVIDIA's TensorRT documentation includes block-based INT4 weight quantization with block sizes 64 and 128 in supported configurations.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)

NVIDIA TensorRT



## 6. Quantifying the improvement: How much error does each method introduce?

We have established why finer granularity can improve precision. Now we need to quantify the effect mathematically.

The quantization error for an individual weight is:

`error_i = original_weight_i - reconstructed_weight_i`

However, a neural network has millions or billions of parameters, so we need a metric that summarizes errors across many weights.

One common metric is mean squared error, or MSE.

`MSE = sum((W - W_hat)²) / N`

Here, `W` is the original weight tensor, `W_hat` is the dequantized tensor, and `N` is the number of weights.

MSE penalizes large errors more heavily than small errors because it squares their magnitudes.

Another useful metric is maximum absolute error:

`MaxError = max(abs(W - W_hat))`

This tells us the largest individual reconstruction error.

We can also calculate relative error, but it must be used carefully. When the original weight approaches zero, dividing the error by the original value can produce an extremely large or undefined result.

Let's compare three methods using our earlier 3 × 4 weight matrix and symmetric INT8 quantization.

| Metric           | Per-tensor | Per-channel | Per-group (2) |
| ---------------- | ---------- | ----------- | ------------- |
| Number of scales | 1          | 3           | 6             |
| MSE              | 0.00009531 | 0.00003131  | 0.00001044    |
| Maximum error    | 0.015748   | 0.015748    | 0.007874      |
| Scale overhead   | Lowest     | Low         | Higher        |

The MSE values are calculated using nearest-integer rounding, symmetric INT8 with Q = 127, and scales derived from each region's maximum absolute value.

Quantization reconstruction error

Mean squared error, symmetric INT8 on the example weight matrix

03E-56E-59E-50.00012Per-tensorPer-channelPer-group (2)

For this particular matrix, per-channel quantization reduces MSE by approximately 67% compared with per-tensor quantization. Per-group quantization with two weights per group reduces MSE by approximately 89% compared with per-tensor quantization.

These are properties of this illustrative tensor, not universal improvement percentages.

The important observation is that all three methods use INT8 storage for the quantized values. Their errors differ because they use different numbers of independently calibrated scales.

### 6.1 Understanding the quantization error bound

With nearest-integer rounding and no clipping, a uniform quantizer has a simple error bound.

For a given scale `s`, the maximum absolute rounding error is:

`abs(error) ≤ s / 2`

Why?

Because adjacent representable values differ by `s`. Any real value lies at most half a step from its nearest representable value.

Consider:

```
scale = 0.1

Representable values:

0.0       0.1       0.2       0.3
 |---------|---------|---------|
```

A value of 0.14 is mapped to 0.1, giving an error of 0.04. A value of 0.16 is mapped to 0.2, giving an error of 0.04.

The largest rounding error is approximately 0.05, half the scale.

Now apply this to granularity.

With per-tensor quantization:

`maximum rounding error ≤ tensor_scale / 2`

With per-channel quantization:

`maximum rounding error for channel c ≤ channel_scale[c] / 2`

With per-group quantization:

`maximum rounding error for group g ≤ group_scale[g] / 2`

This explains why smaller groups often improve accuracy. If a group has a smaller numerical range, its quantization step becomes smaller, and its maximum possible rounding error decreases.

The bound assumes an ordinary uniform quantizer, round-to-nearest, and no saturation. Clipping error is not covered by this formula.

### 6.2 Why smaller scale also reduces expected noise

Under a simplified model where rounding errors are uniformly distributed between `-s/2` and `+s/2`, the expected squared error is:

`Expected squared error ≈ s² / 12`

This is a classical uniform-quantization noise approximation.

It reveals something important: the expected squared error grows quadratically with the scale.

If the scale doubles, the expected squared error becomes approximately four times larger under these assumptions.

If the scale is reduced by a factor of ten, expected squared error falls by approximately a factor of one hundred.

This is why per-channel and per-group quantization can produce substantial accuracy improvements even without changing the bit width.

However, real neural-network weights are not uniformly distributed and errors are often correlated with the original values. Consequently, this formula is best understood as a useful approximation rather than a guaranteed prediction of model accuracy.

## 7. Why quantization error matters during matrix multiplication

So far, we have measured errors in individual weights. But transformer models do not simply store weights. They repeatedly multiply weights by activations.

Consider a linear layer:

`y = W × x`

Here, `W` is a weight matrix and `x` is an input activation vector.

After quantization, we use an approximate matrix:

`W_hat = W + E`

where `E` is the matrix of quantization errors.

The quantized output becomes:

`y_hat = (W + E) × x`

Expanding:

`y_hat = W × x + E × x`

Since the original output is `y = W × x`, the output error becomes:

`output_error = y_hat - y = E × x`

This equation is fundamental to understanding model quality after quantization.

The error is not determined exclusively by how accurately we reconstruct the weights. It also depends on the activations that multiply those weights.

Suppose the original weights are:

```
W = [0.01, 0.02, 0.03]
```

and our quantized weights are:

```
W_hat = [0.00, 0.02, 0.03]
```

Only the first weight has an error:

```
E = [-0.01, 0, 0]
```

Now consider two activation vectors.

For the first input:

```
x = [1, 1, 1]
```

The output error is:

`-0.01 × 1 = -0.01`

This is relatively small.

For the second input:

```
x = [1000, 1, 1]
```

The output error becomes:

`-0.01 × 1000 = -10`

The exact same weight quantization error produces an output error one thousand times larger.

A weight's importance depends partly on the activations it interacts with, not merely its numerical magnitude.

This is a major reason why modern LLM quantization methods go beyond ordinary minimum-and-maximum-based scale selection.

Explore activation-amplified quantization error

Activation magnitude

100

Weight error

## -0.01

Output error

## -1.00

Magnitude of output error (maximum shown: 10)

The weight error stays constant. Only its associated activation changes, demonstrating why weight MSE alone cannot determine downstream model accuracy.

### 7.1 The complete error equation when both weights and activations are quantized

So far, only weights have been quantized.

But in W8A8 inference, both weights and activations use 8-bit representations.

Suppose:

`W_hat = W + E_W`

`x_hat = x + e_x`

The approximate output is:

`y_hat = (W + E_W)(x + e_x)`

Expanding:

`y_hat = Wx + W e_x + E_W x + E_W e_x`

Therefore:

`output_error = W e_x + E_W x + E_W e_x`

There are three sources of error: activation quantization error, weight quantization error, and their interaction.

This is why excellent weight reconstruction does not automatically guarantee excellent end-to-end inference results.

### 7.2 A useful matrix bound

The output error due to weight quantization satisfies:

`||E x||₂ ≤ ||E||₂ × ||x||₂`

Using the Frobenius norm gives a simpler upper bound:

`||E x||₂ ≤ ||E||F × ||x||₂`

This tells us that reducing weight reconstruction error can reduce the upper bound on output error.

But the actual output error depends on the direction of `x` relative to the error matrix `E`.

This distinction matters. Two quantization configurations can have nearly identical weight MSE but produce very different errors on real inference workloads.

### 7.3 Why outliers matter disproportionately

Transformer activations frequently exhibit nonuniform distributions and can contain unusually large values in particular feature dimensions.

Consider a vector where most activations are around 0.1, but one feature reaches 50.

That feature can amplify errors in corresponding weight columns.

If we quantize the weights solely based on their numerical range, we may give insufficient precision to the weights that multiply those important activations.

This motivates activation-aware approaches such as AWQ and methods that redistribute quantization difficulty, such as SmoothQuant.

Granularity determines the structure of scales, while the quantization algorithm determines how those scales and quantized values are chosen. Both influence final quality.

## 8. Per-channel is not always the same as per-row

A subtle but important concept is that per-channel quantization depends on the selected axis.

Suppose:

```
W.shape = [4096, 11008]
```

For a PyTorch `nn.Linear` layer, the conventional weight layout is:

`[out_features, in_features]`

If we quantize by output channel, the scale tensor has shape:

```
scales.shape = [4096]
```

The dequantization operation conceptually broadcasts scales as:

```
W_hat = quantized_W * scales[:, None]
```

Every row has its own scale.

But if we choose the input-feature axis, the scales would have shape:

```
scales.shape = [11008]
```

and conceptually:

```
W_hat = quantized_W * scales[None, :]
```

Every column now gets its own scale.

Neither convention is universally correct for every quantization method. The proper axis depends on the operator, tensor layout, sensitivity patterns, and the inference kernel.

For example, per-output-channel quantization is convenient for many linear and convolution weight kernels. By contrast, activation-aware methods may reason about input-feature importance and perform transformations along the input dimension.

For convolutional weights, the tensor is often shaped:

```
[out_channels, in_channels, kernel_h, kernel_w]
```

A common per-channel configuration uses one scale for each output channel, spanning all input channels and spatial kernel positions belonging to that filter.

In short, the term channel is meaningful only after we define the tensor shape and quantization axis.

## 9. Group size: Why 32, 64, and 128 are important

Per-group quantization introduces a new hyperparameter:

`group_size`

Suppose we have a linear-layer weight matrix:

```
W.shape = [4096, 4096]
```

If groups partition each row along the input-feature dimension, the number of groups per row is:

`groups_per_row = ceil(in_features / group_size)`

The total number of groups is:

`total_groups = out_features × groups_per_row`

For group size 128:

`groups_per_row = 4096 / 128 = 32`

`total_groups = 4096 × 32 = 131,072`

Therefore, we store 131,072 independently selected scales.

For group size 64:

`groups_per_row = 64`

`total_groups = 262,144`

For group size 32:

`groups_per_row = 128`

`total_groups = 524,288`

Smaller groups mean more scales and more flexibility.

But what happens to memory?

Assume we are using INT4 weights and FP16 scales. We will initially ignore zero-points, alignment, padding, and additional metadata.

Each INT4 weight consumes four bits, or 0.5 bytes.

For the 4096 × 4096 matrix:

`raw_weight_storage = 16,777,216 × 0.5 = 8 MiB`

Each FP16 scale consumes two bytes.

The overhead is therefore determined by the number of scale values.

INT4 group-size memory calculator

4096 × 4096 weight matrix, FP16 scale per group, no zero-points or alignment overhead

Group size

32

64

128

256

Independent scales

# 131,072

Total storage

# 8.250 MiB

INT4 payload

8.000 MiB

Scale metadata

0.250 MiB

Weights

Scales

Effective bits per weight: 4.125 bits, rather than exactly 4 bits.

The effective bits per weight for symmetric group-wise quantization with one scale per group can be approximated as:

`effective_bits = weight_bits + scale_bits / group_size`

If we store four-bit weights and one FP16 scale per group:

`effective_bits = 4 + 16 / group_size`

For group size 128:

`effective_bits = 4 + 16 / 128 = 4.125 bits`

For group size 64:

`effective_bits = 4.25 bits`

For group size 32:

`effective_bits = 4.5 bits`

For group size 16:

`effective_bits = 5 bits`

This reveals the fundamental trade-off: smaller groups improve numerical flexibility but consume more metadata.

If asymmetric group-wise quantization also stores an integer zero-point, the effective storage cost increases further.

For instance, if each group stores a 16-bit scale and an 8-bit zero-point, the approximate formula becomes:

`effective_bits = 4 + 24 / group_size`

Real implementations can pack or compress zero-points differently, so the exact effective bit rate depends on the storage format.

## 10. Does finer granularity always reduce error?

This needs careful treatment.

For basic symmetric quantization that uses the maximum absolute value within each quantization region, subdividing a region can only maintain or reduce its maximum magnitude.

That means the resulting local scale cannot be larger than the original scale.

As a result, the worst-case rounding error bound for each smaller region cannot increase, assuming the same quantizer, no clipping, and compatible grouping.

However, the actual MSE does not have to decrease monotonically when we merely recompute local min-max scales.

Quantization grids change when scales change. A value that happened to lie almost exactly on the original coarse grid might fall between points on a new finer grid. Its individual reconstruction error could increase.

The situation becomes more complicated when comparing non-nested group boundaries, asymmetric schemes, clipped calibration ranges, and learned rounding strategies.

If we optimize quantization parameters for the true reconstruction objective, a more flexible set of scales can achieve an error no worse than a less flexible set, provided the finer configuration can reproduce the coarser solution. But a simple min-max calibration heuristic does not necessarily find that optimum.

There is also a larger issue: lower weight MSE does not always mean better language-model quality.

A small change in an attention projection could alter which tokens receive attention. Quantization error in the final language-model head can change the ranking of probable next tokens. Some transformer layers are more sensitive than others.

Consequently, production quantization needs to consider both numerical error and downstream model performance.

## 11. Complete PyTorch implementation: Compare all three granularities

We can now implement symmetric affine quantization in Python using PyTorch.

The implementation below supports arbitrary two-dimensional weight matrices, signed integer bit widths from 2 to 8, configurable group sizes, and comparison metrics. It explicitly handles zero-valued groups, checks invalid inputs, and returns the quantized values, dequantized weights, and scales.

This is a reference implementation for numerical experimentation. It deliberately does not pack INT4 values into physical four-bit storage or implement a fused GPU kernel; a normal PyTorch integer tensor still consumes its underlying integer datatype's storage.

```
"""Reference implementation of symmetric quantization granularities."""from dataclasses import dataclassimport torch@dataclassclass QuantizationResult:    """Hold integer codes, reconstructed weights, and quantization scales."""    quantized: torch.Tensor    reconstructed: torch.Tensor    scales: torch.Tensor    bits: int    granularity: strdef symmetric_quantize(    weights: torch.Tensor,    bits: int = 8,    granularity: str = "per_tensor",    group_size: int = 128,) -> QuantizationResult:    """    Quantize a 2D floating-point weight matrix using symmetric quantization.    Args:        weights: Weight matrix [out_features, in_features].        bits: Signed integer precision, between 2 and 8.        granularity: per_tensor, per_channel, or per_group.        group_size: Number of input-feature weights in each group.    Returns:        QuantizationResult with logical integer codes, scales,
```

When you run this experiment, compare the reconstruction MSE and number of scales. Because the tensor contains deliberate outliers and strong differences between channel magnitudes, it highlights the weaknesses of sharing one quantization scale across unrelated numerical distributions.

The implementation also exposes an important engineering distinction: numerical quantization is not the same as physical compression. Although the reference represents INT4 codes mathematically, it stores them in `torch.int8`, so the raw codes occupy eight bits each. A production INT4 engine must pack two four-bit codes into one byte and use an appropriate decoding or fused matrix-multiplication kernel.

The reference also uses Python loops to make the granularity explicit. Those loops are useful for learning but unsuitable for high-throughput production quantization. Real implementations vectorize calibration and quantization, handle arbitrary tensor layouts, and often fuse dequantization with matrix multiplication.

## 12. Granularity and the memory-bandwidth problem

Quantization is not only about reducing numerical precision. One of its biggest benefits for LLM inference is reducing the amount of weight data transferred from GPU high-bandwidth memory (HBM) into compute units.

During autoregressive decoding, especially at small batch sizes, the GPU frequently spends considerable time fetching model weights from memory.

Consider an illustrative model with eight billion parameters.

Ignoring metadata and non-weight allocations:

| Representation | Raw weight memory |
| -------------- | ----------------- |
| FP32           | 32 GB             |
| FP16/BF16      | 16 GB             |
| INT8           | 8 GB              |
| INT4           | 4 GB              |

These values use decimal GB and assume all eight billion parameters share the stated representation.

If the weights are stored in INT4, memory traffic can potentially be much lower than FP16.

However, group-wise quantization introduces additional memory reads for the scales and possibly zero-points.

Suppose we use symmetric INT4 with FP16 scales and group size 128.

The effective bits per weight are approximately:

`4 + 16/128 = 4.125 bits`

For eight billion weights:

`8 billion × 4.125 / 8 = 4.125 GB`

The approximate metadata overhead is 125 MB beyond the four-gigabyte raw INT4 payload.

For group size 32:

`effective_bits = 4.5`

`total ≈ 4.5 GB`

The metadata overhead is now approximately 500 MB.

These figures exclude weight-layout padding, alignment, unquantized layers, and runtime workspace.

From a purely memory-capacity perspective, both configurations offer a substantial reduction compared with 16 GB of FP16 weights.

But the smaller group size requires more scales to be loaded and applied during computation.

That produces a critical performance trade-off: lower numerical error versus greater scaling and dequantization overhead.

### 12.1 Why group-wise quantization can slow the kernel

Imagine a matrix multiplication kernel processing INT4 weights.

With per-tensor quantization, it can use one scale for the complete matrix.

With per-output-channel quantization, the kernel may apply one scale per output channel, which is often relatively convenient.

With group-wise quantization, the scale can change repeatedly as the kernel moves across the reduction dimension.

Consider a dot product with 512 weights and group size 128.

```
Input features: 512

|---- 128 ----|---- 128 ----|---- 128 ----|---- 128 ----|
     Group 0       Group 1       Group 2       Group 3
       s0            s1            s2            s3
```

If both weights and activations are quantized, scaling may need to be applied to partial accumulations according to their respective quantization regions.

For a simplified case with symmetric per-group weights and a single activation scale, the output can be expressed as:

`y ≈ activation_scale × sum(weight_scale[g] × integer_dot_product[g])`

Each group's integer dot product uses its own weight scale.

This can require additional scale loads, partial reductions, multiplication operations, and synchronization or layout handling, depending on the kernel implementation.

For weight-only quantization with FP16 activations, kernels commonly unpack and dequantize weight fragments inside the matrix multiplication pipeline.

Modern kernels can fuse these operations, but not all group layouts are equally efficient.

A smaller group size may therefore improve model quality while decreasing throughput.

### 12.2 Group alignment and memory coalescing

GPUs execute memory operations most efficiently when adjacent threads access appropriately aligned, contiguous memory locations.

A group size of 128 is attractive partly because it can align well with common matrix tile sizes, vectorized memory operations, and quantized kernel layouts.

But there is no universal rule that 128 is optimal. The best value depends on the quantized format, GPU architecture, matrix dimensions, kernel tiling, and calibration method.

For example, NVIDIA TensorRT supports INT4 block-wise schemes with block sizes 64 and 128, while its NVFP4 format uses blocks of 16. These are distinct numerical formats and kernel contracts, not interchangeable configurations.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)

NVIDIA TensorRT



This is an important production principle: choosing a group size is not simply a question of minimizing tensor reconstruction error. It must also match the inference backend's actual supported layout.

## 13. Per-token, per-head, and per-block: Additional granularities used in LLMs

So far, we have focused on weights.

But quantization granularity also applies to activations and KV caches.

For activations, the tensor shape is typically different from the weight matrix.

For example, hidden states might have the shape:

```
[batch_size, sequence_length, hidden_size]
```

Per-tensor quantization would use one scale for the entire activation tensor.

Per-token quantization instead assigns a scale to the hidden-state vector for each token.

Consider:

```
Token 1 activations:
[0.01, 0.02, -0.03]

Token 2 activations:
[2.0, -3.0, 1.5]

Token 3 activations:
[0.4, 0.2, -0.1]
```

These token vectors have very different numerical ranges.

If they share a single scale, the largest activations from Token 2 dominate the calibration.

Per-token quantization can instead compute a scale for each token vector, allowing smaller ranges to receive finer precision.

For symmetric INT8:

`scale_token[t] = max(abs(activation[t, :])) / 127`

Then all hidden features within token `t` use that scale.

Per-token quantization is often dynamic because activations depend on the input being processed.

This means the runtime calculates scales during inference rather than storing all of them as fixed model parameters.

TensorRT-LLM distinguishes per-channel weight scaling, per-group weight scaling, and dynamically calculated per-token scaling in its quantization infrastructure.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 13.1 Per-head quantization for KV caches

The KV cache has a different structure.

A common logical layout is:

```
[batch, num_kv_heads, sequence_length, head_dim]
```

Although physical inference engines may reorder this layout for performance, the conceptual dimensions remain useful.

Suppose different attention heads produce very different key or value magnitudes.

If all heads share a single quantization scale, a large outlier in one head can reduce the precision available to other heads.

Per-head quantization assigns separate calibration parameters to each KV head.

More fine-grained options can use separate scales for keys and values, specific head dimensions, token blocks, or groups of elements within each head.

However, the metadata layout and quantization implementation become more complicated as granularity increases.

A particularly important constraint is that KV-cache memory grows with the sequence length and number of active requests. Per-token or per-block scale metadata therefore contributes to runtime memory usage rather than only the model checkpoint.

### 13.2 Why K and V might need different quantization strategies

Although keys and values are stored together conceptually as the KV cache, they serve different mathematical roles in attention.

Keys interact with queries to produce attention scores:

`attention_scores = Q × K.T / sqrt(head_dim)`

Values are combined using the attention probabilities:

`attention_output = softmax(attention_scores) × V`

Quantization error in keys can change attention scores and subsequently alter the softmax distribution.

Quantization error in values directly affects the weighted combination of value vectors.

This means that equal quantization MSE for keys and values does not necessarily imply equal impact on the attention output.

Some cache-quantization methods therefore use different granularities or calibration policies for keys and values.

The ideal strategy depends on the architecture, context length, target precision, and actual attention kernels.

## 14. How granularity relates to GPTQ, AWQ, SmoothQuant, and FP4

Granularity is a design dimension, not a complete quantization algorithm. This distinction is essential when comparing production techniques.

For example, two methods can both use 4-bit weights and group size 128 but still produce substantially different quantized models because their calibration objectives differ.

### GPTQ: Minimize the effect of weight quantization on layer outputs

GPTQ is a post-training quantization method that uses approximate second-order information derived from calibration activations to reduce the impact of weight quantization.

Rather than only considering each weight's distance from its nearest quantization level, GPTQ accounts for how weight errors influence a layer's outputs.

It quantizes weights in a carefully chosen order and updates remaining weights to compensate for quantization errors.

Group-wise scaling can be part of the resulting weight representation, but group size is not the central mechanism that defines GPTQ.

### AWQ: Protect activation-salient weights

Activation-aware Weight Quantization approaches the problem differently.

It observes that not all weights are equally important to the model's outputs. Weights associated with particular activation patterns can be especially sensitive.

AWQ applies activation-informed transformations and searches for quantization parameters that reduce the degradation of important weights.

A model can therefore use group-wise INT4 storage while obtaining greater accuracy than ordinary max-absolute group-wise quantization.

The improvement comes from the calibration strategy, not merely the existence of independent groups.

### SmoothQuant: Redistribute activation difficulty

SmoothQuant addresses another difficult problem: activation outliers.

It uses an equivalent transformation of the linear operation to redistribute numerical magnitude between activations and weights.

The mathematical objective is to make the operands easier to quantize while preserving the original unquantized computation, subject to numerical rounding.

This is particularly useful for weight-and-activation quantization such as W8A8.

### FP8, MXFP8, and NVFP4

Floating-point quantization also has granularity, but it differs from integer affine quantization.

In FP8 or FP4 formats, each stored number contains a floating-point encoding rather than a uniformly spaced integer grid.

Scales can still be shared across entire tensors, rows, or small blocks.

For example, NVIDIA's NVFP4 quantization employs block-level scaling with 16-value blocks in its documented TensorRT scheme. MXFP8 similarly uses block-oriented scaling with its own scale representation.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)

NVIDIA TensorRT



These schemes make the broader point that the numerical datatype and quantization granularity are independent architectural decisions, although actual hardware and kernel implementations support particular combinations.

## 15. Choosing the correct granularity in production

There is no universally optimal granularity. The right selection depends on the target hardware, layer sensitivity, model architecture, quantization algorithm, and acceptable latency.

| Strategy           | Precision potential                        | Metadata cost    | Typical suitability             |
| ------------------ | ------------------------------------------ | ---------------- | ------------------------------- |
| Per-tensor         | Lower for heterogeneous tensors            | Minimal          | Simple INT8/FP8 paths           |
| Per-output-channel | Better when channels have different ranges | Low              | INT8 weights                    |
| Per-group, 128     | Good accuracy/overhead balance             | Moderate         | INT4 LLM weights                |
| Per-group, 64      | Potentially better local precision         | Higher           | More sensitive INT4 layers      |
| Per-group, 32      | High local flexibility                     | Higher still     | Aggressive low-bit quantization |
| Per-token          | Adapts to each input token                 | Runtime scales   | Dynamic activation quantization |
| Per-head/block     | Adapts to attention-cache structure        | Layout-dependent | Selected KV-cache schemes       |

These are qualitative tendencies, not universal performance rankings.

A production system must measure whether an apparent improvement in numerical reconstruction quality translates into better downstream inference behavior.

### 15.1 Calibration and evaluation

For a language model, a useful experiment begins by selecting a representative calibration dataset that includes the distribution of prompts the application will actually serve.

If the application supports coding, reasoning, retrieval-augmented generation, and long-context question answering, the calibration set should include examples of those workloads rather than consisting only of short general text.

The model can then be quantized with different configurations, for example INT4 group sizes 32, 64, and 128.

The first evaluation measures reconstruction errors at the tensor or layer level. MSE, normalized reconstruction error, cosine similarity, and output activation differences help identify sensitive layers.

The second evaluation measures actual model quality. This can include perplexity, next-token logit differences, answer quality, task success, and accuracy on representative benchmarks.

The third evaluation measures infrastructure performance. End-to-end latency, decode tokens per second, time to first token, GPU memory allocation, HBM bandwidth, and kernel execution time determine whether the configuration is viable in production.

A configuration that improves weight MSE but degrades inference throughput may not be the best choice.

Likewise, a configuration that looks excellent on a small calibration dataset may fail on long-context or distribution-shifted workloads.

### 15.2 A production evaluation flow

FP16/BF16 reference model

Baseline accuracy, latency, and memory

Representative calibration dataset

Short, long-context, and domain-specific prompts

Per-tensor

Per-channel

Per-group

Layer and model evaluation

Reconstruction error, logits, perplexity, task performance

GPU serving benchmark

TTFT, TPOT, throughput, memory, kernel efficiency

Select the deployment configuration

Quality constraints + memory budget + latency SLOs

For observability, preserve the quantization configuration alongside model-version metadata. Record the bit width, group size, quantization axis, scale datatype, calibration dataset version, inference-engine version, and kernel configuration. This makes performance regressions and quality changes easier to reproduce.

Quantized checkpoints should also be protected like other model artifacts: use access-controlled artifact repositories, verify integrity with hashes, and avoid loading untrusted serialized code. These concerns are separate from quantization accuracy but remain important in a production inference system.

## 16. The relationship between granularity and mixed-precision quantization

An additional strategy worth understanding is mixed-precision quantization.

So far, we assumed every layer uses the same bit width. For example, the entire model might use INT4.

But different transformer layers may have different sensitivity to quantization.

The embedding layer, attention projections, feed-forward projections, normalization operations, and output language-model head can respond differently to reduced precision.

A mixed-precision configuration might keep some sensitive operations in BF16, quantize most large weight matrices to INT4, and use a different format for selected activation paths.

Granularity can also vary across layers.

For example, a system could use per-channel INT8 for one set of matrices and group-wise INT4 for another set, provided the serving engine supports those formats efficiently.

This introduces a larger optimization problem.

Instead of choosing one group size for the whole model, the inference system seeks a configuration that minimizes error while satisfying memory and latency constraints.

Conceptually:

`Minimize: total model quality degradation`

Subject to:

`model_memory ≤ available_memory`

`inference_latency ≤ latency_budget`

`quantization_format is supported by hardware`

`accuracy ≥ required_threshold`

The optimization variables can include bit width, quantization algorithm, granularity, group size, clipping thresholds, and the selection of layers that remain in higher precision.

In advanced inference systems, this becomes a constrained hardware-aware optimization problem rather than a simple conversion from FP16 to INT4.

## 17. Final mental model

The entire topic can be reduced to one principle:

Quantization granularity defines how much of a tensor must share the same numerical approximation grid.

Per-tensor quantization forces all values to share a single grid. It is efficient and simple, but large outliers can reduce precision across the whole tensor.

Per-channel quantization allows different channels to use different grids. This improves precision when channels have different value distributions, usually with relatively modest metadata overhead.

Per-group quantization divides channels or other selected dimensions into smaller regions. This is especially useful for low-bit weight quantization, where the limited number of integer levels makes range selection critical.

Finer granularity tends to improve the achievable reconstruction fidelity, but it also increases metadata, scale management, and potentially dequantization overhead. It does not automatically guarantee better downstream model quality or faster inference.

The practical decision is therefore not simply which granularity gives the lowest MSE?

It is:

Which combination of precision, quantization granularity, calibration method, and GPU kernel preserves sufficient model quality while meeting the memory and latency requirements of the serving workload?

That is the key engineering question behind efficient INT8, INT4, and emerging block-scaled low-precision inference.

For further reference, the [NVIDIA TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) and [TensorRT-LLM quantization documentation](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html) provide hardware-specific details on supported granularities and inference formats.
