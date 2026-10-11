# QwT — Quantization without Tears: From First Principles to Inference Architecture

QwT (Quantization without Tears) is a post-training quantization technique that takes a different approach from GPTQ and AWQ. Instead of concentrating exclusively on how weights should be quantized, it asks a broader question:

After a model has been quantized, can we recover some of the lost accuracy by adding a small, mathematically calculated correction module?

The method was introduced in Quantization without Tears, published at CVPR 2025. It proposes adding lightweight linear compensation modules to a quantized neural network and calculating their parameters using a closed-form solution rather than lengthy retraining. The researchers evaluated the approach across CNNs, vision transformers, multimodal models, diffusion models, and LLMs.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



A subsequent paper, QwT-v2: Practical, Effective and Efficient Post-Training Quantization, introduced channel-wise affine compensation to address additional parameter overhead, latency, and hardware compatibility.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



## 1. Why QwT was introduced

Suppose a linear layer computes:

`Y = X × W.T + b`

After INT8 or INT4 quantization, its approximate output becomes:

`Yq = Q(X) × Q(W).T + bq`

The quantized result generally differs from the original:

`E = Y - Yq`

This difference is the output reconstruction error.

GPTQ attempts to reduce this error by modifying the remaining weights as each column is quantized. AWQ redistributes numerical precision using activation-aware channel rescaling.

QwT takes another direction. It starts with an already-quantized model and introduces a correction function designed to predict the remaining error.

For the original QwT approach, the corrected block output has the form:

`Y_corrected = Yq + C(Xq)`

where `C` is a lightweight linear compensation function.

For example:

`C(Xq) = Xq × A.T + c`

Here, `A` is a compensation weight matrix and `c` is a bias vector.

The corrected output becomes:

`Y_corrected = Yq + Xq × A.T + c`

The goal is to make this output resemble the corresponding full-precision output.

Original QwT: Correcting quantization error

<img width="476" height="236" alt="image" src="https://github.com/user-attachments/assets/6ec5f7c7-9306-48a1-9a44-0cf11dcc9bc7" />


The essential innovation is that the quantized network does not have to preserve exactly the same structure as the original network. A small additional module can compensate for errors introduced by quantization.

## 2. How the compensation parameters are calculated

Consider a calibration dataset containing input activations `X`.

We run these inputs through both the full-precision and quantized networks.

The original output is:

`Y_full`

The quantized output is:

`Y_quantized`

Calculate the residual:

`R = Y_full - Y_quantized`

Now we want a linear function that predicts this residual from the quantized block input:

`R ≈ Xq × A.T + c`

This is a multivariate linear regression problem.

For simplicity, suppose `Xq` has shape `[N, d_in]`, and the residual has shape `[N, d_out]`.

We can augment `Xq` with a column of ones to represent the bias, producing `X_aug`.

The least-squares objective becomes:

`minimize ||X_aug × B - R||²`

A closed-form ridge-regularized solution is:

`B = inverse(X_aug.T × X_aug + λI) × X_aug.T × R`

A numerically stable implementation would solve the associated linear system or use a least-squares routine rather than explicitly forming the inverse.

The coefficient matrix `B` contains the compensation weights and biases.

This is why QwT can be substantially simpler than a full quantization-aware training pipeline: its compensation parameters can be calculated directly from calibration data.

There is still a cost, however. If a block has thousands of input and output features, the additional correction matrix may become substantial, particularly when stored in FP16.

This leads to QwT-v2.

## 3. QwT-v2: Channel-wise affine compensation

The original QwT technique has an important disadvantage: even a lightweight full linear compensation matrix may contain millions of parameters.

Suppose a transformer block has input and output hidden dimensions of 4,096.

A full compensation matrix has:

`4096 × 4096 = 16,777,216 parameters`

At FP16, that matrix occupies 32 MiB.

If many blocks require similar compensation layers, their additional storage and computations can erode the benefits of weight quantization.

QwT-v2 addresses this problem by replacing full matrix compensation with a simple transformation for each output channel:

`Y_corrected[:, c] = alpha[c] × Yq[:, c] + beta[c]`

Each channel needs just two parameters, `alpha` and `beta`.

For 4,096 output channels, this means only 8,192 compensation parameters, rather than more than 16 million.

The dimensionality reduction is enormous.

### 3.1 Deriving the closed-form solution

Suppose one output channel produces the following outputs during calibration:

```
Full precision: [2.0, 4.0, 6.0, 8.0]
Quantized:      [1.5, 3.5, 5.5, 7.5]
```

We want to find two constants, `alpha` and `beta`, satisfying:

`Y_full ≈ alpha × Y_quantized + beta`

In this case, the correct parameters are:

`alpha = 1`

`beta = 0.5`

Applying the correction:

```
1.5 + 0.5 = 2.0
3.5 + 0.5 = 4.0
5.5 + 0.5 = 6.0
7.5 + 0.5 = 8.0
```

The quantized outputs now match the full-precision outputs exactly.

For real calibration data, the relationship is rarely this perfect. We therefore use least-squares regression.

For each output channel:

`alpha = Cov(Yq, Y_full) / Var(Yq)`

`beta = Mean(Y_full) - alpha × Mean(Yq)`

These are the ordinary least-squares slope and intercept formulas. No backpropagation is required.

For a channel with constant quantized outputs, the variance is zero, and the slope is not identifiable. A robust implementation must handle that case explicitly.

The QwT-v2 paper defines precisely this channel-wise regression objective.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



### 3.2 Why post-compensation is different from the original QwT

The two versions use different signals to predict the correction.

| Feature             | Original QwT           | QwT-v2                                |
| ------------------- | ---------------------- | ------------------------------------- |
| Compensation input  | Quantized block input  | Quantized module output               |
| Mathematical form   | `Yq + Xq × A.T + c`    | `alpha × Yq + beta`                   |
| Parameter structure | Full linear projection | Two vectors                           |
| Placement           | Typically block-level  | Can be module-level                   |
| Runtime overhead    | Additional operations  | Can be fused into quantized operators |
| Optimization        | Closed-form regression | Closed-form per-channel regression    |

QwT-v2's central observation is that correcting outputs directly makes a much simpler, channel-wise transformation effective.

The researchers also show that, for compatible integer inference engines, the correction can be incorporated into existing quantization multipliers and bias calculations. Consequently, a separate floating-point compensation operation need not be executed at runtime. This is a hardware-specific integration result rather than an automatic property of every PyTorch model.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



## 4. A numerical example: Recovering quantized outputs

Imagine a neural-network layer with two output channels.

The full-precision outputs are:

```
Y_full = [
    [2.0, 10.0],
    [4.0, 20.0],
    [6.0, 30.0],
    [8.0, 40.0]
]
```

The quantized outputs are:

```
Y_quant = [
    [1.5,  8.0],
    [3.5, 16.0],
    [5.5, 24.0],
    [7.5, 32.0]
]
```

For the first channel, the correction is:

`alpha_1 = 1.0`

`beta_1 = 0.5`

For the second channel:

`alpha_2 = 1.25`

`beta_2 = 0`

Therefore:

```
alpha = [1.0, 1.25]
beta  = [0.5, 0.0]
```

Now apply the correction:

`Y_corrected = Y_quant × alpha + beta`

The resulting output is exactly equal to `Y_full` in this constructed example.

This illustrates the advantage of channel-wise compensation: it can correct systematic scaling and offset errors without introducing another dense matrix multiplication.

However, it cannot recover arbitrary information that quantization has destroyed. For instance, if two very different full-precision outputs collapse to the same quantized output, a deterministic affine transformation of that quantized output cannot distinguish them.

That limitation becomes important at very low bit widths.

## 5. Python implementation of QwT-v2 from scratch

The following PyTorch example demonstrates how to quantize a linear layer, collect full-precision and quantized outputs, calculate channel-wise compensation parameters, and evaluate the correction on separate test data.

It uses synthetic data to keep the process reproducible. This is an educational implementation of QwT-v2's central compensation mechanism, not the complete paper implementation or an optimized integer-only inference kernel.

```
"""QwT-v2: Channel-Wise Affine Compensation From Scratch.This example demonstrates:    1. Quantizing a linear-layer weight matrix.    2. Collecting full-precision and quantized outputs.    3. Solving channel-wise affine compensation analytically.    4. Evaluating the results on unseen test inputs.Requires:    pip install torch"""from dataclasses import dataclassimport torchimport torch.nn.functional as F@dataclassclass CompensationParameters:    """Store the affine compensation parameters for output channels."""    alpha: torch.Tensor    beta: torch.Tensordef symmetric_weight_quantization(    weights: torch.Tensor,    bits: int = 4,) -> torch.Tensor:    """    Simulate per-output-channel symmetric weight quantization.    Args:
```

The central operation is `fit_channel_compensation()`. It computes channel-wise means, covariance, and variance, then derives the slope and intercept analytically.

No optimizer, learning rate, backpropagation, or training loop is required.

The test dataset is deliberately separate from the calibration dataset. This matters because compensation is fitted to calibration outputs, and an improvement on those same outputs does not guarantee improvement on unseen inputs.

For a production implementation, the separate affine transformation may be folded into compatible quantization parameters. For an ordinary floating-point linear layer, the equivalent operation can also be understood as transforming its weights and bias, although the integer-kernel fusion rules require additional care.

## 6. GPTQ vs. AWQ vs. QwT vs. QwT-v2

| Method | Main idea                                    | Mechanism                             | Extra runtime structure               |
| ------ | -------------------------------------------- | ------------------------------------- | ------------------------------------- |
| GPTQ   | Compensate weight rounding errors            | Hessian-based sequential adjustments  | No additional correction layer        |
| AWQ    | Protect activation-sensitive weights         | Channel rescaling before quantization | Typically folded                      |
| QwT    | Correct residual output error                | Closed-form linear correction         | Additional linear modules             |
| QwT-v2 | Correct systematic per-channel output errors | Closed-form affine correction         | Can be fused into quantized execution |

A useful distinction is that GPTQ and AWQ primarily improve the creation of quantized weights. QwT and QwT-v2 attempt to improve the outputs of an already-quantized model.

That means QwT-style correction can conceptually complement a pre-existing quantization algorithm. Whether a particular combination improves overall model performance requires empirical validation.

## 7. Architectural limitations and production trade-offs

Original QwT's biggest concern is the additional computation and memory associated with its correction layers. Although these layers may be relatively small compared with the original network, their overhead can become meaningful when the base model is aggressively quantized.

QwT-v2 substantially reduces that overhead by using only per-channel affine parameters. However, its correction capacity is also more limited. It can correct systematic channel-level scaling and shifting, but cannot directly represent complex relationships between different output channels.

The deployment advantage of QwT-v2 depends on whether the target inference backend can fuse its affine parameters into existing integer quantization or requantization operations. Without this integration, PyTorch still executes additional operations, and latency benefits cannot be assumed.

For production evaluation, compare the full-precision baseline, the quantized model without compensation, and the compensated quantized model using held-out data. Measure layer-output MSE, end-to-end task accuracy, checkpoint size, GPU memory consumption, prefill latency, decode latency, and throughput.

Calibration inputs should be representative of the deployment workload, and the resulting correction parameters should be versioned with the quantized checkpoint.

## 8. Primary research and implementation references

The official resources are the [original QwT CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_Quantization_without_Tears_CVPR_2025_paper.html), the [official QwT PyTorch repository](https://github.com/wujx2001/QwT), and the [QwT-v2 research paper](https://arxiv.org/abs/2505.20932).

The central takeaway: GPTQ corrects weights during quantization, AWQ makes salient weights easier to quantize, and QwT corrects the resulting output distortion. QwT-v2 makes that correction much lighter through channel-wise affine regression, with a pathway to efficient integer-kernel integration.
