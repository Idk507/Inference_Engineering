# AWQ: Activation-Aware Weight Quantization — From First Principles to INT4 LLM Inference

Activation-Aware Weight Quantization (AWQ) is a post-training quantization technique that compresses large language models into low-bit representations, commonly INT4, while protecting weights that are particularly important to the model's computations.

The fundamental idea behind AWQ is that not all neural-network weights are equally sensitive to quantization, and the importance of a weight cannot be determined solely by looking at its magnitude.

Instead, we should examine the activations that interact with those weights.

AWQ introduces an elegant approach called activation-aware channel rescaling. It scales selected weight channels before quantization and applies an inverse transformation to their corresponding activations. The two transformations cancel in exact arithmetic, preserving the original linear computation before quantization. However, the rescaled weights can be represented more accurately after low-bit quantization.

This is a very different strategy from GPTQ, which uses approximate second-order information and sequentially modifies remaining weights to compensate for quantization errors.

AWQ was introduced in the paper [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) by Ji Lin and collaborators and received the MLSys 2024 Best Paper Award. The paper reported that protecting a small fraction of salient weights can significantly improve quantization quality without requiring mixed-precision storage for those individual weights.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



## 1. Why ordinary INT4 quantization is not sufficient

Consider a transformer linear layer:

`Y = X × W.T`

Here, `X` contains input activations and `W` contains learned weights.

Suppose:

```
Input activation:

X = [0.1, 100.0, 0.2]

Weights:

W = [0.8, 0.05, 0.6]
```

The linear layer output is:

```
Y = 0.1 × 0.8
  + 100.0 × 0.05
  + 0.2 × 0.6

  = 0.08 + 5.0 + 0.12

  = 5.20
```

Notice that the second weight, `0.05`, is numerically the smallest, but it contributes the most to the output.

Its activation is 100.

Now imagine quantizing the weights using a coarse low-bit representation that changes:

`0.05 → 0.00`

The quantized output becomes:

```
Y_quantized = 0.1 × 0.8
            + 100.0 × 0
            + 0.2 × 0.6

            = 0.20
```

The output error is:

`5.20 - 0.20 = 5.00`

This is an enormous error caused by removing a numerically small weight.

The important observation is that a weight's importance is related to the activation multiplying it.

For a single output neuron:

`y = sum(x_i × w_i)`

If quantization changes a weight by `e_i`:

`w_quantized_i = w_i + e_i`

then the output error is:

`delta_y = sum(x_i × e_i)`

An error of 0.05 associated with an activation of 100 may be far more harmful than the same error associated with an activation of 0.1.

This is the central motivation for AWQ.

## 2. Understanding salient weights and activation channels

AWQ's research found that quantization sensitivity is highly nonuniform. A small fraction of important weights can disproportionately influence model quality.

However, protecting individual weights by retaining them in FP16 can create an irregular mixed-precision representation.

A GPU would have to process some weights as INT4 while handling a scattered collection of higher-precision weights.

Such irregular access patterns can complicate vectorization, memory access, and fused matrix-multiplication kernels.

Instead, AWQ uses a structured transformation that keeps the low-bit weight representation regular.

To understand the terminology, consider a weight matrix:

```
W.shape = [out_features, in_features]
```

For example:

```
W.shape = [4, 6]
```

```
             Input channels
            0   1   2   3   4   5

Output 0    w   w   w   w   w   w
Output 1    w   w   w   w   w   w
Output 2    w   w   w   w   w   w
Output 3    w   w   w   w   w   w
```

Each column corresponds to an input feature. A large activation in input channel 2 interacts with every weight in column 2.

Therefore, AWQ can apply a shared scaling factor to an input-feature column across all output rows.

Activation-aware selection of weight channels

<img width="482" height="211" alt="image" src="https://github.com/user-attachments/assets/59249ccb-55a4-4f65-998a-0858ef1866ba" />


AWQ searches for scaling factors along the input-feature dimension. It does not necessarily give each highlighted weight a separate storage datatype.

This distinction from per-output-channel quantization is essential.

Per-output-channel quantization often gives each row a separate quantization scale. AWQ's activation-aware rescaling instead operates across input-feature columns. The resulting weights can still be quantized in groups along the input-feature dimension.

The two mechanisms work together, but they serve different purposes.

## 3. Deriving activation-aware channel rescaling

This is the mathematical heart of AWQ.

Start with a linear transformation:

`Y = X × W.T`

We introduce a positive scaling vector:

`S = [s1, s2, ..., sn]`

where each scaling factor corresponds to one input feature.

Define transformed activations:

`X_scaled = X / S`

and transformed weights:

`W_scaled = W × S`

Here, the multiplication and division are element-wise along corresponding input-feature dimensions.

Now calculate the result:

`Y_scaled = (X / S) × (W × S).T`

For a single output neuron:

`Y_scaled = sum((x_i / s_i) × (w_i × s_i))`

The scaling terms cancel:

`(x_i / s_i) × (w_i × s_i) = x_i × w_i`

Therefore:

`Y_scaled = Y`

This is an equivalent transformation.

Before quantization, the model's linear output is mathematically unchanged.

However, after quantization, the behavior becomes more interesting.

Instead of quantizing the original weights:

`Q(W)`

we quantize the scaled weights:

`Q(W × S)`

The final computation is approximately:

`Y_quantized = (X / S) × Q(W × S).T`

The scaling changes which weights receive finer effective precision when the quantization grid is determined.

### 3.1 A numerical example

Consider just two input channels:

```
X = [0.1, 100.0]

W = [0.8, 0.05]
```

The original output is:

`0.1 × 0.8 + 100 × 0.05 = 5.08`

Suppose the scaling factors are:

```
S = [1, 10]
```

The transformed weights become:

```
W_scaled = [0.8, 0.5]
```

The transformed activations become:

```
X_scaled = [0.1, 10]
```

The output remains:

`0.1 × 0.8 + 10 × 0.5 = 5.08`

Now suppose we use symmetric INT4 quantization, with integer levels from -7 through +7 and one shared scale.

For the original weights:

`scale = 0.8 / 7 ≈ 0.114286`

The weight `0.05` rounds to zero.

For the scaled weights:

`scale = 0.8 / 7 ≈ 0.114286`

The transformed weight `0.5` now maps to approximately four integer steps, reconstructing as `0.457143`.

The quantized output becomes:

`0.1 × 0.8 + 10 × 0.457143 ≈ 4.65143`

The error is approximately 0.42857, compared with 5.0 without rescaling.

FP reference

# 5.08

Naive INT4

# 0.08

Rescaled INT4

# 4.651

Output absolute error in the two-weight example



<img width="510" height="335" alt="image" src="https://github.com/user-attachments/assets/25306ed0-f7bb-46fc-9d01-01e0761283eb" />

This illustrates why rescaling matters: it can make a previously underrepresented weight occupy more quantization levels while proportionally reducing the activation that multiplies it.

But scaling every channel by a very large number would not solve the problem.

The rescaled weight range would also change, potentially increasing the quantization step and harming other channels.

That is why AWQ needs to search for suitable scaling factors.

## 4. Why rescaling reduces quantization error

Let's derive the quantization error after rescaling.

Suppose the original weight is `w`, and we multiply it by a scaling factor `s`.

We quantize the transformed weight:

`q_scaled = Q(s × w)`

Let the scaled-domain quantization error be:

`e_scaled = Q(s × w) - s × w`

During computation, the activation is divided by `s`.

Therefore the effective weight used by the model is:

`w_effective = Q(s × w) / s`

Substituting:

`w_effective = w + e_scaled / s`

The effective weight error becomes:

`e_effective = e_scaled / s`

This is the core of the AWQ intuition.

If we increase `s`, the quantization error in the scaled domain is divided by a larger number when mapped back to the original domain.

Under a simplified fixed-quantization-step assumption, the absolute effective weight error bound changes from:

`delta / 2`

to approximately:

`delta / (2 × s)`

But we must be careful with this interpretation.

In real group-wise quantization, the quantization step is often recalculated after the weights are rescaled.

Therefore, increasing `s` can also change the group's maximum magnitude and its quantization step.

We cannot assume that making `s` arbitrarily large will continuously reduce error.

AWQ is fundamentally about balancing how scaling changes the quantization error distribution across a group.

### 4.1 The scaling trade-off

Consider two weights sharing a quantization group:

```
w1 = 0.80
w2 = 0.05
```

The second weight has a much larger activation.

Initially, the first weight determines the maximum range.

Increasing the second weight by a factor of 10 produces:

```
w1_scaled = 0.80
w2_scaled = 0.50
```

The group's maximum remains 0.80.

Therefore the quantization step does not increase, but the second weight now occupies more integer levels.

This is advantageous.

However, if we scale the second weight by 100:

```
w1_scaled = 0.80
w2_scaled = 5.00
```

Now the second weight determines the group range.

The quantization step becomes:

`5 / 7 ≈ 0.7143`

The first weight may now be reconstructed poorly.

So the optimal scaling factor is neither necessarily 1 nor the largest possible value.

It depends on the distribution of weights, the activation magnitudes, and the quantizer's grouping structure.

Explore the channel-scaling trade-off

Scaling factor for second input channel

10.0×

Quantization step

## 0.1143

Output absolute error

## 0.4286

| Original output         | 5.0800 |
| ----------------------- | ------ |
| Quantized output        | 4.6514 |
| Effective first weight  | 0.8000 |
| Effective second weight | 0.0457 |

The weight group is requantized at each scaling factor. Quantization error is not monotonic; overly aggressive scaling can harm other weights.

## 5. How AWQ chooses the scaling factors

AWQ does not typically search independently for an arbitrary floating-point scale for every weight.

That would be expensive and would introduce an impractically large optimization space.

Instead, it builds candidate channel scales from activation statistics and searches over a relatively small family of transformations.

The commonly used formulation is:

`scale_i = activation_magnitude_i ^ alpha`

where `alpha` is a search parameter, typically in the interval from 0 to 1.

The activation magnitude for channel `i` can be estimated as:

`activation_magnitude_i = mean(abs(X[:, i]))`

assuming the calibration activation matrix has samples in rows and features in columns.

When `alpha = 0`:

`scale_i = 1`

This means no relative activation-aware scaling.

When `alpha = 1`:

`scale_i = activation_magnitude_i`

Channels with larger average activation magnitudes receive proportionally larger weight scaling factors.

For intermediate values, such as `alpha = 0.5`:

`scale_i = sqrt(activation_magnitude_i)`

This provides a softer transformation.

The original AWQ implementation performs a grid search over candidate alpha values, tests the resulting quantized module or submodule against its unquantized output, and selects the candidate minimizing reconstruction MSE. Its `auto_scale.py` implementation also normalizes candidate scales using their geometric mean of extrema to control their global magnitude.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 5.1 Why does the alpha parameter matter?

Imagine three activation channels with average magnitudes:

```
Channel 0 = 1
Channel 1 = 16
Channel 2 = 100
```

Compare the corresponding scale vectors.

| Alpha | Channel 0 | Channel 1 | Channel 2 |
| ----- | --------- | --------- | --------- |
| 0.0   | 1         | 1         | 1         |
| 0.25  | 1         | 2         | 3.162     |
| 0.5   | 1         | 4         | 10        |
| 0.75  | 1         | 8         | 31.623    |
| 1.0   | 1         | 16        | 100       |

At alpha = 0, no channel receives preferential treatment.

At alpha = 1, high-activation channels receive aggressive scaling, which can increase the quantization error of other channels in their shared groups.

Intermediate values often provide a useful balance.

The best alpha depends on the weight distribution and layer-output error, not activation magnitude alone.

### 5.2 The AWQ search objective

For a candidate scaling vector `S`, we can define the effective quantized weights as:

`W_hat(S) = Q(W × S) / S`

The search objective is:

`S_best = argmin_S ||X × W.T - X × W_hat(S).T||²`

Equivalently:

`S_best = argmin_S ||X × W.T - (X / S) × Q(W × S).T||²`

The minimization is performed over the limited set of candidate scaling vectors, not over all possible continuous values.

The real implementation may evaluate a larger submodule rather than a single linear operation. This allows scaling choices to be judged by their effects on an attention or feed-forward computation.

This is one of the most important differences between AWQ and GPTQ.

GPTQ uses a curvature approximation to update unquantized weights sequentially.

AWQ uses activation statistics and a small search over equivalent scaling transformations.

AWQ does not require calculating or inverting a Hessian.

## 6. Why AWQ remains hardware-friendly

A naïve implementation of AWQ might explicitly divide the input activations by `S` during every forward pass.

For example:

```
x_scaled = x / scalesoutput = linear(x_scaled, quantized_scaled_weights)
```

Although this is mathematically valid, it introduces additional runtime operations.

AWQ aims to avoid unnecessary overhead by folding the scaling transformation into adjacent operations where possible.

Suppose the original model contains:

`LayerNorm → Linear`

The linear layer receives:

`x = LayerNorm(h)`

The AWQ transformation requires:

`x_scaled = x / S`

If the normalization layer has learned affine parameters:

`x = normalized_h × gamma + beta`

then we can divide the affine parameters by `S`:

`gamma_new = gamma / S`

`beta_new = beta / S`

The resulting normalization output becomes:

`x_scaled = x / S`

At the same time, the following linear layer's weight columns are multiplied by `S`.

The transformations cancel before quantization.

This is called scale folding.

Similar folding techniques can be applied across selected adjacent linear layers. However, they must preserve every path using the intermediate tensor. For example, a residual connection or a nonlinear transformation between two layers can prevent arbitrary movement of scales.

The official implementation includes specific scaling handlers for normalization-to-linear, linear-to-linear, and selected activation-to-linear structures, reflecting these architectural constraints.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



The result is that AWQ can retain a regular low-bit weight representation and avoid introducing a separate FP16 representation for scattered salient weights.

## 7. AWQ versus GPTQ: Two different ways to protect accuracy

We have now seen the core mechanisms of both algorithms.

GPTQ starts with the question: If I round this weight, how can I adjust the remaining weights to compensate for the resulting output error?

AWQ starts with a different question: Can I transform the weights and activations, without changing the original function, so that important channels become easier to quantize?

| Dimension               | GPTQ                                | AWQ                                             |
| ----------------------- | ----------------------------------- | ----------------------------------------------- |
| Main technique          | Sequential error compensation       | Activation-aware channel rescaling              |
| Calibration information | Input-feature correlations          | Activation magnitudes and output reconstruction |
| Hessian required        | Yes, approximate                    | No                                              |
| Matrix factorization    | Cholesky-based                      | Not required for basic AWQ                      |
| Weight update           | Modifies future unquantized weights | Rescales input-channel weight columns           |
| Search or processing    | Sequential weight-column decisions  | Candidate scale-ratio search                    |
| Common storage          | Group-wise INT4                     | Group-wise INT4                                 |
| Main computational cost | Hessian processing and compensation | Candidate calibration evaluations               |

Neither technique guarantees higher accuracy across every model and workload.

GPTQ can exploit correlations between input features to compensate for quantization decisions. AWQ instead seeks a transformation that protects activation-sensitive directions before quantization.

AWQ can also be easier to calibrate in settings where Hessian storage and factorization would be costly, although actual quantization time depends on implementation and hardware.

## 8. Implementing AWQ from scratch in PyTorch

The following example implements the central AWQ rescaling idea with a small linear layer. It uses activation statistics to build scaling candidates, simulates symmetric four-bit group-wise quantization, searches for the best alpha, and compares the output reconstruction error against ordinary group-wise INT4 quantization.

This is an educational implementation of AWQ's activation-aware scaling mechanism, not a replacement for the full architecture-specific quantization pipeline. It runs on CPU or CUDA without requiring a pretrained LLM.

```

"""
AWQ From Scratch: Activation-Aware Channel Rescaling.

Demonstrates:
    1. Activation-aware channel statistics.
    2. Equivalent weight and activation transformations.
    3. Symmetric group-wise INT4 quantization.
    4. AWQ alpha grid search.
    5. Output reconstruction error comparison.

This is a numerical reference implementation.
Integer codes are stored as int8, not physically packed into INT4.
"""

from dataclasses import dataclass

import torch


@dataclass
class AWQResult:
    """Store AWQ search results and quantized weight reconstruction."""

    alpha: float
    channel_scales: torch.Tensor
    integer_codes: torch.Tensor
    effective_weights: torch.Tensor
    output_mse: float


def validate_tensors(
    weights: torch.Tensor,
    inputs: torch.Tensor,
) -> None:
    """Validate linear-layer weights and calibration inputs."""
    if weights.ndim != 2 or inputs.ndim != 2:
        raise ValueError("Both weights and inputs must be 2D.")

    if weights.shape[1] != inputs.shape[1]:
        raise ValueError(
            "Input feature dimension must match weight columns."
        )

    if weights.numel() == 0 or inputs.numel() == 0:
        raise ValueError("Tensors cannot be empty.")

    if not weights.is_floating_point() or not inputs.is_floating_point():
        raise TypeError("Expected floating-point tensors.")

    if weights.device != inputs.device:
        raise ValueError("Weights and inputs must use the same device.")

    if not torch.isfinite(weights).all().item():
        raise ValueError("Weights contain nonfinite values.")

    if not torch.isfinite(inputs).all().item():
        raise ValueError("Inputs contain nonfinite values.")


def groupwise_quantize(
    weights: torch.Tensor,
    bits: int = 4,
    group_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply symmetric group-wise integer quantization.

    Each output row is divided into consecutive groups
    along the input-feature dimension.

    Args:
        weights: Matrix [out_features, in_features].
        bits: Signed integer precision from 2 to 8.
        group_size: Maximum number of weights per group.

    Returns:
        Tuple of logical integer codes and reconstructed weights.
    """
    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError("bits must be between 2 and 8.")

    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be positive.")

    rows, cols = weights.shape
    qmax = (1 << (bits - 1)) - 1

    codes = torch.empty_like(weights, dtype=torch.int8)
    recovered = torch.empty_like(weights)

    for start in range(0, cols, group_size):
        end = min(start + group_size, cols)

        # Quantization groups are independent in every output row.
        block = weights[:, start:end]
        max_abs = block.abs().amax(dim=1, keepdim=True)

        # Avoid division by zero for entirely zero-valued groups.
        scale = torch.where(
            max_abs > 0,
            max_abs / qmax,
            torch.ones_like(max_abs),
        )

        block_codes = torch.round(block / scale)
        block_codes = block_codes.clamp(-qmax, qmax)

        codes[:, start:end] = block_codes.to(torch.int8)
        recovered[:, start:end] = block_codes * scale

    return codes, recovered


def activation_scales(
    inputs: torch.Tensor,
    alpha: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute AWQ-like input-channel scaling factors.

    Formula:
        magnitude_i = mean(abs(X[:, i]))
        scale_i = magnitude_i ** alpha

    The geometric-midpoint normalization controls global scale.

    Args:
        inputs: Calibration matrix [samples, input_features].
        alpha: Search exponent in [0, 1].
        eps: Small positive clamp for numerical stability.

    Returns:
        Positive scaling vector [input_features].
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")

    if eps <= 0:
        raise ValueError("eps must be positive.")

    magnitude = inputs.abs().mean(dim=0)

    scales = magnitude.clamp_min(eps).pow(alpha)

    # A common multiplier across every channel does not change
    # the mathematical effective weights for an ideal homogeneous
    # group quantizer. Normalize for numerical stability.
    normalization = torch.sqrt(scales.max() * scales.min())
    scales = scales / normalization

    return scales


def reconstruct_with_scales(
    weights: torch.Tensor,
    channel_scales: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize scaled weights and map them back to original coordinates.

    AWQ transformation:
        W_scaled = W * S
        X_scaled = X / S

    Effective reconstructed weights:
        W_effective = Q(W * S) / S
    """
    scaled_weights = weights * channel_scales.unsqueeze(0)

    codes, quantized_scaled = groupwise_quantize(
        scaled_weights,
        bits=bits,
        group_size=group_size,
    )

    effective_weights = (
        quantized_scaled / channel_scales.unsqueeze(0)
    )

    return codes, effective_weights


def reconstruction_mse(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    effective_weights: torch.Tensor,
) -> float:
    """Calculate mean squared difference between linear outputs."""
    reference = inputs @ weights.T
    approximate = inputs @ effective_weights.T

    return torch.mean((reference - approximate).square()).item()


def search_awq_scales(
    weights: torch.Tensor,
    inputs: torch.Tensor,
    bits: int = 4,
    group_size: int = 8,
    num_candidates: int = 20,
) -> AWQResult:
    """
    Search for activation-aware channel scales.

    Candidate alphas are evaluated against output reconstruction MSE.

    Args:
        weights: Linear weights [out_features, in_features].
        inputs: Calibration activations [samples, in_features].
        bits: Quantization precision.
        group_size: Quantization group size.
        num_candidates: Number of alpha values in [0, 1).

    Returns:
        Best AWQResult found during grid search.
    """
    validate_tensors(weights, inputs)

    if not isinstance(num_candidates, int) or num_candidates < 1:
        raise ValueError("num_candidates must be positive.")

    # FP32 provides adequate precision for this demonstration.
    w = weights.detach().to(torch.float32)
    x = inputs.detach().to(torch.float32)

    best_result = None
    best_error = float("inf")

    for index in range(num_candidates):
        # Includes alpha=0, the unscaled baseline.
        alpha = index / num_candidates

        scales = activation_scales(x, alpha)

        codes, effective = reconstruct_with_scales(
            weights=w,
            channel_scales=scales,
            bits=bits,
            group_size=group_size,
        )

        error = reconstruction_mse(x, w, effective)

        if error < best_error:
            best_error = error

            best_result = AWQResult(
                alpha=alpha,
                channel_scales=scales.clone(),
                integer_codes=codes.clone(),
                effective_weights=effective.clone(),
                output_mse=error,
            )

    if best_result is None:
        raise RuntimeError("AWQ search produced no valid candidates.")

    return best_result


def main() -> None:
    """Compare regular INT4 quantization and activation-aware INT4."""
    torch.manual_seed(42)

    input_features = 32
    output_features = 16
    samples = 512
    bits = 4
    group_size = 8

    # Simulate weights from a linear projection.
    weights = torch.randn(
        output_features,
        input_features,
        dtype=torch.float32,
    ) * 0.15

    # Construct calibration inputs with unequal feature magnitudes.
    inputs = torch.randn(
        samples,
        input_features,
        dtype=torch.float32,
    )

    # Make a small subset of input channels more active.
    inputs[:, [3, 11, 19]] *= 20.0

    # Ordinary group-wise INT4 quantization.
    _, plain_weights = groupwise_quantize(
        weights,
        bits=bits,
        group_size=group_size,
    )

    plain_mse = reconstruction_mse(
        inputs,
        weights,
        plain_weights,
    )

    # Activation-aware scaling and candidate selection.
    awq = search_awq_scales(
        weights,
        inputs,
        bits=bits,
        group_size=group_size,
        num_candidates=20,
    )

    print("Ordinary INT4 output MSE:", f"{plain_mse:.8f}")
    print("AWQ INT4 output MSE:     ", f"{awq.output_mse:.8f}")
    print("Best AWQ alpha:          ", awq.alpha)

    if plain_mse > 0:
        improvement = 100.0 * (
            1.0 - awq.output_mse / plain_mse
        )
        print("MSE reduction (%):       ", f"{improvement:.2f}")

    # Verify that activation/weight scaling is mathematically
    # equivalent before introducing quantization.
    scaled_inputs = inputs / awq.channel_scales
    scaled_weights = weights * awq.channel_scales.unsqueeze(0)

    original_output = inputs @ weights.T
    transformed_output = scaled_inputs @ scaled_weights.T

    assert torch.allclose(
        original_output,
        transformed_output,
        atol=2e-3,
        rtol=1e-5,
    ), "Equivalent transformation check failed."

    # Verify the resulting low-bit representation.
    assert awq.integer_codes.min().item() >= -7
    assert awq.integer_codes.max().item() <= 7
    assert torch.isfinite(awq.effective_weights).all()

    print("All checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"AWQ experiment failed: {exc}") from exc

```

### Understanding what the implementation is doing

The first important function is `activation_scales`. It calculates the average absolute activation magnitude for each input feature and raises that magnitude to a candidate exponent `alpha`.

The second is `reconstruct_with_scales`. It multiplies the weight columns by these activation-aware scales, applies group-wise quantization, and then divides the reconstructed weights by the same scales.

This produces the effective weight matrix that would be observed when the inverse scaling is applied to activations.

The third is `search_awq_scales`. It evaluates multiple scaling candidates against the original layer's output and keeps the candidate with the lowest reconstruction MSE.

The candidate `alpha = 0` represents the unscaled baseline, so the search always has access to ordinary group-wise quantization. It may select a different alpha when the calibration activations show that rescaling improves output fidelity.

This simple example does not implement architectural scale folding, optional clipping optimization, packed INT4 storage, or optimized CUDA kernels. Those are separate steps required for a deployable AWQ pipeline.

## 9. AWQ, scaling, and quantization granularity

An important connection to the earlier granularity discussion is that AWQ's channel scaling and its group-wise quantization are two different layers of numerical control.

The activation-aware scale vector `S` has one value per input feature. For a weight matrix of shape `[4096, 4096]`, it can contain 4,096 scaling factors.

Separately, the quantized weight representation may use groups of 128 weights along each output row.

Each group has its own quantization scale and possibly a zero-point, depending on the format.

The activation-aware scaling factors may be folded into preceding network operations, while the group quantization parameters remain part of the compressed weight representation.

Consider a simplified group:

```
Weights before AWQ:

[0.80, 0.05, 0.30, 0.20]

Activation-aware scales:

[1.0, 10.0, 1.0, 1.0]

Scaled weights:

[0.80, 0.50, 0.30, 0.20]
```

The group quantizer now operates on the transformed values.

The important weight, originally 0.05, occupies a larger fraction of the group's numerical range, and therefore can receive a more useful quantized representation.

The corresponding activation is divided by 10 to preserve the original function.

This interaction between channel rescaling and group-wise quantization is the main mechanism behind AWQ.

## 10. Where AWQ fits in production inference

In production, AWQ is normally applied offline. Calibration samples are collected, suitable channel scales are searched, and those scales are folded into compatible network operators. The transformed weights are then quantized and packed into a supported low-bit format.

At runtime, the inference engine loads the quantized checkpoint and executes matrix multiplications using kernels designed for the storage layout.

The offline search is not repeated for every token.

For an illustrative eight-billion-parameter model, FP16 weights require approximately 16 GB, while raw four-bit weights require approximately 4 GB, excluding scales, zero-points, unquantized parameters, and runtime allocations.

This weight-memory reduction can be particularly useful during autoregressive decoding, where repeatedly reading model weights from GPU memory may limit performance.

However, lower memory use does not automatically produce lower latency. The runtime kernel must efficiently unpack and consume the quantized weights. Performance should be measured across real request batch sizes, prefill lengths, and decode lengths.

The original AWQ project also introduced TinyChat, an inference framework with low-bit kernels and platform-aware weight packing. Its published speedups are specific to the paper's hardware and baselines, not guaranteed improvements for every deployment.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 11. Limitations and engineering considerations

AWQ does not eliminate quantization error. Its scaling transformation redistributes precision across channels, which means protecting one channel can sometimes make another channel less accurately represented.

The search depends on representative calibration activations. If the deployment workload differs substantially from calibration, the selected scales may be less effective.

AWQ also needs architecture-aware implementation. Moving scales between operators is only valid when the resulting computation remains equivalent. Residual branches, shared activations, normalization layouts, nonlinear functions, and tensor-parallel partitioning can make scale folding more complicated.

Numerical stability matters as well. Very small activation magnitudes can produce near-zero scales, while highly skewed distributions can produce large scale ratios. Practical implementations clamp and normalize candidate scales and verify that transformed tensors remain finite.

For a production deployment, evaluate output reconstruction error, model-level benchmark quality, latency, throughput, GPU memory, and kernel compatibility. Record the calibration configuration, chosen quantization format, group size, and model version to make results reproducible.

## 12. Final comparison: What makes AWQ different?

The essence of the two major methods can be summarized through their core operations.

GPTQ:

```
Original weights
      |
      v
Calibration Hessian
      |
      v
Quantize current column
      |
      v
Calculate quantization error
      |
      v
Compensate remaining columns
      |
      v
Low-bit weights
```

AWQ:

```
Original weights + calibration activations
      |
      v
Measure activation magnitudes
      |
      v
Search channel scaling factors
      |
      v
Scale weights and inversely scale activations
      |
      v
Apply group-wise quantization
      |
      v
Fold scales into compatible operators
      |
      v
Low-bit weights
```

GPTQ is primarily based on error compensation after quantization decisions.

AWQ is primarily based on making important weight channels easier to quantize before making those decisions.

The most important AWQ equation to remember is:

`Y ≈ (X / S) × Q(W × S).T`

The transformation is exact before quantization. Its purpose is to change the distribution of quantization error after low-bit rounding, giving more protection to activation-sensitive weight channels.

For deeper implementation study, the authoritative resources are the [AWQ research paper](https://arxiv.org/abs/2306.00978), the [official AWQ repository](https://github.com/mit-han-lab/llm-awq), and its [activation-aware scale-search implementation](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py).

The next important concept is SmoothQuant, which also uses equivalent transformations but targets the difficulty of quantizing activation outliers in weight-and-activation schemes such as W8A8, rather than focusing primarily on low-bit weight-only quantization.
