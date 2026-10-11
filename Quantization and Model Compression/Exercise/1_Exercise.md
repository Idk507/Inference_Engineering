"""
Exercise 1: By hand (NumPy, no library), implement symmetric per-tensor quantization and per-channel quantization for a random weight matrix with a few artificial outlier values injected. Measure and compare reconstruction error (mean squared error between original and dequantized) for both granularities. Confirm per-channel wins, and explain why using your own numbers.
"""

# Exercise 1: Symmetric Per-Tensor vs. Per-Channel Quantization Using NumPy

We will implement both quantization strategies from scratch using NumPy, inject artificial outliers into a weight matrix, and measure reconstruction error using Mean Squared Error (MSE).

The objective is to demonstrate why per-channel quantization usually preserves weights more accurately when different channels have different numerical ranges.

## 1. Experimental setup

We will use a weight matrix with 8 output channels and 64 input features.

`W.shape = (8, 64)`

Most weights will be randomly sampled from a normal distribution with mean 0 and standard deviation 0.06.

To simulate outliers, we will manually inject three unusually large values into different output channels:

```
W[0, 3] = 9.0W[3, 20] = -6.0W[6, 42] = 4.5
```

We will use symmetric INT8 quantization, meaning integer values range from -127 to +127 with a zero-point of 0.

The quantization equations are:

`scale = max(abs(W)) / 127`

`q = clip(round(W / scale), -127, 127)`

`W_reconstructed = q × scale`

For per-tensor quantization, `max(abs(W))` is calculated across the entire matrix. For per-channel quantization, it is calculated independently for each row.

## 2. Complete NumPy implementation

```
     
"""
Exercise 1: Symmetric Per-Tensor vs Per-Channel Quantization.

Requirements:
    pip install numpy

Demonstrates:
    - INT8 symmetric affine quantization
    - Global and row-wise scale calculation
    - Artificial outlier injection
    - Quantization and dequantization
    - Mean Squared Error comparison
    - Quantization-induced zero values

No quantization library is used.
"""

import numpy as np


def quantize_symmetric(
    weights: np.ndarray,
    per_channel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize a 2D weight matrix using symmetric INT8.

    Args:
        weights: Floating-point matrix [out_channels, in_features].
        per_channel: If True, use one scale per output row.
                     Otherwise, use one scale for the entire tensor.

    Returns:
        quantized: INT8 matrix with the original shape.
        reconstructed: Dequantized floating-point matrix.
        scales: Global scale [1,1] or per-channel scales [rows,1].

    Raises:
        TypeError: If input is not a NumPy array or is not floating-point.
        ValueError: If weights are empty, not 2D, or contain nonfinite values.
    """
    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a NumPy array.")

    if weights.ndim != 2 or weights.size == 0:
        raise ValueError("Expected a nonempty 2D weight matrix.")

    if not np.issubdtype(weights.dtype, np.floating):
        raise TypeError("Weight matrix must contain floating-point values.")

    if not np.isfinite(weights).all():
        raise ValueError("Weight matrix contains NaN or infinity.")

    # Signed symmetric INT8 uses [-127, +127].
    qmin = -127
    qmax = 127

    # STEP 1: Determine the calibration range.
    #
    # Per-tensor:
    #     Maximum absolute value across the whole matrix.
    #
    # Per-channel:
    #     Maximum absolute value in each output row.
    #
    # keepdims=True makes NumPy broadcasting straightforward.
    if per_channel:
        max_abs = np.max(
            np.abs(weights),
            axis=1,
            keepdims=True,
        )
    else:
        max_abs = np.max(
            np.abs(weights),
            keepdims=True,
        )

    # STEP 2: Calculate quantization scales.
    scales = max_abs / qmax

    # A completely zero-valued region requires a safe scale.
    scales = np.where(scales == 0, 1.0, scales)

    # STEP 3: Convert floating-point values to integer codes.
    #
    # np.rint performs nearest-even rounding for exact ties.
    integer_values = np.rint(weights / scales)

    # STEP 4: Clamp values to the signed INT8 range.
    integer_values = np.clip(
        integer_values,
        qmin,
        qmax,
    )

    quantized = integer_values.astype(np.int8)

    # STEP 5: Dequantize using the corresponding scales.
    reconstructed = quantized.astype(np.float64) * scales

    return quantized, reconstructed, scales


def calculate_mse(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """
    Calculate Mean Squared Error between two matrices.

    Formula:
        MSE = mean((original - reconstructed) ** 2)
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Input matrices must have identical shapes.")

    return float(np.mean((original - reconstructed) ** 2))


def main() -> None:
    """Create an outlier-heavy matrix and compare both methods."""
    # Fixed random seed makes the experiment reproducible.
    rng = np.random.default_rng(42)

    # Generate mostly small weights.
    weights = rng.normal(
        loc=0.0,
        scale=0.06,
        size=(8, 64),
    ).astype(np.float64)

    # Inject three large outliers.
    weights[0, 3] = 9.0
    weights[3, 20] = -6.0
    weights[6, 42] = 4.5

    # EXPERIMENT A: One scale for the entire tensor.
    qt, reconstructed_tensor, tensor_scale = quantize_symmetric(
        weights,
        per_channel=False,
    )

    # EXPERIMENT B: Independent scale for every output channel.
    qc, reconstructed_channel, channel_scales = quantize_symmetric(
        weights,
        per_channel=True,
    )

    # Calculate reconstruction errors.
    tensor_mse = calculate_mse(
        weights,
        reconstructed_tensor,
    )

    channel_mse = calculate_mse(
        weights,
        reconstructed_channel,
    )

    improvement = (
        1.0 - channel_mse / tensor_mse
    ) * 100.0 if tensor_mse > 0 else 0.0

    # Count originally nonzero values rounded to integer zero.
    tensor_zeroed = np.count_nonzero(
        (qt == 0) & (weights != 0)
    )

    channel_zeroed = np.count_nonzero(
        (qc == 0) & (weights != 0)
    )

    print("=" * 60)
    print("SYMMETRIC INT8 QUANTIZATION COMPARISON")
    print("=" * 60)

    print(f"Weight matrix shape: {weights.shape}")
    print(f"Number of parameters: {weights.size}")

    print("\nPer-tensor scale:")
    print(tensor_scale)

    print("\nPer-channel scales:")
    print(channel_scales.flatten())

    print("\nReconstruction errors:")
    print(f"Per-tensor MSE  : {tensor_mse:.10f}")
    print(f"Per-channel MSE : {channel_mse:.10f}")
    print(f"MSE reduction   : {improvement:.2f}%")

    print("\nOriginally nonzero weights rounded to zero:")
    print(f"Per-tensor  : {tensor_zeroed}")
    print(f"Per-channel : {channel_zeroed}")

    print("\nPer-channel reconstruction analysis:")
    tensor_row_mse = np.mean(
        (weights - reconstructed_tensor) ** 2,
        axis=1,
    )
    channel_row_mse = np.mean(
        (weights - reconstructed_channel) ** 2,
        axis=1,
    )

    for row in range(weights.shape[0]):
        print(
            f"Channel {row}: "
            f"Tensor MSE={tensor_row_mse[row]:.8f}, "
            f"Channel MSE={channel_row_mse[row]:.8f}"
        )

    # Assertions for this deterministic experiment.
    assert qt.dtype == np.int8
    assert qc.dtype == np.int8
    assert qt.shape == weights.shape
    assert qc.shape == weights.shape
    assert channel_mse < tensor_mse

    print("\nAll assertions passed.")


if __name__ == "__main__":
    try:
        main()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Experiment failed: {exc}") from exc

```

## 3. Actual experimental results

I executed the experiment with NumPy using seed 42 and the specified outliers.

<img width="478" height="487" alt="image" src="https://github.com/user-attachments/assets/66fd2759-eda7-4707-a517-7582e081a219" />


The exact numerical measurements are:

| Metric                         | Per-tensor   | Per-channel  |
| ------------------------------ | ------------ | ------------ |
| MSE                            | 0.0004126533 | 0.0000934218 |
| Number of scales               | 1            | 8            |
| Nonzero values rounded to zero | 241          | 69           |

The per-channel approach produced approximately 4.42 times lower reconstruction MSE than per-tensor quantization.

### 3.1 Why did it win?

The largest outlier in the entire matrix is 9.0.

Therefore, the per-tensor scale is:

`scale_tensor = 9 / 127 ≈ 0.07086614`

Every weight, even those with magnitudes near 0.01, must use this large scale.

Consider a typical small weight:

`w = 0.02`

Using per-tensor quantization:

`q = round(0.02 / 0.07086614) = 0`

The reconstructed value is:

`w_hat = 0`

The entire value has been lost.

However, consider Channel 1, which does not contain an artificial outlier. Its maximum magnitude in this experiment produces a scale of approximately:

`scale_channel_1 = 0.00081606`

Now:

`q = round(0.02 / 0.00081606) ≈ 25`

The reconstructed value is:

`w_hat ≈ 25 × 0.00081606 = 0.0204015`

The reconstruction is much closer to the original 0.02.

The important difference is that Channel 1 does not have to share the scale created by the extreme outlier in Channel 0.

## 4. Comparing individual channel scales

Here are the actual scales calculated from the generated matrix.

| Channel | Scale      | Artificial outlier |
| ------- | ---------- | ------------------ |
| 0       | 0.07086614 | +9.0               |
| 1       | 0.00081606 | None               |
| 2       | 0.00137663 | None               |
| 3       | 0.04724409 | -6.0               |
| 4       | 0.00121259 | None               |
| 5       | 0.00109968 | None               |
| 6       | 0.03543307 | +4.5               |
| 7       | 0.00140056 | None               |

Channels 0, 3, and 6 have relatively large scales because we deliberately inserted extreme values.

The remaining five channels have much smaller scales, typically around 0.001.

Per-channel quantization allows those five channels to use finer numerical grids.

Per-tensor quantization forces every channel to use the coarse 0.070866 scale.

This also explains why the number of values rounded to zero decreases from 241 to 69.

## 5. A mathematical explanation of the result

For symmetric INT8 with round-to-nearest, the rounding error of an unclipped value is bounded by approximately:

`absolute_error ≤ scale / 2`

For per-tensor quantization:

`maximum rounding error ≤ 0.07086614 / 2 ≈ 0.035433`

For Channel 1 under per-channel quantization:

`maximum rounding error ≤ 0.00081606 / 2 ≈ 0.000408`

So Channel 1's maximum rounding-error bound becomes roughly 87 times smaller.

Under a simplified uniform rounding-noise model:

`expected squared error ≈ scale² / 12`

This explains why even a modest scale reduction can produce a substantial MSE reduction.

One qualification is important: per-channel quantization is not mathematically guaranteed to produce lower actual MSE for every possible weight matrix when scales are selected using a simple maximum-absolute-value heuristic. Smaller regions give greater representational flexibility and tighter local error bounds, but changing a quantization grid can occasionally increase individual rounding errors.

For the weight matrix in this exercise, however, the improvement is explicitly verified.

## 6. Final conclusion

Our experiment confirms that per-channel symmetric INT8 quantization performs significantly better on this outlier-heavy weight matrix.

The per-tensor MSE was 0.00041265, while the per-channel MSE was 0.00009342, representing a 77.36% reduction in reconstruction error.

The improvement occurs because per-channel quantization prevents a large outlier in one output channel from determining the numerical precision available to every other channel.

The trade-off is additional metadata: per-tensor quantization requires one scale, while per-channel quantization requires one scale per output channel. In practice, that modest storage increase is often worthwhile for neural-network weight matrices.

This exercise demonstrates the foundation behind more advanced quantization techniques. GPTQ builds upon the idea that output sensitivity matters, AWQ uses activation information to redistribute numerical precision, and group-wise quantization further localizes the range assigned to each quantization scale.
