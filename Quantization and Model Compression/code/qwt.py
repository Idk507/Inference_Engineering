
"""
QwT-v2: Channel-Wise Affine Compensation From Scratch.

This example demonstrates:
    1. Quantizing a linear-layer weight matrix.
    2. Collecting full-precision and quantized outputs.
    3. Solving channel-wise affine compensation analytically.
    4. Evaluating the results on unseen test inputs.

Requires:
    pip install torch
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CompensationParameters:
    """Store the affine compensation parameters for output channels."""

    alpha: torch.Tensor
    beta: torch.Tensor


def symmetric_weight_quantization(
    weights: torch.Tensor,
    bits: int = 4,
) -> torch.Tensor:
    """
    Simulate per-output-channel symmetric weight quantization.

    Args:
        weights: Weight matrix [out_features, in_features].
        bits: Signed quantization precision from 2 to 8.

    Returns:
        Dequantized floating-point approximation of the weights.
    """
    if weights.ndim != 2 or not weights.is_floating_point():
        raise ValueError("Weights must be a floating-point 2D matrix.")

    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError("bits must be between 2 and 8.")

    if not torch.isfinite(weights).all().item():
        raise ValueError("Weights must contain finite values.")

    qmax = (1 << (bits - 1)) - 1

    # Each output row receives its own symmetric scale.
    maximum = weights.abs().amax(dim=1, keepdim=True)

    # Protect entirely zero-valued rows from division by zero.
    scale = torch.where(
        maximum > 0,
        maximum / qmax,
        torch.ones_like(maximum),
    )

    codes = torch.round(weights / scale).clamp(-qmax, qmax)

    # Simulate dequantization for the reference experiment.
    return codes * scale


def fit_channel_compensation(
    full_outputs: torch.Tensor,
    quantized_outputs: torch.Tensor,
) -> CompensationParameters:
    """
    Fit one affine transformation per output channel.

    For each channel c, solve:
        full[:, c] ~= alpha[c] * quantized[:, c] + beta[c]

    Closed-form solution:
        alpha = covariance(quantized, full) / variance(quantized)
        beta = mean(full) - alpha * mean(quantized)

    Args:
        full_outputs: Full-precision outputs [N, channels].
        quantized_outputs: Quantized outputs [N, channels].

    Returns:
        CompensationParameters containing alpha and beta.
    """
    if full_outputs.shape != quantized_outputs.shape:
        raise ValueError("Output shapes must match.")

    if full_outputs.ndim != 2 or full_outputs.shape[0] < 2:
        raise ValueError("Expected at least two 2D calibration samples.")

    if not torch.isfinite(full_outputs).all().item():
        raise ValueError("Full outputs contain nonfinite values.")

    if not torch.isfinite(quantized_outputs).all().item():
        raise ValueError("Quantized outputs contain nonfinite values.")

    # Channel-wise means.
    mean_full = full_outputs.mean(dim=0)
    mean_quant = quantized_outputs.mean(dim=0)

    # Center each output channel.
    centered_full = full_outputs - mean_full
    centered_quant = quantized_outputs - mean_quant

    # Covariance and variance with the same normalization.
    covariance = (
        centered_quant * centered_full
    ).mean(dim=0)

    variance = centered_quant.square().mean(dim=0)

    # Handle constant or effectively constant quantized channels.
    # In this case the least-squares optimum is a constant output,
    # represented by alpha=0 and beta=mean(full).
    tolerance = torch.finfo(variance.dtype).eps
    valid = variance > tolerance

    alpha = torch.where(
        valid,
        covariance / variance.clamp_min(tolerance),
        torch.zeros_like(variance),
    )

    beta = mean_full - alpha * mean_quant

    return CompensationParameters(alpha=alpha, beta=beta)


def apply_compensation(
    quantized_outputs: torch.Tensor,
    parameters: CompensationParameters,
) -> torch.Tensor:
    """
    Apply a learned channel-wise affine correction.

    Formula:
        Y_corrected = alpha * Y_quantized + beta
    """
    if quantized_outputs.shape[-1] != parameters.alpha.numel():
        raise ValueError("Output channel dimensions do not match.")

    return (
        quantized_outputs * parameters.alpha
        + parameters.beta
    )


def mse(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    """Return mean squared reconstruction error."""
    if reference.shape != prediction.shape:
        raise ValueError("MSE inputs must have identical shapes.")

    return torch.mean((reference - prediction).square()).item()


def main() -> None:
    """Run the QwT-v2 numerical demonstration."""
    torch.manual_seed(42)

    input_features = 64
    output_features = 32

    calibration_samples = 1024
    test_samples = 256

    # Create a synthetic pretrained linear layer.
    weights = torch.randn(
        output_features,
        input_features,
        dtype=torch.float64,
    ) * 0.2

    bias = torch.randn(
        output_features,
        dtype=torch.float64,
    ) * 0.05

    # Quantize its weights to simulated INT4.
    quantized_weights = symmetric_weight_quantization(
        weights,
        bits=4,
    )

    # Generate independent calibration and test inputs.
    x_cal = torch.randn(
        calibration_samples,
        input_features,
        dtype=torch.float64,
    )

    x_test = torch.randn(
        test_samples,
        input_features,
        dtype=torch.float64,
    )

    # Step 1: Collect calibration outputs.
    full_cal = F.linear(x_cal, weights, bias)
    quant_cal = F.linear(x_cal, quantized_weights, bias)

    # Step 2: Calculate closed-form affine compensation.
    parameters = fit_channel_compensation(
        full_cal,
        quant_cal,
    )

    # Step 3: Evaluate on independent test data.
    full_test = F.linear(x_test, weights, bias)
    quant_test = F.linear(x_test, quantized_weights, bias)

    corrected_test = apply_compensation(
        quant_test,
        parameters,
    )

    before = mse(full_test, quant_test)
    after = mse(full_test, corrected_test)

    print("QwT-v2 reference experiment")
    print("-" * 45)
    print(f"Test MSE before compensation: {before:.8f}")
    print(f"Test MSE after compensation:  {after:.8f}")

    if before > 0:
        reduction = (1 - after / before) * 100
        print(f"Relative MSE reduction:     {reduction:.2f}%")

    print(f"Number of alpha values: {parameters.alpha.numel()}")
    print(f"Number of beta values:  {parameters.beta.numel()}")

    # Validate output shapes and numerical integrity.
    assert corrected_test.shape == full_test.shape
    assert torch.isfinite(corrected_test).all()
    assert torch.isfinite(parameters.alpha).all()
    assert torch.isfinite(parameters.beta).all()


if __name__ == "__main__":
    try:
        main()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"QwT-v2 experiment failed: {exc}") from exc
