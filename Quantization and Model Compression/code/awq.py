
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
