
"""
GPTQ From Scratch
=================

Educational implementation of GPTQ (Generative Pre-trained
Transformer Quantization) using PyTorch.

This example demonstrates:

1. Creating a small neural-network weight matrix.
2. Generating representative calibration activations.
3. Computing the input-space Hessian approximation.
4. Applying diagonal damping for numerical stability.
5. Computing the Cholesky factor of the inverse Hessian.
6. Quantizing weights column by column.
7. Compensating future columns for quantization errors.
8. Comparing GPTQ against ordinary Round-to-Nearest (RTN).

This is an educational reference implementation, not an
optimized production GPTQ inference kernel.

The INT4 codes are held in int8 tensors for convenience.
They are not physically packed into four-bit storage.
"""

from dataclasses import dataclass

import torch


@dataclass
class QuantizationResult:
    """Store quantized weights and evaluation information."""

    integer_weights: torch.Tensor
    dequantized_weights: torch.Tensor
    scales: torch.Tensor
    output_mse: float


def build_calibration_data(
    input_features: int,
    num_samples: int,
    latent_features: int = 4,
) -> torch.Tensor:
    """
    Generate correlated calibration activations.

    GPTQ uses calibration activations to estimate how sensitive
    a linear layer is to changes in its weights.

    Args:
        input_features: Number of input features.
        num_samples: Number of calibration examples.
        latent_features: Number of shared latent features.

    Returns:
        Calibration matrix with shape [input_features, num_samples].
    """
    if min(input_features, num_samples, latent_features) <= 0:
        raise ValueError("All dimensions must be positive.")

    # Latent features introduce correlations between input channels.
    latent = torch.randn(
        latent_features,
        num_samples,
        dtype=torch.float64,
    )

    mixing = torch.randn(
        input_features,
        latent_features,
        dtype=torch.float64,
    )

    # X = mixing @ latent creates correlated input features.
    x = mixing @ latent

    # Small noise prevents input channels from being exactly dependent.
    x += 0.05 * torch.randn_like(x)

    return x


def calculate_scales(
    weights: torch.Tensor,
    bits: int = 4,
) -> torch.Tensor:
    """
    Calculate one symmetric quantization scale per output channel.

    For signed symmetric INT4 we use integer values [-7, 7].

    Formula:
        scale = max(abs(weights_in_row)) / 7

    Args:
        weights: Matrix [output_features, input_features].
        bits: Number of quantization bits.

    Returns:
        Scales of shape [output_features, 1].
    """
    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError("bits must be an integer between 2 and 8.")

    if weights.ndim != 2 or weights.numel() == 0:
        raise ValueError("weights must be a nonempty 2D matrix.")

    qmax = (1 << (bits - 1)) - 1

    # Find the largest absolute weight in every output row.
    max_values = weights.abs().amax(dim=1, keepdim=True)

    # Prevent division by zero for completely zero-valued rows.
    scales = torch.where(
        max_values > 0,
        max_values / qmax,
        torch.ones_like(max_values),
    )

    return scales


def quantize_column(
    values: torch.Tensor,
    scales: torch.Tensor,
    bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize one weight column using a fixed scale per row.

    Formula:
        q = clamp(round(w / scale), -Q, Q)
        w_hat = q * scale

    Args:
        values: Current column [output_features, 1].
        scales: Per-output-channel scales [output_features, 1].
        bits: Number of quantization bits.

    Returns:
        Tuple containing integer codes and reconstructed weights.
    """
    qmax = (1 << (bits - 1)) - 1

    # Floating-point weights -> low-bit integer codes.
    codes = torch.round(values / scales)
    codes = codes.clamp(-qmax, qmax).to(torch.int8)

    # Reconstruct approximate floating-point weights.
    reconstructed = codes.to(values.dtype) * scales

    return codes, reconstructed


def build_hessian(
    calibration_inputs: torch.Tensor,
    damping_fraction: float = 0.01,
) -> torch.Tensor:
    """
    Build and stabilize GPTQ's input-space Hessian approximation.

    For a linear layer, the Hessian of squared output error
    is proportional to X @ X.T.

    H = (2 / N) * X @ X.T

    Damping:
        H_regularized = H + lambda * I

    Args:
        calibration_inputs: Matrix [input_features, num_samples].
        damping_fraction: Fraction of average Hessian diagonal.

    Returns:
        Regularized Hessian matrix [input_features, input_features].
    """
    if calibration_inputs.ndim != 2:
        raise ValueError("Calibration inputs must be 2D.")

    if calibration_inputs.numel() == 0:
        raise ValueError("Calibration inputs cannot be empty.")

    if not torch.isfinite(calibration_inputs).all().item():
        raise ValueError("Calibration inputs contain nonfinite values.")

    if damping_fraction <= 0:
        raise ValueError("damping_fraction must be positive.")

    num_samples = calibration_inputs.shape[1]
    num_features = calibration_inputs.shape[0]

    # Hessian derived from calibration activation correlations.
    hessian = (
        2.0 / num_samples
    ) * (calibration_inputs @ calibration_inputs.T)

    # Compute damping relative to average diagonal curvature.
    damping = damping_fraction * hessian.diagonal().mean()

    if damping.item() <= 0:
        raise ValueError("Calibration data has no usable signal.")

    identity = torch.eye(
        num_features,
        dtype=hessian.dtype,
        device=hessian.device,
    )

    # Improve conditioning before Cholesky factorization.
    hessian = hessian + damping * identity

    return hessian


def calculate_inverse_hessian_factor(
    hessian: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the upper-triangular factor used in GPTQ updates.

    We seek U such that:
        inverse(H) = U.T @ U

    Args:
        hessian: Symmetric positive-definite matrix.

    Returns:
        Upper-triangular matrix U.

    Raises:
        RuntimeError: If matrix factorization fails.
    """
    try:
        # First Cholesky factorization of regularized H.
        lower_h = torch.linalg.cholesky(hessian)

        # Solve for the inverse using the Cholesky factor.
        inverse_h = torch.cholesky_inverse(lower_h)

        # Factor the inverse:
        # inverse_h = L @ L.T
        lower_inverse = torch.linalg.cholesky(inverse_h)

        # Convert to the desired upper-triangular factor:
        # U.T @ U = inverse_h
        #
        # IMPORTANT:
        # Directly transposing lower_inverse does NOT produce
        # U.T @ U = inverse_h in general.
        #
        # torch.linalg.cholesky(...).mH gives a factor U where
        # inverse_h = U.T @ U only when using upper=True.
        upper = torch.linalg.cholesky(
            inverse_h,
            upper=True,
        )

        return upper

    except torch.linalg.LinAlgError as error:
        raise RuntimeError(
            "Hessian factorization failed. Increase damping."
        ) from error


def evaluate_output_mse(
    original_weights: torch.Tensor,
    quantized_weights: torch.Tensor,
    calibration_inputs: torch.Tensor,
) -> float:
    """
    Evaluate how much quantization changes layer outputs.

    Original:
        Y = W @ X

    Quantized:
        Y_hat = W_hat @ X

    Returns:
        Mean squared difference between Y and Y_hat.
    """
    original_output = original_weights @ calibration_inputs
    quantized_output = quantized_weights @ calibration_inputs

    mse = torch.mean(
        (original_output - quantized_output).square()
    )

    return mse.item()


def round_to_nearest(
    weights: torch.Tensor,
    calibration_inputs: torch.Tensor,
    bits: int = 4,
) -> QuantizationResult:
    """
    Apply ordinary symmetric quantization without compensation.

    Each weight is independently mapped to its nearest
    representable integer using its output-channel scale.
    """
    scales = calculate_scales(weights, bits)

    codes, reconstructed = quantize_column(
        weights,
        scales,
        bits,
    )

    mse = evaluate_output_mse(
        weights,
        reconstructed,
        calibration_inputs,
    )

    return QuantizationResult(
        integer_weights=codes,
        dequantized_weights=reconstructed,
        scales=scales,
        output_mse=mse,
    )


def gptq_quantize(
    weights: torch.Tensor,
    calibration_inputs: torch.Tensor,
    bits: int = 4,
    damping_fraction: float = 0.01,
    verbose: bool = True,
) -> QuantizationResult:
    """
    Quantize a weight matrix using sequential GPTQ compensation.

    Algorithm:

    1. Compute the Hessian from calibration inputs.
    2. Stabilize it using diagonal damping.
    3. Compute the upper Cholesky factor U of inverse(H).
    4. Iterate over input-feature columns.
    5. Quantize the current column.
    6. Measure its signed quantization error.
    7. Update all remaining unquantized columns.
    8. Return low-bit codes and reconstructed weights.

    Note:
        This version does not use blocked lazy updates,
        activation ordering, or group-wise scales.
    """
    if weights.ndim != 2 or calibration_inputs.ndim != 2:
        raise ValueError("Weights and calibration inputs must be 2D.")

    if weights.shape[1] != calibration_inputs.shape[0]:
        raise ValueError(
            "Weights input dimension must match calibration features."
        )

    if not weights.is_floating_point():
        raise TypeError("Weights must be floating-point.")

    if not torch.isfinite(weights).all().item():
        raise ValueError("Weights contain nonfinite values.")

    if weights.device != calibration_inputs.device:
        raise ValueError("Weights and calibration inputs must share a device.")

    # FP64 reduces numerical error in this teaching implementation.
    original = weights.detach().to(torch.float64)
    inputs = calibration_inputs.detach().to(torch.float64)

    output_features, input_features = original.shape

    # Step 1: Calculate scales from the original weights.
    # They remain fixed throughout this example.
    scales = calculate_scales(original, bits)

    # Step 2: Build a Hessian from calibration activations.
    hessian = build_hessian(inputs, damping_fraction)

    # Step 3: Calculate the factor used for error compensation.
    upper = calculate_inverse_hessian_factor(hessian)

    # Step 4: Create a mutable working copy of the weights.
    working = original.clone()

    # Integer codes are stored separately from reconstructed weights.
    integer_weights = torch.empty(
        original.shape,
        dtype=torch.int8,
        device=original.device,
    )
    reconstructed = torch.empty_like(original)

    # Step 5: Process input-feature columns sequentially.
    for column_index in range(input_features):

        # Extract the current, possibly already adjusted column.
        current = working[:, column_index:column_index + 1]

        # Quantize this column using the established INT4 grid.
        codes, quantized = quantize_column(
            current,
            scales,
            bits,
        )

        # Save its final quantized representation.
        integer_weights[
            :, column_index:column_index + 1
        ] = codes

        reconstructed[
            :, column_index:column_index + 1
        ] = quantized

        # Step 6: Compute the quantization error.
        #
        # GPTQ divides this error by the corresponding
        # diagonal entry of the inverse-Hessian factor.
        diagonal = upper[column_index, column_index]

        if diagonal.item() <= 0:
            raise RuntimeError("Invalid Cholesky diagonal.")

        error = (current - quantized) / diagonal

        # Step 7: Compensate all remaining free columns.
        #
        # The Cholesky row tells us how the quantization
        # error should be distributed over future weights.
        #
        # Previously quantized columns remain untouched.
        correction = (
            error
            * upper[
                column_index,
                column_index:
            ].unsqueeze(0)
        )

        working[:, column_index:] -= correction

        if verbose:
            print(
                f"Column {column_index:02d} | "
                f"Mean abs quantization error: "
                f"{(current - quantized).abs().mean().item():.6f}"
            )

    # Step 8: Evaluate output reconstruction error.
    mse = evaluate_output_mse(
        original,
        reconstructed,
        inputs,
    )

    return QuantizationResult(
        integer_weights=integer_weights,
        dequantized_weights=reconstructed,
        scales=scales,
        output_mse=mse,
    )


def main() -> None:
    """Run a reproducible RTN versus GPTQ experiment."""
    torch.manual_seed(42)

    output_features = 8
    input_features = 16
    calibration_samples = 256

    # Create a small floating-point weight matrix.
    weights = torch.randn(
        output_features,
        input_features,
        dtype=torch.float64,
    ) * 0.3

    # Generate calibration activations with correlated features.
    calibration_inputs = build_calibration_data(
        input_features=input_features,
        num_samples=calibration_samples,
        latent_features=4,
    )

    # Ordinary quantization without error compensation.
    rtn_result = round_to_nearest(
        weights,
        calibration_inputs,
        bits=4,
    )

    print("\nRunning GPTQ quantization:\n")

    # GPTQ with Hessian-based error compensation.
    gptq_result = gptq_quantize(
        weights,
        calibration_inputs,
        bits=4,
        damping_fraction=0.01,
        verbose=True,
    )

    print("\n" + "=" * 55)
    print("QUANTIZATION COMPARISON")
    print("=" * 55)

    print(f"Original weight shape: {tuple(weights.shape)}")
    print(f"Quantization bit width: 4")
    print(f"RTN output MSE : {rtn_result.output_mse:.8f}")
    print(f"GPTQ output MSE: {gptq_result.output_mse:.8f}")

    if rtn_result.output_mse > 0:
        improvement = (
            1.0
            - gptq_result.output_mse / rtn_result.output_mse
        ) * 100

        print(f"Relative improvement: {improvement:.2f}%")

    print("\nOriginal weights (first row):")
    print(weights[0])

    print("\nRTN reconstructed weights (first row):")
    print(rtn_result.dequantized_weights[0])

    print("\nGPTQ reconstructed weights (first row):")
    print(gptq_result.dequantized_weights[0])

    # Verify that logical INT4 codes are within [-7, 7].
    assert gptq_result.integer_weights.min().item() >= -7
    assert gptq_result.integer_weights.max().item() <= 7
    assert torch.isfinite(gptq_result.dequantized_weights).all()

    print("\nAll basic validation checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Experiment failed: {exc}") from exc
