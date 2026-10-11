
"""
Quantization-Aware Training (QAT) from scratch with PyTorch.

Demonstrates:
    - FP32 baseline training
    - Symmetric INT4 weight fake quantization
    - Custom clipped Straight-Through Estimator (STE)
    - QAT fine-tuning
    - Post-training quantization comparison
    - Held-out evaluation and assertions

This is an educational, weight-only QAT implementation.

Note:
    This code simulates quantization numerics. It does not pack
    weights into a physical INT4 format or implement an INT4 kernel.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class FakeQuantizeSTE(torch.autograd.Function):
    """Simulate integer quantization using a clipped STE."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        scales: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        """
        Perform quantize-dequantize in the forward pass.

        Floating-point output:
            q = clamp(round(x / scale), qmin, qmax)
            x_hat = q * scale
        """
        if torch.any(scales <= 0).item():
            raise ValueError("Quantization scales must be positive.")

        normalized = inputs / scales

        # Preserve the non-saturated gradient region for backward.
        active = (normalized >= qmin) & (normalized <= qmax)
        ctx.save_for_backward(active)

        quantized = torch.round(normalized).clamp(qmin, qmax)

        return quantized * scales

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        Apply a clipped identity-gradient approximation.

        Inside the quantization range, gradients pass unchanged.
        Outside the range, gradients are blocked.

        Scales are treated as nondifferentiable in this example.
        """
        (active,) = ctx.saved_tensors

        grad_inputs = grad_output * active.to(grad_output.dtype)

        return grad_inputs, None, None, None


class QATLinear(nn.Module):
    """Linear layer with symmetric per-output-channel fake quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
    ) -> None:
        """Initialize floating-point parameters and quantization settings."""
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("Layer dimensions must be positive.")

        if not isinstance(bits, int) or not 2 <= bits <= 8:
            raise ValueError("bits must be between 2 and 8.")

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bits = bits
        self.fake_quant_enabled = True

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def quantized_weight(self) -> torch.Tensor:
        """
        Calculate the fake-quantized representation of master weights.

        Per-output-channel scaling gives each row an independent
        quantization range.
        """
        qmax = (1 << (self.bits - 1)) - 1
        qmin = -qmax

        # Detach scale estimation in this simple QAT implementation.
        # Only floating-point master weights receive gradients.
        with torch.no_grad():
            max_abs = self.weight.detach().abs().amax(
                dim=1,
                keepdim=True,
            )

            scales = torch.where(
                max_abs > 0,
                max_abs / qmax,
                torch.ones_like(max_abs),
            )

        return FakeQuantizeSTE.apply(
            self.weight,
            scales,
            qmin,
            qmax,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer with optional fake weight quantization."""
        weights = (
            self.quantized_weight()
            if self.fake_quant_enabled
            else self.weight
        )

        return F.linear(inputs, weights, self.bias)


class TinyNetwork(nn.Module):
    """Small MLP for comparing ordinary and quantization-aware training."""

    def __init__(self, qat: bool = False) -> None:
        """Construct a two-layer neural network."""
        super().__init__()

        layer_type = QATLinear if qat else nn.Linear

        self.fc1 = layer_type(16, 64)
        self.fc2 = layer_type(64, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply hidden projection, ReLU, and output projection."""
        hidden = F.relu(self.fc1(inputs))
        return self.fc2(hidden)


def make_dataset(
    count: int,
    input_dim: int,
    teacher: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate supervised examples using a fixed nonlinear teacher.

    Labels are derived from the teacher network and do not require
    downloading external datasets.
    """
    if count <= 0 or input_dim <= 0:
        raise ValueError("Dataset dimensions must be positive.")

    inputs = torch.randn(count, input_dim)

    with torch.no_grad():
        labels = teacher(inputs).argmax(dim=1)

    return inputs, labels


def train_model(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> None:
    """Train a model using cross-entropy and the AdamW optimizer."""
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("Training configuration must be positive.")

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        predictions = model(inputs)
        loss = F.cross_entropy(predictions, labels)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite loss encountered.")

        loss.backward()

        # Bound gradient norms for numerical robustness.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Loss: {loss.item():.6f}"
            )


def evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Return classification accuracy on evaluation data."""
    model.eval()

    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)

    return (predictions == labels).float().mean().item()


def copy_to_qat(
    original: TinyNetwork,
) -> TinyNetwork:
    """Copy pretrained floating-point weights into a QAT model."""
    target = TinyNetwork(qat=True)

    with torch.no_grad():
        for layer_name in ("fc1", "fc2"):
            source_layer = getattr(original, layer_name)
            target_layer = getattr(target, layer_name)

            target_layer.weight.copy_(source_layer.weight)
            target_layer.bias.copy_(source_layer.bias)

    return target


def run_experiment() -> None:
    """Compare floating-point, PTQ, and QAT models."""
    torch.manual_seed(42)

    # Teacher supplies nonlinear targets.
    teacher = TinyNetwork(qat=False)
    teacher.eval()

    train_x, train_y = make_dataset(2048, 16, teacher)
    test_x, test_y = make_dataset(512, 16, teacher)

    # STEP 1: Train a high-precision baseline.
    print("\nTraining floating-point baseline")

    fp_model = TinyNetwork(qat=False)

    train_model(
        fp_model,
        train_x,
        train_y,
        epochs=100,
        learning_rate=0.01,
    )

    fp_accuracy = evaluate(fp_model, test_x, test_y)

    # STEP 2: PTQ comparison.
    #
    # Copy the trained model into fake-quantized layers.
    # Do not fine-tune these weights.
    ptq_model = copy_to_qat(fp_model)
    ptq_accuracy = evaluate(ptq_model, test_x, test_y)

    # STEP 3: QAT starts from the same pretrained weights.
    qat_model = deepcopy(ptq_model)

    print("\nQuantization-aware fine-tuning")

    train_model(
        qat_model,
        train_x,
        train_y,
        epochs=60,
        learning_rate=0.001,
    )

    qat_accuracy = evaluate(qat_model, test_x, test_y)

    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)

    print(f"FP32 accuracy:      {fp_accuracy:.4f}")
    print(f"INT4 PTQ accuracy:  {ptq_accuracy:.4f}")
    print(f"INT4 QAT accuracy:  {qat_accuracy:.4f}")

    # Verify that fake-quantized training remains numerically valid.
    for name, parameter in qat_model.named_parameters():
        assert torch.isfinite(parameter).all(), (
            f"Nonfinite parameter: {name}"
        )

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    try:
        run_experiment()
    except (ValueError, TypeError, RuntimeError) as exc:
        raise SystemExit(f"QAT experiment failed: {exc}") from exc
