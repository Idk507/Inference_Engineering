# Quantization-Aware Training (QAT): From First Principles to Production LLM Inference

Quantization-Aware Training (QAT) is a technique for training or fine-tuning a neural network while deliberately simulating the numerical errors that will occur when the model is eventually deployed in a low-precision format such as INT8 or INT4.

The central idea is straightforward: instead of training a model in high precision and introducing quantization errors only afterward, we expose the model to those errors during training so that its parameters learn to tolerate them.

This makes QAT fundamentally different from post-training quantization (PTQ), including the GPTQ, AWQ, and QwT approaches we have discussed.

The crucial mechanism is fake quantization: forward computation imitates quantization and dequantization, while training generally retains floating-point parameters and gradients. PyTorch's TorchAO documentation describes this prepare–train–convert workflow for modern QAT, including low-bit LLM configurations.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

torchao 0.17 documentation

+1



We will examine the complete system, including the mathematical foundation, fake-quantization operations, straight-through estimators, training stability, learnable scaling, LLM-specific configurations, Python implementation, and actual deployment considerations.

## 1. Why do we need Quantization-Aware Training?

Imagine a neural network trained using FP32 parameters.

During training, its optimizer continuously adjusts weights using gradients:

`W_next = W_current - learning_rate × gradient`

Because the weights are floating-point numbers, the optimizer can make very small adjustments.

For example, a parameter might evolve like this:

```
Step 1: 0.123456
Step 2: 0.123412
Step 3: 0.123387
Step 4: 0.123371
```

Now suppose the trained model must run on hardware that favors INT4 weights.

Signed symmetric INT4 has only a small number of representable integer levels. Under the commonly used symmetric range -7 through +7, the weights must be approximated by just 15 numerical levels for each scale.

If a weight is originally 0.123371, its nearest representable value might be 0.10 or 0.15, depending on the scale.

Every quantized weight changes slightly, and these errors can accumulate across the many matrix multiplications in a transformer.

The problem is especially challenging at four bits or below because the quantization grid is extremely coarse.

### 1.1 What PTQ does

In Post-Training Quantization, we train a model normally.

```
FP32/BF16 training
       |
       v
Trained model
       |
       v
Quantization algorithm
       |
       v
INT8 / INT4 model
```

The model only encounters quantization after its training process is complete.

GPTQ uses input curvature and sequential error compensation. AWQ uses activation-aware scaling. QwT uses output compensation.

All of them attempt to reduce quantization degradation without performing conventional full-model quantization-aware training.

### 1.2 What QAT changes

QAT moves the quantization simulation inside the training process.

```
Initial FP32/BF16 weights
          |
          v
Insert fake quantization
          |
          v
Forward pass with quantization noise
          |
          v
Calculate task loss
          |
          v
Backpropagate gradients
          |
          v
Update floating-point weights
          |
          v
Repeat
          |
          v
Convert to deployment quantization format
```

The optimizer can now discover parameter values that perform well even after being rounded to the target quantization grid.

For example, rather than learning a weight of 0.123371, the training process may move it toward a representable value such as 0.125, provided that doing so improves the model's actual task objective.

This does not mean every weight necessarily converges exactly to an integer-grid point. Instead, the entire network learns to operate under simulated quantization constraints.

<img width="475" height="238" alt="image" src="https://github.com/user-attachments/assets/8e14b3cf-6913-424a-989c-e8f1e4162959" />


## 2. First principles: The quantization equation inside QAT

Recall the affine quantization equation:

`q = clamp(round(x / scale) + zero_point, q_min, q_max)`

The dequantization equation is:

`x_hat = (q - zero_point) × scale`

For symmetric quantization, the zero-point is normally zero:

`q = clamp(round(x / scale), -Q, Q)`

`x_hat = q × scale`

During ordinary quantized inference, the integer codes may be physically packed into INT8 or INT4 representations.

During QAT, however, we generally keep computation in a floating-point representation but force tensor values through a quantization-and-dequantization simulation.

This operation is called fake quantization.

### 2.1 Fake quantization

Suppose:

```
Original weight = 0.37
Scale = 0.10
Zero-point = 0
```

Quantization gives:

`q = round(0.37 / 0.10) = 4`

Dequantization gives:

`x_hat = 4 × 0.10 = 0.40`

The fake-quantized value is therefore 0.40.

But here is the important distinction:

Real quantization

## 4

Integer code stored in a low-bit format

Fake quantization

## 0.40

Reconstructed value retained as floating point

Fake quantization simulates the numerical distortion without requiring the whole training computation to use low-bit integer arithmetic.

That allows standard automatic differentiation and floating-point optimizers to continue operating.

However, there is a mathematical problem.

## 3. The fundamental difficulty: Rounding is not differentiable

A neural network learns through backpropagation.

To update a weight, the optimizer needs the gradient of the loss with respect to that weight.

For a simplified linear operation:

`y = w × x`

the derivative is:

`dy/dw = x`

This is straightforward.

But suppose the weight is fake-quantized:

`w_q = round(w / scale) × scale`

Now:

`y = w_q × x`

The derivative of the rounding function is zero almost everywhere and undefined at its discontinuities.

Conceptually, the quantization function looks like a staircase.

A uniform quantizer creates a staircase function

<img width="537" height="296" alt="image" src="https://github.com/user-attachments/assets/a8c269f8-dc42-417f-861b-f671a41c3154" />


If we use the true derivative of rounding, gradients through the quantizer become zero almost everywhere.

The optimizer would be unable to make useful weight updates through that operation.

QAT solves this with an approximation called the Straight-Through Estimator (STE).

## 4. Straight-Through Estimator: The mechanism that makes QAT trainable

The Straight-Through Estimator uses different behavior during forward and backward computation.

In the forward pass, it performs real rounding and clipping numerically.

In the backward pass, it approximates the derivative of quantization, commonly as an identity function for values inside the representable range.

The conceptual approximation is:

`d(fake_quantize(x))/dx ≈ 1`

for values within the active quantization range.

For values outside that range, a clipped STE commonly uses a gradient of zero.

### 4.1 A simple example

Suppose:

`w = 0.37`

`scale = 0.10`

Forward quantization produces:

`w_q = 0.40`

Then:

`y = w_q × x`

If:

`x = 2`

the output is:

`y = 0.8`

Suppose the loss gradient arriving from later layers is:

`dL/dy = 3`

Then:

`dL/dw_q = 3 × 2 = 6`

The STE passes an approximate gradient of 6 back to the underlying floating-point parameter `w`.

The optimizer can therefore update the original floating-point weight even though the forward computation used its quantized approximation.

This is not the true mathematical derivative of the rounding operation. It is a surrogate gradient chosen because it enables useful optimization.

### 4.2 Implementing STE in PyTorch

A common PyTorch technique uses `.detach()`:

```
def fake_quantize_ste(x, scale):    """Fake quantize a tensor using an identity-gradient STE."""    quantized = torch.round(x / scale) * scale    # Forward result equals quantized.    # Backward gradient with respect to x is approximately identity.    return x + (quantized - x).detach()
```

Why does this work?

The expression:

`(quantized - x).detach()`

is treated as a constant during backpropagation.

Therefore the gradient flows only through the first `x`.

But the forward value is:

`x + quantized - x = quantized`

So we get the best of both behaviors for this simplified quantizer.

However, this implementation has no clipping and does not block gradients for out-of-range inputs. Production fake-quantization operations often implement the clipping mask explicitly.

The STE makes QAT possible, but it also introduces a train–deploy approximation: training relies on surrogate gradients, while deployment uses actual quantized arithmetic.

## 5. Complete QAT training lifecycle

A practical QAT workflow usually begins with a pretrained floating-point model.

Training from scratch with fake quantization is possible, but fine-tuning an existing model is often more economical, especially for large transformers.

### Phase 1: Establish a floating-point baseline

We begin by measuring the pretrained model's task accuracy, loss, and inference behavior.

This establishes the reference against which quantization degradation will be measured.

### Phase 2: Select the quantization configuration

Before QAT starts, we must define the target numerical representation.

For example:

```
Weight precision:       INT4
Activation precision:   INT8
Weight granularity:     Per-group, size 128
Activation granularity: Per-token
Weight zero-point:      0 (symmetric)
Training compute:       BF16/FP32
```

This choice is critical because the training simulation should match the quantization behavior of the actual deployment kernel.

Training with one quantization granularity and deploying with another can introduce numerical mismatches.

### Phase 3: Insert fake-quantization operations

The original linear layer:

`Y = X × W.T + b`

becomes conceptually:

`Y = FakeQuant(X) × FakeQuant(W).T + b`

The training graph now simulates quantization noise for weights and, when configured, activations.

### Phase 4: Fine-tune the model

Each training iteration performs fake-quantized forward computation, calculates the task loss, backpropagates surrogate gradients, and updates the floating-point parameters.

The optimizer does not normally update packed four-bit codes directly.

Instead, it updates floating-point master weights whose quantized versions are used during forward computation.

### Phase 5: Stabilize the quantization parameters

Activation ranges may change substantially during early training.

If the quantizer continuously recalculates its scales, the numerical grid can move while the model is adapting to it.

A common strategy is to collect activation statistics initially and freeze observer updates later, while allowing the model weights to continue learning.

Not every QAT scheme uses observers. Dynamically quantized activations may calculate their scales from each current input instead.

### Phase 6: Convert and deploy

After training, the floating-point master weights are converted into actual low-bit integer codes.

The resulting model must be lowered to an inference backend supporting the intended quantized operations.

PyTorch's current TorchAO workflow explicitly separates preparation, fake-quantized training, and conversion.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

<img width="477" height="413" alt="image" src="https://github.com/user-attachments/assets/302ae380-979f-47b9-959b-8ebd06b226a6" />


## 6. Scale selection during QAT

QAT can use the same granularity options we discussed earlier: per-tensor, per-channel, per-group, and per-token.

The scale determines how values map onto the quantization grid.

For symmetric quantization, a common simple choice is:

`scale = max(abs(x)) / Q`

where Q is the maximum positive integer magnitude.

But calculating this from the current tensor every iteration has consequences.

If a single outlier grows, the scale increases. That can reduce numerical precision for all other values sharing the same quantization region.

Training can adapt weights to such changes, but repeatedly changing the quantization grid can also destabilize optimization.

Different QAT methods address this using observers, clipping, frozen calibration statistics, dynamic scaling, or learnable quantization parameters.

### 6.1 Fixed versus dynamic scales

Fixed-scale QAT establishes quantization parameters and then holds them constant during some or all training.

This is useful when deployment uses fixed quantization scales.

Dynamic-scale QAT calculates quantization parameters from the current activations or weights during forward computation.

This may more closely resemble a dynamically quantized inference kernel.

For example, per-token activation quantization can determine the activation scale separately for each token's hidden-state vector.

The correct choice depends on the deployed backend, not merely on which scheme produces lower training loss.

### 6.2 Learnable Step Size Quantization

A more advanced approach is Learned Step Size Quantization (LSQ).

Instead of treating the quantization scale only as a statistic derived from the tensor, LSQ treats it as a parameter optimized through training.

The forward equation is still:

`x_hat = scale × clamp(round(x / scale), q_min, q_max)`

But the scale receives an approximate gradient.

This allows the optimizer to balance rounding precision against clipping error.

A very small scale creates a fine numerical grid but clips large values.

A large scale covers more of the value distribution but increases rounding error.

LSQ uses training to learn a useful trade-off.

The scale gradient is also approximated because the quantizer includes rounding. LSQ uses gradient scaling to stabilize optimization across tensors with different sizes and integer ranges.

In practice, learnable scales introduce another source of optimization instability, so learning rates, scale initialization, clipping, and numerical precision must be handled carefully.

## 7. Why QAT matters for transformer models

Transformers contain repeated linear projections, attention operations, and feed-forward networks.

During inference, quantization errors can affect each of these operations.

For example, attention scores are calculated as:

`Scores = Q × K.T / sqrt(head_dim)`

If the Q and K projection weights are quantized, their outputs change slightly.

Those changes propagate into attention scores and subsequently into the softmax operation.

Because softmax is nonlinear, a modest difference in attention scores can change the attention distribution.

The resulting difference propagates through value aggregation and subsequent transformer layers.

With PTQ, the model has no opportunity to adapt its parameters to those errors through ordinary training.

With QAT, simulated quantization errors are present while the task loss is being optimized. The model can adjust multiple interacting layers together.

This can be particularly valuable for aggressive low-bit quantization.

PyTorch reported experiments where QAT recovered substantial degradation relative to PTQ for Llama 3 under an INT8 dynamic-activation and grouped INT4-weight configuration. These are benchmark-specific findings, not guarantees for arbitrary LLMs.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)





### 7.1 QAT does not necessarily reduce training memory

One common misunderstanding is that training with INT4 fake quantization automatically requires four bits per parameter.

In ordinary QAT, floating-point master parameters, gradients, and optimizer state still exist.

The fake-quantized values are used to simulate deployment behavior, not necessarily to reduce the physical memory required for training.

For an optimizer such as AdamW, optimizer states can consume substantial additional memory.

Consequently, QAT may require similar or greater training resources than ordinary fine-tuning.

This is distinct from QLoRA, which keeps a quantized pretrained backbone and trains low-rank adapters.

QAT and LoRA can also be combined, but their purposes are different: QAT prepares a model to tolerate quantization, while LoRA reduces the number of trainable parameters.

## 8. Complete PyTorch implementation: QAT from scratch

We will now implement a small neural network that learns a nonlinear function from synthetic data.

We will compare three approaches: a floating-point model, ordinary INT4 PTQ applied after training, and QAT with simulated INT4 weights during fine-tuning.

This implementation includes a custom autograd function for the Straight-Through Estimator, symmetric per-output-channel fake quantization, floating-point master weights, gradient-based optimization, and an evaluation against held-out test data.

It is deliberately small enough to run locally on CPU or GPU without requiring a pretrained LLM.

```
"""Quantization-Aware Training (QAT) from scratch with PyTorch.Demonstrates:    - FP32 baseline training    - Symmetric INT4 weight fake quantization    - Custom clipped Straight-Through Estimator (STE)    - QAT fine-tuning    - Post-training quantization comparison    - Held-out evaluation and assertionsThis is an educational, weight-only QAT implementation.Note:    This code simulates quantization numerics. It does not pack    weights into a physical INT4 format or implement an INT4 kernel."""from copy import deepcopyimport torchfrom torch import nnfrom torch.nn import functional as Fclass FakeQuantizeSTE(torch.autograd.Function):    """Simulate integer quantization using a clipped STE."""    @staticmethod    def forward(        ctx,        inputs: torch.Tensor,        scales: torch.Tensor,        qmin: int,        qmax: int,
```

This experiment is designed to demonstrate the mechanics rather than guarantee that QAT improves classification accuracy for every random initialization and dataset. For some cases, the initial INT4 approximation may already be adequate; in others, QAT may recover accuracy lost during PTQ.

For a rigorous comparison, repeat the experiment with several random seeds and record both the mean and variability of test accuracy.

### Understanding the critical implementation details

The most important component is `FakeQuantizeSTE`.

Its forward method maps floating-point weights onto the quantization grid and reconstructs them in floating point.

Its backward method allows gradients to flow through values inside the permitted quantization range.

`QATLinear` owns floating-point master weights. On every forward pass, it computes a fake-quantized representation and uses that representation for matrix multiplication.

When the optimizer updates the layer, it changes the master weights, not the stored integer codes.

The current example recomputes per-channel scales from the master weights at each forward pass. Since scale estimation is detached from autograd, the implementation ignores derivatives through the scale calculation. This is a useful introductory approximation, but it differs from learnable-scale schemes such as LSQ.

The implementation also quantizes only the weights. It does not simulate activation quantization. A W8A8 QAT model would additionally insert fake quantizers on suitable activation paths.

## 9. How this maps to modern PyTorch TorchAO

For production-oriented implementations, PyTorch's TorchAO is generally a better starting point than maintaining custom STE kernels.

As of the current TorchAO documentation, the recommended module-oriented QAT interface uses `QATConfig` together with `quantize_()`, while graph-oriented PT2E QAT is available for suitable exported models and backends.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

torchao 0.17 documentation

+1



The high-level flow looks like this:

```
"""Illustrative TorchAO QAT configuration.Requires compatible versions of PyTorch and torchao.The selected deployment backend must support the target format."""import torchfrom torchao.quantization import (    quantize_,    Int8DynamicActivationIntxWeightConfig,    PerGroup,)from torchao.quantization.qat import QATConfigdef prepare_qat_model(model: torch.nn.Module) -> torch.nn.Module:    """Prepare supported linear layers for fake-quantized training."""    base_config = Int8DynamicActivationIntxWeightConfig(        weight_dtype=torch.int4,        weight_granularity=PerGroup(32),    )    quantize_(        model,        QATConfig(            base_config,            step="prepare",        ),    )    return modeldef convert_qat_model(model: torch.nn.Module) -> torch.nn.Module:
```

The intended sequence is to construct the model, prepare it for QAT, fine-tune it using a normal optimizer and loss function, and then convert it.

Conversion alone does not guarantee faster inference. The runtime must provide an efficient kernel for the converted representation.

For a working framework example, see the [TorchAO QAT workflow](https://docs.pytorch.org/ao/stable/workflows/qat.html) and [TorchAO fine-tuning tutorial](https://docs.pytorch.org/ao/stable/eager_tutorials/finetuning.html).

## 10. QAT versus GPTQ, AWQ, QwT, and QLoRA

| Technique | Core idea                                       | Training/fine-tuning                 | Typical purpose                   |
| --------- | ----------------------------------------------- | ------------------------------------ | --------------------------------- |
| GPTQ      | Hessian-guided weight error compensation        | No conventional gradient fine-tuning | Low-bit weight PTQ                |
| AWQ       | Activation-aware channel rescaling              | No conventional gradient fine-tuning | Protect salient weights           |
| QwT       | Fit residual output compensation                | Closed-form fitting                  | Recover quantized output quality  |
| QAT       | Learn under fake quantization                   | Yes                                  | Adapt model to low-bit deployment |
| QLoRA     | Train adapters over a quantized frozen backbone | Yes, adapters                        | Memory-efficient adaptation       |

The particularly important distinction is between QAT and QLoRA.

QLoRA uses low-bit storage primarily to reduce fine-tuning memory while updating a relatively small number of adapter parameters.

QAT simulates the target low-bit inference numerics during training so that the eventual quantized model suffers less degradation.

These methods can be combined. TorchAO documents QAT integrations with parameter-efficient fine-tuning workflows.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

torchao 0.17 documentation

+1



## 11. Production engineering: Accuracy, security, scalability, and observability

A successful QAT experiment does not end when training loss decreases.

The target checkpoint must be converted and tested using the actual inference backend. The quantized kernel should reproduce the fake-quantized behavior closely enough that a significant new accuracy gap does not appear after conversion.

Measure that gap explicitly by comparing the QAT-prepared model's outputs with the deployed quantized model's outputs on the same held-out inputs.

For large LLMs, full-model QAT can be expensive because master weights, gradients, optimizer states, activations, and fake-quantization operations contribute to training memory. Distributed training, activation checkpointing, low-rank adaptation, and careful batch-size selection can reduce these costs.

Security controls should cover calibration and training datasets, checkpoint integrity, access permissions, and any sensitive inputs appearing in logs. Quantization should never require indiscriminate retention of private production prompts.

Observability should include gradient norms, scale ranges, clipping or saturation rates, loss curves, validation perplexity, task-level accuracy, quantized-versus-reference logit differences, and inference latency.

A particularly useful metric is the quantization gap:

`quantization_gap = metric(FP_reference) - metric(quantized_model)`

For metrics where higher is better, such as accuracy, a positive value indicates degradation. For lower-is-better metrics such as perplexity or error, reverse the comparison or report the values separately.

Tracking this gap throughout QAT gives a direct view of whether the training process is successfully adapting the model to its target quantization configuration.

## 12. Final mental model

GPTQ, AWQ, and QAT address quantization error at different stages.

GPTQ adjusts unquantized weights to compensate for rounding errors during post-training quantization.

AWQ transforms weight and activation scales to make important weight channels easier to represent.

QAT changes the training process itself, allowing the model to learn parameters while repeatedly experiencing quantization noise.

The three most important equations are:

Fake quantization:

`x_fake = scale × clamp(round(x / scale) + zero_point, q_min, q_max) - scale × zero_point`

Straight-Through Estimator:

`d(fake_quantize(x))/dx ≈ 1` inside the active quantization range.

QAT optimization:

`minimize task_loss(model_using_fake_quantized_weights_and_activations)`

The fundamental idea is that QAT teaches a model to function under quantization before it encounters the final low-bit inference representation.

For LLM inference engineering, this becomes especially relevant when post-training methods cannot satisfy the required model-quality threshold at INT4 or lower precision. In those scenarios, QAT offers a way to recover accuracy, provided the additional training cost and deployment-kernel constraints are justified.
