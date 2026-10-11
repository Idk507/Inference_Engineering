# GPTQ Explained: Hessian-Based, Column-by-Column Weight Quantization and Error Compensation

GPTQ, short for Generative Pre-trained Transformer Quantization, is one of the foundational algorithms behind low-bit LLM inference. Its central contribution is not simply converting FP16 weights into INT4. It is finding a better way to make that conversion by accounting for how quantization errors affect the output of a neural network.

Consider a transformer containing billions of floating-point parameters. If we convert each weight independently to its nearest four-bit value, we save memory, but we also change the model's mathematical behavior. A small error in one weight can affect the output of a linear layer, alter attention scores, change intermediate hidden states, and ultimately influence which token the model generates.

Ordinary round-to-nearest quantization ignores these interactions.

GPTQ takes a different approach: after quantizing a weight, it adjusts the remaining unquantized weights to compensate for the error introduced by that decision.

The compensation is guided by an approximate Hessian matrix, which captures how combinations of input features influence the reconstruction error of a layer.

The original GPTQ paper, GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, was published by Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh and appeared at ICLR 2023. It demonstrated that very large transformer models could be quantized to three or four bits per weight while retaining much of their original quality.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)





We will examine how the algorithm works, beginning with ordinary matrix multiplication, and progressively derive the Hessian, inverse-Hessian compensation rule, Cholesky factorization, column-wise processing, blocked GPU implementation, and production inference architecture.

## 1. The fundamental problem GPTQ attempts to solve

A transformer performs an enormous number of matrix multiplications. One typical linear layer computes:

`Y = X × W.T`

Here, `X` contains input activations, `W` contains learned weights, and `Y` contains the resulting output activations.

Suppose a linear layer has the following weight matrix:

```
W = [
    [0.12, 0.38, 0.72],
    [0.21, 0.56, 0.91]
]
```

And its input activation is:

```
X = [2.0, 4.0, 1.0]
```

The first output neuron produces:

```
y1 = (2.0 × 0.12)
   + (4.0 × 0.38)
   + (1.0 × 0.72)

   = 0.24 + 1.52 + 0.72

   = 2.48
```

Now suppose we use a simplified quantizer that rounds each weight to the nearest multiple of 0.25.

Our quantized weights become approximately:

```
Wq = [
    [0.00, 0.50, 0.75],
    [0.25, 0.50, 1.00]
]
```

The first output now becomes:

```
y1_quantized = (2.0 × 0.00)
             + (4.0 × 0.50)
             + (1.0 × 0.75)

             = 2.75
```

The output error is:

`2.75 - 2.48 = 0.27`

Notice that this is not the same as looking at the error of any single weight.

Every weight error is multiplied by its corresponding input activation. Consequently, the effect of quantization depends on the input distribution.

Ordinary quantization attempts to minimize the distance between weights and their quantized representations.

GPTQ is designed around a more useful target: minimizing the difference between the outputs of the original layer and the quantized layer on representative inputs.

This is the first major conceptual shift.

## 2. The GPTQ objective: Preserve layer outputs, not just weights

Let a linear layer contain a weight matrix:

`W.shape = [out_features, in_features]`

For calibration, collect `N` representative input vectors and arrange them into a matrix:

`X.shape = [in_features, N]`

The original layer outputs are:

`Y = W × X`

The quantized layer outputs are:

`Yq = Wq × X`

The reconstruction objective is:

`L(Wq) = ||W × X - Wq × X||²_F`

The notation `||...||²_F` means the sum of squared elements of a matrix.

This equation can be rewritten as:

`L(Wq) = ||(W - Wq) × X||²_F`

Define the quantization error matrix:

`E = W - Wq`

Then:

`L = ||E × X||²_F`

This tells us something extremely important.

A quantization error is costly when it causes substantial changes to the outputs under the calibration inputs. It may be less costly when other weights can compensate for it, or when it occurs along input directions with little activity.

Therefore, two quantized models with similar weight MSE can have different output reconstruction errors.

GPTQ uses the geometry of this objective to decide how the remaining weights should change.

### Weight error versus output error
<img width="506" height="227" alt="image" src="https://github.com/user-attachments/assets/e7b88baa-4f01-4bc8-a203-975c60ad1e49" />

## 3. Why does GPTQ need a Hessian?

The Hessian is a mathematical object used to describe the curvature of a function.

To understand why it matters, begin with a simple one-dimensional loss:

`L(w) = (w - 2)²`

This loss is minimized at:

`w = 2`

The first derivative is:

`dL/dw = 2(w - 2)`

The second derivative is:

`d²L/dw² = 2`

The first derivative tells us the direction in which the loss increases or decreases. The second derivative tells us how sharply the loss curves.

For a multidimensional function, the Hessian generalizes the second derivative into a matrix.

Suppose we have two weights, `w1` and `w2`.

Their Hessian looks like:

```
H = [
    [ d²L/dw1²,       d²L/dw1dw2 ],
    [ d²L/dw2dw1,     d²L/dw2²   ]
]
```

The diagonal entries describe curvature associated with each weight coordinate.

The off-diagonal entries describe interactions between coordinates.

Those interactions matter for quantization because changing one weight may be partly offset by changing another weight.

### 3.1 Deriving the Hessian for a linear layer

Consider one row of the weight matrix, represented as a column vector `w`.

The calibration inputs are collected in `X`.

The original output is:

`y = X.T × w`

Suppose we perturb the weights by `delta`.

The new weights are:

`w_new = w + delta`

The output changes by:

`delta_y = X.T × delta`

The squared output reconstruction loss is:

`L(delta) = ||X.T × delta||²`

Expanding:

`L(delta) = delta.T × X × X.T × delta`

This is a quadratic function.

Its gradient is:

`gradient = 2 × X × X.T × delta`

And its Hessian is:

`H = 2 × X × X.T`

That is the key formula behind GPTQ.

For a matrix with 4,096 input features:

```
X.shape = [4096, N]

H.shape = [4096, 4096]
```

The Hessian depends on the calibration activations, not directly on the values of the weights.

This is an important efficiency advantage.

If a linear layer has 4,096 output neurons, all those output rows can reuse the same input-space Hessian.

Without that property, calculating independent second-order information for millions of weights would become prohibitively expensive.

In practice, an overall positive normalization factor can be applied to `X × X.T` without changing the ideal compensation direction. Implementations often normalize the accumulated activation statistics and add damping for stability.

### 3.2 What does the Hessian actually capture?

Consider two input features, `x1` and `x2`.

If the features are strongly correlated, changing the weight associated with `x1` can potentially be offset by changing the weight associated with `x2`.

For example, suppose the input data usually satisfies:

`x2 ≈ x1`

The layer computes:

`y = w1 × x1 + w2 × x2`

If we increase `w1` slightly and decrease `w2` by a similar amount, the output may remain approximately unchanged.

Why?

Because:

`delta_y = delta_w1 × x1 + delta_w2 × x2`

And if `x1 ≈ x2`, then choosing:

`delta_w2 ≈ -delta_w1`

makes the two changes cancel.

This is precisely the kind of dependency GPTQ exploits.

The Hessian's off-diagonal entries help describe the statistical relationships between input features.

A diagonal-only approximation would miss these interactions.

## 4. The inverse Hessian: The mathematical foundation of compensation

We now know that GPTQ builds a matrix of second-order information from calibration activations.

But the algorithm needs something more specific: it must determine how to adjust the remaining weights after a particular weight is forced to a quantized value.

This is where the inverse Hessian appears.

Suppose we want to quantize weight `w_j`.

The chosen quantized value is `q_j`.

The weight change is:

`delta_j = q_j - w_j`

GPTQ wants to make this change while minimizing the corresponding increase in the quadratic reconstruction objective.

Conceptually, we are solving:

`minimize 0.5 × delta.T × H × delta`

subject to:

`delta[j] = q_j - w_j`

The factor 0.5 is conventional and does not change the minimizer.

The constraint means that weight `j` must be fixed to its quantized value. All remaining unquantized weights are allowed to move.

Solving this constrained quadratic optimization produces the Optimal Brain Surgeon-style update:

`delta = (q_j - w_j) × H_inv[:, j] / H_inv[j, j]`

Here:

`H_inv = inverse(H)`

More precisely, the inverse is taken over the still-unquantized, or free, coordinates. Once a weight has been quantized, it is no longer allowed to change.

The additional quadratic loss associated with fixing coordinate `j` is proportional to:

`(w_j - q_j)² / H_inv[j, j]`

With the 0.5 quadratic-loss convention, the value is:

`0.5 × (w_j - q_j)² / H_inv[j, j]`

This is not just a mathematical detail. It reveals the role of curvature in quantization.

The cost of quantizing a weight depends both on its raw rounding error and on how easily other available weights can compensate for that error.

A larger value of `H_inv[j,j]`, all else equal, corresponds to a smaller optimal quadratic loss increase for that fixed weight error.

GPTQ therefore uses the inverse Hessian to identify the proper compensation direction rather than adjusting every remaining weight arbitrarily.

### 4.1 Why not simply subtract the error from the next weight?

Suppose the quantization error of one weight is `0.1`.

A naïve idea would be to subtract `0.1` from the next weight.

But what if those two input features behave independently? Their effects on the output would not cancel reliably.

What if the features are negatively correlated? The appropriate correction may even have the opposite sign.

What if the second feature is rarely active? A larger adjustment may be required to compensate for the same output change.

The inverse Hessian incorporates this local quadratic structure. It tells the algorithm how to distribute the correction among the remaining free coordinates to minimize the increase in the reconstruction loss.

## 5. A complete numerical example of Hessian-based compensation

Let's work through a small example using only two weights.

Assume one output neuron has:

```
w = [0.60, 0.24]
```

We use a simplified quantization grid with representable values:

```
..., 0.0, 0.5, 1.0, ...
```

Suppose the curvature matrix, ignoring an irrelevant common normalization factor, is:

```
H = [
    [2, 1],
    [1, 2]
]
```

Its inverse is:

```
H_inv = [
    [ 2/3, -1/3],
    [-1/3,  2/3]
]
```

We quantize the first weight:

`w1 = 0.60`

Its nearest grid value is:

`q1 = 0.50`

Therefore, the signed difference between the original and quantized value is:

`w1 - q1 = 0.10`

The inverse-Hessian diagonal for the first coordinate is:

`H_inv[1,1] = 2/3`

Using the zero-based indexing equivalent in code, the normalized error is:

`e = 0.10 / (2/3) = 0.15`

The inverse-Hessian interaction with the second coordinate is:

`H_inv[1,2] = -1/3`

Therefore the compensation update for the second weight is:

`w2_new = w2 - e × H_inv[1,2]`

Substituting:

`w2_new = 0.24 - 0.15 × (-1/3)`

`w2_new = 0.29`

The remaining full-precision weight has moved from 0.24 to 0.29.

When it is eventually rounded to our grid, it becomes 0.5 instead of 0.0.

Without compensation, ordinary rounding gives:

```
Original weights:
[0.60, 0.24]

Round-to-nearest:
[0.50, 0.00]
```

With compensation:

```
Original weights:
[0.60, 0.24]

Quantize first coordinate:
[0.50, 0.24]

Compensate remaining coordinate:
[0.50, 0.29]

Quantize second coordinate:
[0.50, 0.50]
```

Let's calculate which result is better using the Hessian-weighted squared reconstruction error.

Define:

`error_vector = original_weights - quantized_weights`

The quadratic reconstruction measure is:

`error_vector.T × H × error_vector`

For ordinary round-to-nearest, the error vector is:

`[0.10, 0.24]`

The quadratic error is:

`0.1832`

For the compensated result, the error vector is:

`[0.10, -0.26]`

The quadratic error is:

`0.1032`

Ordinary rounding

# 0.1832

Quadratic output reconstruction error

Error compensation

# 0.1032

Quadratic output reconstruction error

In this example, sequential compensation reduces the quadratic reconstruction error by approximately 43.7%.

The precise result depends on the calibration statistics, the quantization grid, and the quantization order. GPTQ is not guaranteed to improve every individual weight or every possible layer configuration, but the example illustrates the mechanism.

The key insight is that the quantized weights need not be individually closest to the original weights to produce a better approximation of the layer's outputs.

## 6. From Optimal Brain Quantization to GPTQ

The mathematics we just derived comes from a broader family of second-order optimization methods, particularly Optimal Brain Surgeon and Optimal Brain Quantization (OBQ).

To appreciate GPTQ's contribution, we need to understand why directly applying the older approach to LLMs is computationally expensive.

In OBQ, the algorithm can select weights based on their estimated quantization costs. It quantizes a weight, compensates the remaining free weights, and updates the inverse-Hessian representation of the remaining problem.

A representative loss-increase estimate for weight `j` is:

`estimated_loss_j = (w_j - q_j)² / (2 × H_inv[j,j])`

The algorithm can use these estimates to prioritize less costly quantization decisions.

However, suppose a layer has 4,096 input features and 4,096 output features.

It contains approximately 16.8 million weights.

Maintaining and updating second-order information separately for every output row, while repeatedly searching for quantization decisions, becomes extremely expensive.

GPTQ makes several important changes.

First, it observes that a fixed quantization order performs competitively with more elaborate greedy ordering in many large layers. This means all rows can process the same input-feature column at the same time.

Second, because all rows share the same calibration input matrix, they can reuse the same input-space Hessian and its factorization.

Third, GPTQ uses a Cholesky-based reformulation to avoid repeatedly updating and reinverting the full remaining Hessian.

Finally, it processes columns in blocks, delaying the expensive updates to distant columns so they can be performed using efficient matrix operations.

The original GPTQ implementation exposes these design choices through its `fasterquant` method, including block size, group size, damping, activation ordering, and static groups.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)



## 7. Why the algorithm operates column by column

Consider:

```
W.shape = [4, 6]
```

The matrix contains four output channels and six input features.

```
             Input-feature columns

              c0  c1  c2  c3  c4  c5

Output 0      w   w   w   w   w   w
Output 1      w   w   w   w   w   w
Output 2      w   w   w   w   w   w
Output 3      w   w   w   w   w   w
```

GPTQ typically follows a shared column order across the output rows.

At the first step, it quantizes column `c0` for all four output rows.

Next, it computes the quantization errors associated with this column and modifies the remaining unquantized columns.

It then processes `c1`, `c2`, and so forth.

Interactive GPTQ column progression
<img width="480" height="290" alt="image" src="https://github.com/user-attachments/assets/af0b7d31-7555-4828-acd8-270247f7a4b3" />


This ordering matters because compensation can only modify weights that have not yet been quantized.

Earlier quantization decisions have more degrees of freedom available for compensation.

Later decisions have fewer.

The final column has no remaining weights in that row to modify.

This explains one of the trade-offs in the original GPTQ design: a simple shared column order makes the algorithm computationally practical, but the order can still affect reconstruction quality.

## 8. Why Cholesky decomposition is essential

We now reach one of the most technically important parts of GPTQ.

The mathematical compensation rule uses the inverse Hessian of the remaining free coordinates.

In a straightforward implementation, we could repeatedly calculate or update inverse-Hessian matrices after each quantization decision.

That would be expensive and could become numerically unstable.

GPTQ instead exploits matrix factorization.

### 8.1 A quick introduction to Cholesky decomposition

Suppose `A` is a symmetric positive-definite matrix.

Cholesky decomposition expresses it as:

`A = L × L.T`

Here, `L` is a lower-triangular matrix.

For example:

```
L = [
    [a, 0, 0],
    [b, c, 0],
    [d, e, f]
]
```

The transpose is upper triangular.

This structure is useful because triangular matrices are relatively easy to work with in numerical linear algebra.

Cholesky factorization is also generally preferable to explicitly forming an inverse when solving ordinary positive-definite linear systems.

### 8.2 The inverse-Hessian Cholesky factor

GPTQ requires a particular transformation.

After constructing and regularizing the Hessian, the algorithm obtains its inverse and calculates an upper-triangular factor `U` satisfying:

`H_inv = U.T × U`

The original implementation performs the equivalent sequence using Cholesky factorization, Cholesky inversion, and then an upper-triangular Cholesky factor of the inverse.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Why introduce `U`?

Because the sequence of inverse-Hessian matrices for the remaining free coordinates can be represented through the appropriate trailing submatrices of this factor.

That makes sequential compensation much easier to implement.

Instead of repeatedly recomputing the inverse Hessian after every column, GPTQ performs updates using rows of `U`.

### 8.3 The practical GPTQ update rule

Suppose we are currently quantizing column `j`.

First, quantize the weight values in that column:

`q_j = Quantize(w_j)`

Then calculate the normalized column error:

`e_j = (w_j - q_j) / U[j,j]`

Finally, update the current and remaining columns:

`W[:, j:] = W[:, j:] - e_j[:,None] × U[j,j:]`

Here, `e_j[:,None]` converts the column-error vector into a two-dimensional column so that the outer-product update broadcasts correctly.

An implementation generally records the quantized value separately in the output matrix `Q`.

The update above also changes the working copy of the current column. The saved quantized output, however, remains fixed.

### 8.4 Why does the denominator use `U[j,j]` instead of `H_inv[j,j]`?

This is a subtle but important distinction.

The original Optimal Brain Surgeon-style update is written using the inverse Hessian of the current free-variable problem.

After GPTQ's Cholesky reformulation, the algorithm uses an upper-triangular factor that implicitly incorporates the sequence of eliminated coordinates.

The corresponding sequential update uses:

`e_j = (w_j - q_j) / U[j,j]`

The estimated incremental quadratic cost in this transformed formulation is:

`loss_j ≈ (w_j - q_j)² / (2 × U[j,j]²)`

It would be incorrect to replace `U[j,j]` with the diagonal of the original full inverse Hessian while keeping the rest of this Cholesky-based update unchanged.

These two expressions belong to related but differently represented stages of the algorithm.

Understanding this distinction is important when implementing GPTQ from scratch, because confusing the full inverse Hessian with its Cholesky factor can produce incorrect compensation updates.

## 9. Damping: Preventing numerical instability

One challenge with Hessian-based methods is that the Hessian may be singular or nearly singular.

Recall:

`H = 2 × X × X.T`

Suppose two input features are nearly identical.

Then two rows of `X` contain almost the same information.

The Hessian becomes poorly conditioned.

If a matrix is singular, its ordinary inverse does not exist.

Even if it is technically invertible, tiny numerical errors can become amplified during inversion.

GPTQ addresses this using diagonal damping.

The regularized Hessian is:

`H_regularized = H + λI`

Here, `I` is the identity matrix and `λ` is a small positive damping coefficient.

A common configuration in the original implementation chooses:

`λ = percdamp × mean(diagonal(H))`

With `percdamp = 0.01`, the damping magnitude is approximately 1% of the mean Hessian diagonal.

This is not the same as adding a fixed value of 0.01 to every diagonal entry. The amount depends on the scale of the observed activation statistics.

Damping improves numerical stability but also changes the quadratic objective used for compensation.

Too little damping can lead to unstable factorizations or extreme corrections.

Too much damping makes the Hessian more diagonal-dominated and reduces the influence of feature correlations.

In practice, damping is a calibration hyperparameter and should be validated rather than treated as universally optimal.

The implementation must also handle input dimensions with no observed activity. In the original code, inactive dimensions are identified through zero Hessian diagonals, and corresponding weight values are zeroed before factorization.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)





## 10. Block-wise GPTQ: Turning a sequential algorithm into a GPU-efficient algorithm

Up to this point, the algorithm quantizes one column and immediately updates all remaining columns.

Mathematically, this works.

Computationally, it is inefficient for large matrices.

Imagine:

```
W.shape = [4096, 4096]
```

For every quantized column, we would need to modify a large remaining section of the matrix.

Early in the algorithm, that section can contain millions of values.

Repeating these updates individually produces substantial GPU memory traffic and many small operations.

GPTQ addresses this using a technique commonly described as lazy batch updates.

### 10.1 The two levels of processing

The algorithm uses two nested levels.

The outer loop processes blocks of columns, often with block size 128.

The inner loop quantizes individual columns inside the current block.

During the inner loop, compensation is applied immediately to the remaining columns within that block.

However, updates to columns beyond the block are delayed.

After all columns in the block have been quantized, the accumulated errors are applied to the rest of the matrix using a matrix multiplication.

This preserves the intended sequential dependencies while improving GPU efficiency.

Blocked GPTQ execution

<img width="482" height="243" alt="image" src="https://github.com/user-attachments/assets/906487fe-1059-4406-9c84-1af5bb8066c4" />


### 10.2 The delayed update equation

Suppose the current block contains columns from index `start` to `end - 1`.

Let `E_block` contain the normalized quantization errors generated while processing those columns.

The columns after the block are updated using:

`W[:, end:] -= E_block × U[start:end, end:]`

This is a matrix multiplication rather than a sequence of many separate vector updates.

That matters because GPUs are exceptionally efficient at large, regular matrix operations.

A block-wise update improves arithmetic intensity by performing more useful computation per memory access.

For large weight matrices, this change can substantially improve quantization speed.

It is important to distinguish the quantization block size from the quantization group size.

The block size controls how many sequential column operations are accumulated before updating distant columns.

The group size controls how many weight values share scale and zero-point parameters.

A common configuration uses 128 for both, but they solve entirely different problems.

## 11. Complete GPTQ algorithm, step by step

We can now assemble the whole algorithm.

1 Calibration

Collect representative layer-input activations X.

2 Hessian construction

Compute an input-space curvature approximation, proportional to X × X.T.

3 Numerical stabilization

Handle inactive features and apply diagonal damping.

4 Matrix factorization

Compute the upper-triangular factor U of the inverse regularized Hessian.

5 Column ordering

Select the shared order, optionally using activation-order reordering.

6 Column quantization

Map the current adjusted weight column to the low-bit grid.

7 Error compensation

Use U to redistribute the rounding error over remaining free columns.

8 Lazy block update

Accumulate corrections and update distant columns with matrix multiplication.

9 Quantized representation

Store integer codes, scales, zero-points, and required layout metadata.

10 Validation and deployment

Evaluate quality, pack weights, and serve through a compatible low-bit kernel.

The following pseudocode captures the core update structure:

```
# Conceptual GPTQ algorithm: not a complete runnable program.H = build_hessian(calibration_activations)H = regularize(H)U = upper_cholesky_factor_of_inverse(H)Q = empty_quantized_matrix()W_working = original_weights.copy()for block_start in range(0, num_columns, block_size):    block_end = min(block_start + block_size, num_columns)    errors = empty_error_matrix_for_block()    for j in range(block_start, block_end):        q = quantize(W_working[:, j])        error = (W_working[:, j] - q) / U[j, j]        Q[:, j] = q        errors[:, j - block_start] = error        # Compensate the current block immediately.        W_working[:, j:block_end] -= (            error[:, None] * U[j, j:block_end]        )    # Apply accumulated compensation to later blocks.    W_working[:, block_end:] -= (        errors @ U[block_start:block_end, block_end:]    )
```

This pseudocode assumes that the quantizer's scale parameters have been prepared and that `quantize` returns dequantized grid values for use in reconstruction. A real implementation must also store the corresponding integer codes, handle quantization-group boundaries, support optional permutations, and manage tensors in the appropriate device and dtype.

## 12. A working PyTorch implementation from scratch

Let's implement a compact, testable GPTQ reference that demonstrates the Hessian-based column compensation and compares it against ordinary round-to-nearest quantization.

To isolate the GPTQ algorithm, we will use a fixed symmetric INT4 grid. Both methods will use the same quantization scales, so any difference in reconstruction error comes from column-wise compensation rather than from different scale calibration.

The implementation uses FP64 for the Hessian calculations because numerical clarity is more important than speed in this educational example. For simplicity, it processes all columns sequentially instead of applying the blocked optimization, and it uses one scale per output row.

```
"""Educational GPTQ with second-order column compensation."""from dataclasses import dataclassimport torch@dataclassclass GPTQResult:    """Store the logical quantized codes and their reconstruction."""    codes: torch.Tensor    dequantized: torch.Tensor    scales: torch.Tensor    reconstruction_mse: floatdef validate_inputs(    weights: torch.Tensor,    inputs: torch.Tensor,) -> None:    """Validate the matrices used for GPTQ calibration."""    if not isinstance(weights, torch.Tensor):        raise TypeError("weights must be a PyTorch tensor.")    if not isinstance(inputs, torch.Tensor):        raise TypeError("inputs must be a PyTorch tensor.")    if weights.ndim != 2 or inputs.ndim != 2:        raise ValueError("Both matrices must be two-dimensional.")    if weights.shape[1] != inputs.shape[0]:        raise ValueError(            "Expected weights [out, in] and inputs [in, samples]."        )
```

### What this implementation demonstrates

First, the model constructs a calibration set with correlated input features. This is intentional: feature correlations give GPTQ opportunities to compensate for quantization errors.

Next, both round-to-nearest and GPTQ use the same symmetric INT4 scales.

GPTQ constructs the Hessian, applies damping, calculates the Cholesky factor of its inverse, and processes the weight columns sequentially.

Finally, both methods are evaluated using the same calibration inputs.

The reported metric is output reconstruction MSE, not merely weight MSE.

This is the appropriate metric for understanding the compensation mechanism.

The exact improvement is not guaranteed to be positive for every tensor. GPTQ makes sequential, approximate decisions over a discrete set of quantization levels. It does not solve the global discrete optimization problem exactly.

For production use, the implementation would additionally need a blocked update mechanism, optimized GPU tensor operations, scalable calibration, configurable group-wise scales, quantized weight packing, an inference kernel, and robust quality evaluation.

## 13. GPTQ's most important practical hyperparameters

The original GPTQ implementation and follow-up integrations expose several settings that materially influence quality and quantization performance.

| Parameter         | Typical example | Purpose                                                       |
| ----------------- | --------------- | ------------------------------------------------------------- |
| `bits`            | 4               | Number of bits per stored weight                              |
| `groupsize`       | 128             | Weights sharing quantization parameters along a row           |
| `blocksize`       | 128             | Columns processed before a delayed outer update               |
| `percdamp`        | 0.01            | Stabilizes the Hessian factorization                          |
| `actorder`        | True            | Reorders columns using activation-based importance            |
| `static_groups`   | True            | Keeps quantization groups fixed under column reordering       |
| `true_sequential` | True            | Accounts for preceding quantized sublayers during calibration |

These settings are not all independent.

For example, activation ordering changes the sequence of input-feature columns being quantized. If group-wise quantization is also active, the algorithm must ensure that group-scale assignments remain consistent with the actual weight layout.

The original repository introduced static groups partly to avoid inference-layout complications when using activation ordering.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 13.1 Activation ordering: Why important columns may be processed first

GPTQ initially showed that a shared, relatively simple quantization order could work well.

However, later improvements demonstrated that processing columns in decreasing order of activation magnitude can improve accuracy.

A common importance proxy is the diagonal of the Hessian:

`importance[j] = H[j,j]`

Since the Hessian is proportional to `X × X.T`:

`H[j,j] ∝ sum(X[j,:]²)`

Therefore, a larger diagonal indicates that the corresponding input feature had greater squared activation magnitude over the calibration set.

The algorithm can sort columns using:

`order = argsort(diag(H), descending=True)`

Both weights and the Hessian must then be permuted consistently.

The intuition is that highly active input features may have a greater effect on the output, so processing them earlier gives the algorithm more opportunities to compensate for their quantization errors.

However, the diagonal alone is not a complete measure of quantization sensitivity because it ignores cross-feature interactions and the actual rounding error.

Activation ordering is a heuristic, not an exact global optimization strategy.

### 13.2 True-sequential quantization

A transformer block contains multiple projections and nonlinear operations.

Consider a simplified transformer pipeline:

```
Hidden states
     |
     v
Q, K, V projections
     |
     v
Attention operation
     |
     v
Output projection
     |
     v
Feed-forward projections
     |
     v
Next block
```

If every layer is calibrated using activations from the original FP16 model, there is a mismatch.

During actual inference, later layers receive activations produced by earlier quantized layers.

Those activations may differ from the FP16 calibration activations.

True-sequential quantization addresses this by calibrating and quantizing sublayers in dependency order, allowing subsequent layers to observe the effects of earlier quantization.

This can reduce the gap between independent layer reconstruction and the actual quantized model execution.

The original GPTQ repository documents true-sequential quantization and activation ordering as important accuracy improvements for particular model families.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 14. Why calibration quality determines GPTQ quality

GPTQ is a post-training quantization algorithm. It does not normally retrain the entire model through backpropagation.

But that does not mean it operates without data.

It uses representative calibration activations to construct its approximate Hessian.

Suppose a model is primarily used for code generation, but we calibrate it exclusively using short general-conversation prompts.

The observed feature relationships may not adequately represent the model's actual deployment workload.

Then the Hessian may guide compensation toward the wrong distribution of inputs.

This is a form of calibration distribution shift.

For production quantization, calibration should represent the expected workload's token distributions, sequence lengths, and relevant input formats.

For a retrieval-augmented application, this might include long system instructions, retrieved passages, structured data, and realistic user questions.

For a coding assistant, it might include code, natural-language instructions, tool output, and longer contexts.

The objective is not merely to gather a large amount of data. It is to capture representative activation statistics.

A larger calibration dataset may improve the reliability of those statistics, but it also increases calibration cost. Moreover, more data from an unrepresentative distribution does not necessarily produce a better quantizer.

## 15. What happens after GPTQ quantization?

GPTQ is primarily an offline procedure.

The Hessian and its factorization are used to determine the quantized weights. They are not ordinarily required during token generation.

This distinction is extremely important.

The deployment architecture looks like this:

**Offline GPTQ preparation**

Original BF16/FP16 model

Calibration input collection

Capture representative activations

Hessian approximation + GPTQ

Column compensation, grouping, block updates

Packed low-bit checkpoint

Integer weights + scales + zero-points + metadata

**Online inference**

Inference engine

Load quantized weights with compatible kernels

Fused dequantization and matrix multiplication

GPU executes the quantized model

Generated tokens

### 15.1 Does GPTQ perform matrix multiplication directly in INT4?

Not necessarily.

The answer depends on the runtime kernel and quantization format.

In many weight-only INT4 implementations, weights are stored as packed four-bit integers, while activations remain in FP16 or BF16.

During inference, a fused GPU kernel unpacks and dequantizes weight fragments, then performs matrix multiplication using a compute path supported by the hardware.

The important optimization is that the entire full-precision weight matrix does not have to be materialized in GPU memory.

Instead, the kernel can load compressed weight fragments and scales, reconstruct the necessary values inside the computation pipeline, and consume them immediately.

Other low-precision formats and hardware generations may support different native integer or floating-point execution paths.

Therefore, using a GPTQ checkpoint does not automatically mean the GPU performs every multiplication as native INT4 arithmetic.

### 15.2 Memory reduction

Consider an illustrative eight-billion-parameter model.

At FP16, raw weights occupy:

`8 billion × 2 bytes = 16 GB`

At four bits per weight:

`8 billion × 0.5 bytes = 4 GB`

With symmetric INT4 group-wise quantization and one FP16 scale per group of 128 weights:

`effective_bits ≈ 4.125`

Approximate total weight storage becomes:

`8 billion × 4.125 / 8 = 4.125 GB`

Actual checkpoints can be larger because of unquantized layers, extra metadata, alignment, zero-points, and format-specific storage.

The Hessian is not included in the deployed checkpoint because it is only needed during offline quantization.

### 15.3 Does a GPTQ model always run faster?

No.

A GPTQ model is usually smaller than its FP16 counterpart, but smaller storage does not automatically guarantee faster execution.

The runtime must unpack quantized weights, apply scaling, and execute the required matrix operations.

Performance depends on GPU architecture, batch size, tensor dimensions, kernel quality, arithmetic intensity, and whether the workload is limited by memory bandwidth or computation.

At small decode batch sizes, reducing weight-memory traffic can deliver substantial benefits.

At larger batch sizes, matrix multiplication may become more compute-intensive, and the overhead of dequantization or an inefficient kernel can offset some of the memory savings.

A production benchmark should measure time to first token, time per output token, throughput at multiple batch sizes, memory consumption, and quality degradation against the BF16 baseline.

## 16. Where GPTQ works well, and where it can struggle

GPTQ is particularly useful when an existing trained model needs to be compressed without expensive full retraining.

It is attractive for weight-only INT4 inference, memory-constrained serving, and deployments where the original model would otherwise require more GPU memory.

However, it has limitations.

First, the algorithm approximates model behavior through layer reconstruction and local second-order statistics. It does not directly optimize the full sequence-level language-model objective.

Second, its compensation process is sequential and greedy. Once a weight has been quantized, that decision is generally fixed.

Third, reconstruction quality depends on calibration data. Poorly representative activations can lead to poor quantization decisions.

Fourth, low-bit schemes introduce format and hardware constraints. A high-quality quantized checkpoint can still perform poorly if the inference runtime lacks an efficient kernel for that layout.

Finally, GPTQ is primarily a weight quantization technique. It does not automatically solve activation quantization, KV-cache quantization, or the memory requirements of long-context attention.

For those components, additional quantization schemes and serving optimizations are required.

## 17. GPTQ versus ordinary RTN and AWQ

| Dimension                  | RTN                                    | GPTQ                              | AWQ                                        |
| -------------------------- | -------------------------------------- | --------------------------------- | ------------------------------------------ |
| Primary mechanism          | Direct rounding                        | Hessian-guided compensation       | Activation-aware weight protection         |
| Calibration inputs         | Optional, depending on scale selection | Required                          | Required                                   |
| Second-order approximation | No                                     | Yes                               | Not central                                |
| Column-wise compensation   | No                                     | Yes                               | No GPTQ-style compensation                 |
| Group-wise INT4            | Possible                               | Yes                               | Yes                                        |
| Full retraining required   | No                                     | No                                | No                                         |
| Principal challenge        | Quantization error                     | Calibration and quantization cost | Finding effective activation-aware scaling |

GPTQ and AWQ are both important post-training weight quantization approaches, but their internal mechanisms are distinct.

GPTQ modifies the remaining weights during quantization to compensate for previous rounding decisions.

AWQ uses activation-informed scaling to protect weights that are important to the model's behavior.

Neither approach is universally superior across all models, bit widths, and serving environments.

The appropriate choice should be based on model-quality benchmarks and actual inference-kernel performance.

## 18. Production considerations: Scaling, security, and observability

For a large model, quantization should usually be orchestrated layer by layer or transformer block by transformer block rather than placing every layer's Hessian and weight matrix on the GPU simultaneously.

A Hessian for 4,096 input features contains approximately 16.8 million entries. Stored in FP32, it occupies 64 MiB. For a layer with 16,384 input features, the Hessian alone grows to approximately 1 GiB in FP32.

This quadratic scaling makes calibration-memory planning essential, particularly for wide feed-forward projections.

A scalable quantization pipeline should control the number of in-memory Hessians, free intermediate activation buffers, use numerically appropriate factorization dtypes, and persist checkpoints incrementally with validation.

Checkpoint artifacts should include sufficient metadata to reconstruct the quantization configuration exactly. This includes bit width, quantization axis, group size, scale dtype, zero-point convention, any column permutation, and the expected inference layout.

For security, calibration datasets should follow appropriate data-access controls, sensitive prompts should not be indiscriminately logged, and model checkpoints should be stored in integrity-checked, access-controlled repositories.

For observability, track quantization time per layer, Hessian factorization failures, estimated reconstruction loss, output reconstruction MSE, peak GPU memory, scale distributions, saturation counts, and any changes in application-level quality. Once deployed, track latency and throughput separately from offline quantization metrics.

This separation makes it easier to diagnose whether a regression comes from the quantization algorithm, the checkpoint format, or the serving runtime.

## 19. The mathematical essence of GPTQ

We can summarize the entire algorithm using four key expressions.

The layer reconstruction objective is:

```
L = ||(W - Wq) X||²
```

The approximate input-space Hessian is:

```
H = 2 X X.T
```

The inverse-Hessian factorization is:

```
H_regularized = H + λI

H_regularized^-1 = U.T U
```

The sequential compensation update is:

```
q_j = Quantize(W[:, j])

e_j = (W[:, j] - q_j) / U[j, j]

W[:, j:] -= e_j[:, None] * U[j, j:]
```

These equations explain the essential mechanism. GPU-efficient blocked processing, group-wise scale selection, activation ordering, and sequential calibration make the method practical for large transformer models.

The most important distinction to retain is this:

RTN asks which quantized value is closest to each weight. GPTQ asks how to quantize a weight and then redistribute its error so that the remaining layer computation changes as little as possible.

That is why the Hessian, inverse-Hessian factorization, and column-by-column compensation are central to GPTQ.

### Primary references

The most useful sources for studying the actual algorithm are the [original GPTQ paper](https://arxiv.org/abs/2210.17323) and the [official GPTQ implementation](https://github.com/IST-DASLab/gptq). The `gptq.py` implementation is particularly valuable because it shows the same Hessian accumulation, Cholesky factorization, sequential error propagation, and delayed block updates developed here.

The next natural concept in this inference-engineering sequence is AWQ — Activation-Aware Weight Quantization, especially how it protects salient weights through activation-informed scaling without GPTQ's Hessian-based sequential compensation.
