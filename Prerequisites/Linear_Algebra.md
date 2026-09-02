# Linear Algebra Refresher (with FLOP-counting in mind)

## Dot Product

For vectors $x, y \in \mathbb{R}^n$:

$$x \cdot y = \sum_{i=1}^n x_i y_i$$

**Cost:** $n$ multiplications + $(n-1)$ additions ≈ $2n$ FLOPs. This "multiply-then-add" pair is the atomic unit of almost all linear algebra cost accounting — you'll see the constant "2" show up everywhere for exactly this reason.

## Matrix-Vector Product

$A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$, result $y = Ax \in \mathbb{R}^m$.

Each of the $m$ output entries is a dot product of a row of $A$ (length $n$) with $x$:

$$y_i = \sum_{j=1}^n A_{ij} x_j$$

**Cost:** $m$ dot products, each $2n$ FLOPs → $2mn$ FLOPs total.

## Matrix-Matrix Product

$A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$, result $C = AB \in \mathbb{R}^{m \times n}$.

Each entry $C_{ij}$ is a dot product of length $k$:

$$C_{ij} = \sum_{l=1}^k A_{il} B_{lj}$$

There are $mn$ output entries, each costing $2k$ FLOPs:

**Cost:** $2mnk$ FLOPs

This is *the* formula you'll use constantly. Sanity check: matrix-vector is the special case $n=1$, giving $2mk$ — matches above.

### Why this matters for transformer FLOP counting
A linear layer mapping $d_{in} \to d_{out}$ applied to a batch of $B$ tokens is a $(B \times d_{in}) \times (d_{in} \times d_{out})$ matmul:

$$\text{FLOPs} = 2 \cdot B \cdot d_{in} \cdot d_{out}$$

This is where the famous "$2 \times \text{params} \times \text{tokens}$" approximation for a forward pass comes from — each parameter is used in one multiply-add per token, and multiply-add = 2 FLOPs.

## Vector Norms

**$L_2$ (Euclidean) norm:**
$$\|x\|_2 = \sqrt{x \cdot x} = \sqrt{\sum_i x_i^2}$$
Cost: essentially a dot product ($2n$ FLOPs) plus one sqrt.

**$L_1$ norm:** $\|x\|_1 = \sum_i |x_i|$ — cost $\approx n$ (just adds, no multiplies).

**$L_\infty$ norm:** $\|x\|_\infty = \max_i |x_i|$ — cost $\approx n$ comparisons.

**General $L_p$:** $\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$.

Norms show up constantly in ML not just geometrically but as regularizers, in gradient clipping (clip by global $L_2$ norm), and in normalization layers (RMSNorm uses $\sqrt{\frac{1}{n}\sum x_i^2}$, essentially a scaled $L_2$ norm).

## Quick reference table

| Operation | Shapes | FLOPs |
|---|---|---|
| Dot product | $x,y \in \mathbb{R}^n$ | $2n$ |
| Matrix-vector | $A \in \mathbb{R}^{m\times n}$ | $2mn$ |
| Matrix-matrix | $A \in \mathbb{R}^{m\times k}, B \in \mathbb{R}^{k\times n}$ | $2mnk$ |
| Batched linear layer | $B$ tokens, $d_{in}\to d_{out}$ | $2Bd_{in}d_{out}$ |

## A few gotchas worth internalizing

- **FLOPs vs FLOP/s**: the former is total work, the latter is a rate (hardware throughput). Don't conflate them when reading papers.
- **The "2" convention isn't universal.** Some sources count a multiply-add as 1 FLOP ("FMA" convention) rather than 2. Always check which convention a paper uses before comparing numbers across sources.
- **Batch dimensions multiply through.** If you're doing $B$ independent matmuls (e.g., attention heads, or batch of sequences), FLOPs scale linearly in $B$ — just another factor in the product.
- **Attention has a different scaling.** QK^T and the subsequent weighted sum with V scale as $O(n^2 d)$ in sequence length $n$, not $O(nd^2)$ like the linear layers — this quadratic-in-sequence-length term is why long-context is expensive and why FLOP derivations for transformers split into "linear layer" terms and "attention" terms separately.
# Full FLOP-Counting Mastery: Transformers Edition

## 1. The core building block, restated precisely

Every dense layer is a matmul. For batch/sequence of $T$ tokens through a weight matrix $W \in \mathbb{R}^{d_{in}\times d_{out}}$:

$$\text{FLOPs}_{\text{fwd}} = 2 \cdot T \cdot d_{in} \cdot d_{out}$$

Notice $d_{in}\cdot d_{out}$ is just the **parameter count** of that weight matrix, $N_{\text{layer}}$. So:

$$\text{FLOPs}_{\text{fwd}} = 2 \cdot T \cdot N_{\text{layer}}$$

This is the seed of the whole "$2N$ per token forward" heuristic — it holds matmul-by-matmul, then you sum over all matmuls in the network.

## 2. Forward pass for a full transformer: $2N$ per token

Summing $2 \cdot T \cdot N_{\text{layer}}$ over every weight matrix in the model gives:

$$\text{FLOPs}_{\text{fwd}} \approx 2 \cdot T \cdot N$$

where $N$ = total non-embedding parameter count. This works because **almost every parameter in a transformer sits inside some matmul that touches every token exactly once** — QKV projections, attention output projection, and the two MLP matrices. Embeddings are usually excluded (they're a lookup, not a matmul — ~0 FLOPs) and the unembedding is sometimes included, sometimes not, depending on the paper.

**What's NOT captured by $2N$:** the attention score computation ($QK^T$) and the attention-weighted sum ($\text{softmax}(QK^T)V$). These don't scale with parameter count — they scale with sequence length squared:

$$\text{FLOPs}_{\text{attn}} \approx 2 \cdot 2 \cdot T^2 \cdot d = 4T^2 d$$

(one $T^2 d$ term for $QK^T$, one for the weighted sum with $V$; the factor 2 inside each is the usual matmul constant). This term is small relative to $2Td$-per-layer-ish terms when $T \ll d$, which is why the $2N$ approximation is popular for moderate context lengths — but it dominates and breaks the approximation once $T$ gets large (long-context regime).

The fuller formula (from the Kaplan/Chinchilla-era papers) is often written:

$$\text{FLOPs}_{\text{fwd}} \approx 2NT + 2 \cdot L \cdot T^2 \cdot d$$

where $L$ = number of layers, $d$ = model dimension — the second term is total attention cost across all layers.

## 3. Backward pass: why it's ~2× forward

This is the part people memorize without deriving, so let's actually derive it.

For a matmul $Y = XW$ (forward), backprop needs **two** gradients:

- $\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} W^T$ — needed to keep propagating the gradient backward to earlier layers
- $\frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Y}$ — needed to actually update this layer's weights

Each of these is **itself a matmul of the same size class** as the forward matmul, i.e., each costs $\approx 2TN_{\text{layer}}$ FLOPs. So:

- Forward: $1 \times (2TN_{\text{layer}})$
- Backward: $2 \times (2TN_{\text{layer}})$ (one for $\partial X$, one for $\partial W$)

Total backward = 2× forward, and forward+backward = 3× forward = **$6TN_{\text{layer}}$**. Summed over the network:

$$\text{FLOPs}_{\text{fwd+bwd}} \approx 6NT$$

This is the origin of the famous **"$6N$ per token"** rule for total training compute per token, and $C \approx 6ND$ for a full training run over $D$ tokens (the Kaplan/Chinchilla scaling-law formula).

**Important subtlety:** the very first layer doesn't need $\partial X$ (nothing behind it to propagate to), so technically it's slightly under $6N$, but this is a rounding error for deep networks and universally ignored.

## 4. Putting it together: the compute formula hierarchy

| Approximation | Formula | Captures |
|---|---|---|
| Single matmul | $2mnk$ | exact |
| One layer, forward | $2TN_{\text{layer}}$ | exact (matmul-based layers only) |
| Full model, forward | $2NT$ | ignores attention quadratic term |
| Full model, fwd+bwd | $6NT$ | ignores attention quadratic term |
| Full training run | $C \approx 6ND$ | $D$ = total training tokens |
| More precise (with attention) | $2NT + 2LT^2d$ (fwd); $\times 3$ for fwd+bwd | includes attention cost |

## 5. Inference-specific wrinkles (very commonly tested/asked about)

- **Prefill vs decode differ enormously.** Prefill processes the whole prompt in one matmul pass — FLOPs scale as above, $\approx 2NT_{\text{prompt}}$, and is **compute-bound** (good GPU utilization).
- **Decode (autoregressive generation)** processes **one token at a time**, so each step is $\approx 2N \cdot 1$ FLOPs — tiny. Decode is dominated by **memory bandwidth** (reading all $N$ parameters from HBM per token), not FLOPs. This is the reason people say decode is "memory-bound" not "compute-bound" — the FLOP/byte ratio is terrible when $T=1$.
- **KV cache** avoids recomputing $K,V$ for past tokens during decode, which is what makes single-token decode cheap in FLOPs at all — without it you'd reprocess the whole growing sequence every step ($O(T^2)$ instead of $O(T)$ over generation).
- **Arithmetic intensity** = FLOPs / bytes moved. Compute-bound when this exceeds the hardware's FLOP/byte ratio (roofline model); memory-bound below it. Prefill: high intensity (compute-bound). Decode: low intensity (memory-bound), which is why batching many requests together during decode (to amortize the weight reads) is the standard trick to get GPU utilization up.

## 6. Common gotchas / things people get subtly wrong

1. **MLP block dominates parameter count.** With hidden dim $d$ and MLP expansion factor 4 (standard), the MLP has $2 \times 4d^2 = 8d^2$ params per layer (up-proj + down-proj) vs. attention's roughly $4d^2$ (Q,K,V,O projections at $d\times d$ each — less if using GQA/MQA to shrink K,V). So MLP is usually ~2/3 of total parameters, meaning ~2/3 of FLOPs too.
2. **GQA/MQA reduce KV projection FLOPs and params**, but Q and O projections are unaffected — don't assume attention FLOPs collapse uniformly.
3. **Bias terms are $O(n)$, negligible.** A bias add is $n$ FLOPs vs. $2mn$ for the matmul it's attached to — safely ignored in back-of-envelope counts.
4. **Softmax, layernorm, activation functions (GELU/SiLU)** are all $O(T \cdot d)$ — linear, not quadratic, and dwarfed by the matmul terms. They matter for wall-clock latency (they're often memory-bound, causing kernel-launch overhead) but not for FLOP accounting.
5. **The "2" convention strikes again.** Always check whether a paper's $C=6ND$ is using FMA=1 or FMA=2 FLOP convention — this alone can make numbers look inconsistent across sources by a factor of 2.
6. **Embedding/unembedding matrices** are sometimes included in $N$, sometimes excluded ("non-embedding parameters"). For large-vocab, small-model regimes this is *not* negligible — always check which $N$ a paper means.

## 7. Worked example

Say a model has $N = 7\times10^9$ non-embedding params, trained on $D = 2\times10^{12}$ tokens.

$$C \approx 6ND = 6 \times 7\times10^9 \times 2\times10^{12} = 8.4\times10^{22} \text{ FLOPs}$$

If you have a cluster doing $312$ TFLOP/s per GPU (A100 bf16 dense peak) at, say, 40% utilization ($\approx 1.25\times10^{14}$ FLOP/s effective) on 100 GPUs ($1.25\times10^{16}$ FLOP/s aggregate):

$$\text{time} = \frac{8.4\times10^{22}}{1.25\times10^{16}} \approx 6.7\times10^{6}\text{ s} \approx 78 \text{ days}$$

This is exactly the kind of estimate you'll be reconstructing constantly from FLOP-cost derivations — worth being able to do it cold.

---
## A) Deriving the attention FLOP term step by step

Setup: sequence length $T$, model dim $d$, number of heads $h$, head dim $d_h = d/h$.

**Step 1 — Q, K, V projections.** These are standard linear layers, already counted in the $2NT$ term (not attention-specific). Skip them here — we're isolating the part $2NT$ *doesn't* capture.

**Step 2 — Compute attention scores $QK^T$.**

For a single head: $Q \in \mathbb{R}^{T \times d_h}$, $K \in \mathbb{R}^{T \times d_h}$, and we compute $QK^T \in \mathbb{R}^{T\times T}$.

This is a matmul with $m=T$, $k=d_h$, $n=T$:
$$\text{FLOPs} = 2 \cdot T \cdot d_h \cdot T = 2T^2 d_h$$

Across $h$ heads: $h \cdot 2T^2 d_h = 2T^2 (h d_h) = 2T^2 d$ (since $h d_h = d$).

**Step 3 — Softmax.** $O(T^2)$ per head (exponentiate + normalize an $T\times T$ matrix), negligible next to the $T^2 d$ matmul terms — ignore, as usual.

**Step 4 — Weighted sum with V: $\text{softmax}(QK^T)V$.**

$\text{scores} \in \mathbb{R}^{T\times T}$, $V \in \mathbb{R}^{T \times d_h}$, product $\in \mathbb{R}^{T \times d_h}$:
$$\text{FLOPs} = 2 \cdot T \cdot T \cdot d_h = 2T^2 d_h$$
Across heads: $2T^2 d$ again.

**Step 5 — Sum steps 2 and 4:**
$$\text{FLOPs}_{\text{attn, 1 layer}} = 2T^2d + 2T^2d = 4T^2d$$

Across $L$ layers:
$$\text{FLOPs}_{\text{attn, total, fwd}} = 4LT^2d$$

Matches what I asserted earlier. Note it's **independent of $h$** — splitting into more heads doesn't change total FLOPs, since $h \cdot d_h = d$ is fixed (only changes how work is parallelized/shaped).

**Full forward formula, now fully justified:**
$$\text{FLOPs}_{\text{fwd}} \approx \underbrace{2NT}_{\text{linear layers}} + \underbrace{4LT^2d}_{\text{attention}}$$

**Fwd+bwd** (×3 rule from before, applies to both terms since attention matmuls also need two backward passes):
$$\text{FLOPs}_{\text{fwd+bwd}} \approx 6NT + 12LT^2d$$

**Where the crossover happens:** the attention term matters when $4LT^2d \gtrsim 2NT$, i.e. roughly $T \gtrsim N/(2Ld)$. Since $N \approx 12Ld^2$ for a standard transformer (12 comes from: 4 attention matmuls at $d^2$ each + 8 MLP at $d^2$ each, roughly, per layer — see below), this simplifies to $T \gtrsim 6d$. So for $d=4096$, attention starts mattering around $T \gtrsim 25000$ tokens — consistent with "quadratic attention cost becomes the bottleneck in long-context regimes."

## B) MoE: active vs. total parameters

Standard dense transformer: every parameter touches every token → $N$ in $2NT$ is unambiguous.

**Mixture-of-Experts breaks this.** The MLP block is replaced by $E$ expert MLPs, but a router sends each token to only $k$ of them (commonly $k=1$ or $k=2$, "top-$k$ routing").

This creates **two different parameter counts** you must distinguish:

- **$N_{\text{total}}$**: sum of all parameters across all $E$ experts (plus attention, embeddings, etc.) — this is what determines **memory footprint** (you must store all experts, since any token could route to any of them).
- **$N_{\text{active}}$**: parameters actually used for a *given* token's forward pass — attention params (always active) + only $k$ out of $E$ experts' worth of MLP params. This determines **FLOPs**.

$$\text{FLOPs}_{\text{fwd}} \approx 2 N_{\text{active}} \cdot T \quad(\text{+ attention term, unaffected by MoE})$$

**Concretely:** if each expert MLP has $P_{\text{mlp}}$ params, $E$ experts total, top-$k$ routing:
$$N_{\text{total}} = N_{\text{attn+other}} + E \cdot P_{\text{mlp}}, \qquad N_{\text{active}} = N_{\text{attn+other}} + k \cdot P_{\text{mlp}}$$

**Example (Mixtral-8x7B-style):** $E=8$ experts, $k=2$ active per token, each expert ~7B-ish share of params. $N_{\text{total}} \approx 47$B, $N_{\text{active}} \approx 13$B. You pay **memory cost of a 47B model** but **compute cost of a 13B model** per token. This is the entire value proposition of MoE — decoupling capacity (total params, helps quality) from compute (active params, controls training/inference cost).

**Gotchas specific to MoE FLOP counting:**

1. **The router itself costs FLOPs too** — a small linear layer $d \to E$ per token, $2Td E$ — but this is tiny relative to expert FLOPs and usually ignored.
2. **Training compute scaling laws change.** The Chinchilla-style $C\approx 6ND$ needs $N\to N_{\text{active}}$ for compute, but the *memory/parameter-count-driven* generalization behavior tracks $N_{\text{total}}$ more closely — meaning MoE scaling laws are genuinely a 2D problem (active vs. total), not reducible to the dense 1D $N$ story. Papers like the DeepSeek-MoE and Switch Transformer scaling analyses treat these as separate axes.
3. **Load balancing FLOPs/losses** — MoE training adds an auxiliary load-balancing loss term to encourage even expert utilization; this is a tiny additional compute cost but a real *engineering* complexity (uneven expert load → uneven GPU utilization → wasted FLOPs at the *hardware* level even though the *algorithmic* FLOP count is fine). This is a case where the clean FLOP formula and real-world achieved throughput diverge — important to know as a caveat when reading papers that quote theoretical vs. measured MFU (Model FLOPs Utilization).
4. **Inference memory-boundedness gets worse, not better, for MoE at low batch size** — during decode, you still need all $E$ experts resident in memory (or fast-loadable) even though only $k$ fire per token, so the memory-bound nature of decode (Section 5 above) is *not* alleviated by MoE the way FLOPs are — a common misconception.

---

