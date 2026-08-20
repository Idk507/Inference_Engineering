# The Complete Engine Manual: LLM Inference, Full Mechanics

Every technique below is taken all the way to the equations, the exact algorithm, and worked numbers — nothing left at the intuition level. This is the single consolidated reference: attention math, memory formulas, every major serving optimization, quantization algorithms, speculative decoding's proof of correctness, parallelism math, and mechanics-accurate pseudocode tying it all together.

---

## 1. The Attention Formula Itself — QKV, Scaling, Causal Masking, Multi-Head Split

Everything downstream in this guide is a variation on one computation, so it comes first.

### 1.1 Q, K, V

Each token embedding `x ∈ ℝ^(d_model)` is projected through three learned matrices:

```
q = x · W_Q      W_Q ∈ ℝ^(d_model × d_k)
k = x · W_K      W_K ∈ ℝ^(d_model × d_k)
v = x · W_V      W_V ∈ ℝ^(d_model × d_v)
```

Stacked over a sequence of `n` tokens: `Q, K, V` have shapes `[n, d_k], [n, d_k], [n, d_v]`. Query = "what am I looking for," Key = "what do I offer," Value = "here's my actual content if selected."

### 1.2 Scaled dot-product attention, term by term

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

- **`QKᵀ`** → `[n, n]` matrix; entry `(i,j) = q_i · k_j`, a relevance score between every pair of tokens.
- **`/ √d_k`** → dot-product variance grows with `d_k`; unscaled, softmax inputs saturate into a regime with near-zero gradient and brittle, near-one-hot outputs. Dividing by `√d_k` keeps values numerically well-behaved regardless of head dimension.
- **`softmax(...)`** row-wise: `softmax(z)_i = exp(z_i) / Σ_j exp(z_j)` — turns each row into a proper probability distribution over which tokens to attend to.
- **`· V`** → `[n,n] × [n,d_v]` = `[n, d_v]`: each token's new representation is a weighted blend of every token's Value vector, weighted exactly by the attention distribution.

### 1.3 Causal masking, precisely

An upper-triangular mask `M` is added before softmax: `M_{ij} = 0` for `j ≤ i`, `M_{ij} = -∞` for `j > i`.

```
masked_scores = (QKᵀ/√d_k) + M
```

`exp(-∞) = 0`, so disallowed future positions get exactly zero attention weight with no separate filtering step needed. This is also *why the KV cache is valid at all*: token `i`'s computation never depends on tokens after it, so once Keys/Values for tokens `1..i-1` are computed they never need to change — permanently cacheable.

### 1.4 Multi-head split

```
head_i = Attention(X·W_Q_i, X·W_K_i, X·W_V_i)     i = 1..h
MultiHead(X) = Concat(head_1, ..., head_h) · W_O
```

Each head gets its own `W_Q_i, W_K_i, W_V_i` (dimension `d_model × d_head`, where `d_head = d_model/h`), letting different heads specialize (syntax tracking, coreference, etc.) purely as an emergent training outcome. `W_O` (`d_model × d_model`) linearly recombines all heads back into one vector.

---

## 2. Prefill — FLOP Cost Derivation and FlashAttention

### 2.1 FLOP cost, derived

Per layer, over `n` tokens:

- `Q = X·W_Q` (and K, V): `[n,d_model]×[d_model,d_k]` → `O(n · d_model · d_k)` — linear in `n`.
- `QKᵀ`: `[n,d_k]×[d_k,n]` → `O(n² · d_k)` — quadratic in `n`.
- `softmax(...)·V`: `[n,n]×[n,d_v]` → `O(n² · d_v)` — quadratic in `n`.

For short sequences the linear projection terms dominate; past a crossover point the `O(n²)` terms dominate, so doubling context length roughly quadruples attention's compute cost. This is the mathematical root of "long context is expensive," and directly motivates FlashAttention.

**Why prefill is compute-bound:** the same weight matrices are loaded once from memory and reused across all `n` tokens in the pass — arithmetic intensity (FLOPs per byte moved) scales upward with `n`, keeping the GPU's math units, not its memory bus, as the bottleneck.

### 2.2 FlashAttention's tiling / online-softmax trick

Naively, `softmax(QKᵀ/√d_k)` requires materializing the full `[n,n]` score matrix in slow GPU HBM — for `n=8192`, ~67 million entries *per head, per layer*, written and read back, which is itself a large memory-bound operation sitting inside an otherwise compute-bound phase.

FlashAttention processes `Q, K, V` in small tiles that fit in the GPU's fast on-chip SRAM, maintaining a **running (online) softmax** as it streams through tiles:

```
For each new tile with local max m_new, given previous running max m_old:
    m_combined = max(m_old, m_new)
    rescale the previously accumulated output and running sum by
    exp(m_old − m_combined), and the new tile's contribution by
    exp(m_new − m_combined), then accumulate both into the running total
```

This correction-factor rescaling means the algorithm never needs the full row of scores in memory at once to compute a numerically correct softmax — it accumulates the exact same result incrementally, tile by tile. The output is mathematically identical (modulo floating-point rounding) to naive full materialization; what changes is how much data moves between slow and fast memory, which is precisely what was the bottleneck.

---

## 3. The KV Cache — Exact Per-Token Byte Formula

### 3.1 The formula

```
bytes/token = 2 (K and V) × L (layers) × H_kv (KV heads) × d_head × bytes_per_element
```

### 3.2 Worked 7B-model calculation

7B-parameter model: `L=32` layers, `H_kv=32` (full MHA), `d_head=128`, FP16 (`bytes_per_element=2`):

```
bytes/token = 2 × 32 × 32 × 128 × 2
            = 2 × 32  = 64
              64 × 32 = 2,048
           2,048 × 128 = 262,144
         262,144 × 2   = 524,288 bytes/token
                       ≈ 512 KB/token
```

For a 4,096-token conversation: `512 KB × 4,096 ≈ 2 GB` — for a *single* request. At 64 concurrent users at that context length: `2 GB × 64 = 128 GB`, which can exceed the 14 GB the model's own weights occupy at FP16 by an order of magnitude. This is the numeric proof that KV cache, not model weights, is often the true memory bottleneck at scale.

### 3.3 Why decode is memory-bound, quantified

Generating one token requires reading back the *entire* cached K, V tensors (all `n` previous tokens, every layer) to compute `Q_new · Kᵀ`, plus the full weight matrices — while the compute performed (`O(n·d_k)` per layer) is tiny relative to prefill's `O(n²·d_k)`. Bytes-moved-to-FLOPs-performed is far higher than in prefill — the formal definition of memory-bound.

---

## 4. MHA vs. MQA vs. GQA — Exact Head-Sharing Math

The `H_kv` term in Section 3.1's formula is the number of *Key/Value* heads, which under modern architectures is a separate, smaller parameter than `H_q`, the number of *Query* heads.

- **MHA:** `H_kv = H_q`. Every query head has its own dedicated K/V head. Max expressiveness, max cache size.
- **MQA:** `H_kv = 1`. All `H_q` query heads share one K/V head — cache shrinks by a factor of `H_q`.
- **GQA:** query heads are split into `G` groups; each group of `H_q/G` heads shares one K/V head. Cache shrinks by `H_q/G` relative to MHA.

**Comparison table, 32 query heads, `d_head=128`, 32 layers, FP16 (same model as Section 3.2):**

| Variant | H_kv | Bytes/token | 4K-token cache size |
|---|---|---|---|
| MHA | 32 | 524,288 B | ~2.0 GB |
| GQA (G=8) | 8 | 131,072 B | ~0.5 GB |
| MQA | 1 | 16,384 B | ~64 MB |

This is Section 3.1's formula with only `H_kv` changed. GQA with `G=8` gives roughly a 4x cache reduction with minimal measured quality loss in published ablations — the reason Llama-3, Mistral, and most contemporary open models converge on group counts in that range.

---

## 5. PagedAttention — Block Table, Allocator Algorithm, Copy-on-Write

### 5.1 The fragmentation problem, quantified

Pre-allocating a contiguous buffer sized for max context (e.g., 4,096 tokens = 2 GB under MHA from Section 3.2) when a request only generates 200 tokens wastes `(4096-200)/4096 ≈ 95%` of that reservation for the request's entire lifetime.

### 5.2 The data structure

KV cache is divided into fixed-size **blocks** (commonly 16 tokens' worth of K/V per block). GPU memory is a flat pool of block-sized slots. Each request keeps a **block table**: `logical_block_index → physical_block_index`, structurally identical to an OS page table mapping virtual to physical pages.

### 5.3 The allocator algorithm

```
class Block:
    physical_id, keys, values, ref_count

class BlockAllocator:
    free_list = [0, 1, ..., total_blocks-1]

    allocate():
        pid = free_list.pop()
        blocks[pid].ref_count = 1
        return blocks[pid]

    free(block):
        block.ref_count -= 1
        if block.ref_count == 0:
            free_list.append(block.physical_id)
```

1. New request arrives with an empty block table.
2. As K/V vectors are produced, they fill the current block; once it reaches 16 tokens, `allocate()` grabs a new physical block from the free-list and appends `(logical→physical)` to the block table.
3. Attention kernels gather the needed blocks via the block table directly — no requirement of physical contiguity.
4. On request completion, every block in its table is passed to `free()`, instantly available for the next request — no defragmentation pass needed, since blocks were never contiguous to begin with.

### 5.4 Copy-on-write for beam search

When multiple candidate continuations (beam search, or parallel `n>1` sampling) share a common prefix, their block tables can point to the *same* physical blocks for that shared prefix, incrementing `ref_count` per block. Only when a branch diverges and needs to *write* a new token into a shared block does the system copy that specific block first — the shared prefix itself is never duplicated, only the point of divergence forward.

### 5.5 Fragmentation bound

At most one partially-filled block per request is wasted (≤15 of 16 token-slots in the worst case) — versus potentially thousands of wasted slots under naive contiguous pre-allocation. Fragmentation drops from `O(max_context_length)` to `O(block_size)`.

---

## 6. Prefix Caching — The Chained-Hash Matching Mechanism

### 6.1 The mechanism

1. Each block is hashed based on the **token IDs it contains plus the hash of the preceding block** — a chained hash, so block `k`'s hash implicitly encodes the entire prefix from block 0 through block `k`.
2. A new request's prompt is hashed the same way, block by block, and checked against a global hash table (`hash → physical block`) of still-resident blocks.
3. Every leading block that matches has the new request's block table point directly to the existing physical block (incrementing its ref count) — no computation, no write. Prefill only runs for tokens after the longest matched prefix.
4. The first non-matching block breaks the chain (since a chained hash differs the instant any earlier token differs); normal prefill resumes from there.

### 6.2 Worked example

An 800-token system prompt (50 blocks at 16 tokens/block) shared by every request: the first request populates and registers 50 physical blocks. Every subsequent request's first 50 logical blocks hash-match immediately — their block tables just reference the existing 50 physical blocks, skipping ~800 tokens of prefill and avoiding duplicating ~800 × 512 KB/token (Section 3.2's MHA figure) of KV memory per request.

### 6.3 Eviction

Cached blocks with zero active references are reclaimed via LRU when the pool runs low, prioritizing keeping frequently-reused prefixes (system prompts) resident over one-off long documents.

---

## 7. Continuous Batching — Scheduler State Machine and Admission Control

### 7.1 Request states

- **Waiting** — arrived, not yet admitted.
- **Running** — occupying a batch slot, producing one token per iteration.
- **(Swapped)** — temporarily evicted to CPU memory under pressure, resumable later (systems that support it).

### 7.2 The scheduling loop, per iteration

1. For every `Running` request, check if its most recent token satisfies a stop condition (Section 11). If so → `Finished`, free its KV blocks (Section 5.3's `free()`) back to the pool.
2. While spare capacity exists — both a free slot up to `max_batch_size` **and** enough free KV blocks in the pool — and `Waiting` is non-empty: pop the next request, run its prefill (treated as a special first "iteration" for scheduling purposes), move it to `Running`.
3. Build one batched forward pass containing exactly one token of computation per `Running` request, gathering each request's needed blocks via its block table, and execute the model across the whole batch simultaneously.
4. Sample the next token per request independently (each may use different temperature/top-p even within the same physical batch).
5. Append each new token's K/V to its own cache, stream to the client, loop to step 1.

### 7.3 Why this maximizes throughput

Step 3's forward pass loads model weights from HBM exactly once per iteration regardless of batch size (Section 3.3) — the fixed bandwidth cost is amortized across however many requests are packed in. Admission in step 2 is deliberately greedy, filling every iteration to the maximum the KV pool can currently support, specifically to maximize this amortization.

### 7.4 Admission control nuance

Naive greedy admission can over-commit KV blocks if many newly-admitted requests turn out to need long generations simultaneously. Production schedulers use a watermark/threshold on free blocks, and can preemptively swap or recompute a lower-priority running request's cache if the pool is exhausted rather than admitting without bound.

---

## 8. Chunked Prefill — Iteration-Level Interleaving Pattern

A pure iteration-level scheduler (Section 7) has one flaw: a very long new prompt's prefill is itself a large, uninterruptible unit of work relative to one decode token — prefilling an 8,000-token document can stall every other request's next token for that entire window.

**The fix:** split prefill into fixed-size chunks (e.g., 512 tokens) and schedule one chunk per iteration, interleaved with the batch's regular decode work:

```
Iteration N:   [decode, decode, decode, prefill_chunk(tokens 0–511)]
Iteration N+1: [decode, decode, decode, prefill_chunk(tokens 512–1023)]
Iteration N+2: [decode, decode, decode, prefill_chunk(tokens 1024–1535)]
```

Each iteration's batched pass now mixes a small amount of compute-bound prefill work with the batch's memory-bound decode work in the same kernel launch, additionally improving overall GPU utilization by blending the two regimes rather than running either in isolation. Chunk size trades off: larger chunks finish long prompts sooner but delay others more per iteration; smaller chunks minimize per-iteration impact but take more iterations to finish a long prompt.

---

## 9. Sampling — Exact Softmax / Top-k / Top-p / Beam Search Algorithms

Given raw logits `z ∈ ℝ^V`:

**Softmax with temperature `T`:** `p_i = exp(z_i/T) / Σ_j exp(z_j/T)`. As `T→0`, converges to one-hot on `argmax(z)` (= greedy). As `T→∞`, converges to uniform. `T=1` recovers the raw distribution.

**Greedy:** `token = argmax(p)`. Deterministic; the `T→0` limit.

**Top-k, exact steps:**
1. Sort `p` descending, keep top `k`: `p_(1) ≥ ... ≥ p_(k)`.
2. Renormalize: `p'_i = p_i / Σ_{j=1}^{k} p_j`.
3. Sample from the resulting `k`-way categorical distribution.

**Top-p (nucleus), exact steps:**
1. Sort `p` descending.
2. Compute cumulative sum `C_m = Σ_{i=1}^{m} p_(i)`.
3. Find smallest `m*` such that `C_{m*} ≥ p_threshold` (e.g. 0.9).
4. Keep top `m*` tokens, renormalize, sample.

Key structural difference from top-k: `m*` varies per step with how peaked/flat the distribution is — a confident distribution (one token at 0.95) yields `m*=1` (effectively greedy that step); a flat, uncertain distribution yields a large `m*`. Fixed-`k` top-k can't adapt this way.

**Beam search, exact algorithm (B beams):**
1. Initialize `B` beams from the prompt, cumulative log-probability score 0.
2. Each step: for every beam, compute the next-token distribution and consider its top `B` extensions, scored as beam's prior cumulative log-prob + `log p(next_token)`.
3. Across all `B × B` candidates, keep only the overall top `B` by cumulative score (the pruning step keeping search tractable).
4. Repeat until every surviving beam emits EOS or hits max length.
5. Return the beam with highest final (often length-normalized) score.

Cost: roughly `B×` a single greedy decode's cost, since `B` sequences' forward passes and KV caches are evaluated in parallel every step — a direct, quantifiable trade-off between optimality and compute per request.

---

## 10. Quantization — Affine Formula, Granularity, GPTQ, and AWQ in Full

### 10.1 The core affine quantization formula

```
Quantize:    W_int = clamp(round(W/s) + z, q_min, q_max)
Dequantize:  W_hat = s × (W_int − z)
```

**Symmetric** (typical for weights): `z=0`, `s = max(|W|) / (2^(b-1) - 1)`.
**Asymmetric** (typical for activations): `s = (max(W)-min(W)) / (q_max-q_min)`, `z = round(q_min - min(W)/s)`.

Max rounding error per value: `|W - W_hat| ≤ s/2` — showing directly why a single global `s` dragged upward by one outlier inflates error for every other value in the tensor.

### 10.2 Granularity levels

- **Per-tensor:** one `(s,z)` for the whole matrix. Cheapest metadata, worst error under outliers.
- **Per-channel:** one `(s,z)` per output channel — outlier in one channel doesn't inflate another's scale. Negligible metadata overhead (e.g., 4096 extra values for a 4096×4096 matrix).
- **Per-group (e.g., 128 weights):** one `(s,z)` per contiguous run along a channel — interpolates between per-tensor and per-channel, standard for aggressive 4-bit weight-only quantization.

### 10.3 GPTQ, the algorithm in full

**Goal:** minimize squared error between the layer's original output and its quantized-weight output, evaluated on real calibration activations — not just raw weight-rounding error in isolation.

**Setup:** given calibration activations `X`, define `H = 2·XᵀX` (Hessian approximation from treating this as least-squares: minimizing `||WX - W_hat·X||²`).

**Core loop, column by column, left to right:**
1. Quantize current column `q`: `w_hat_q = quantize(w_q)`.
2. Compute error `e = (w_q - w_hat_q) / [H⁻¹]_{qq}`.
3. Propagate a corrective update to all not-yet-quantized columns:
   ```
   W_{:, >q} ← W_{:, >q} − e · H⁻¹_{q, >q}
   ```
   nudging remaining full-precision columns to partially cancel the error just introduced, using the Hessian's off-diagonal terms to determine how much each should shift.
4. Move to the next column, repeat.

Uses a Cholesky decomposition of `H⁻¹` computed once up front for numerical stability across matrices with tens of thousands of columns, avoiding a fresh full inverse at every column. Net effect: error is actively redistributed and partially cancelled rather than accumulating independently, giving lower reconstruction error at 4-bit and even 3-bit than naive round-to-nearest.

### 10.4 AWQ, the algorithm in full

**Observation:** weight magnitude alone poorly predicts importance — what matters is how large the corresponding *activations* are, since a weight multiplied against a large, consistently-occurring activation has outsized influence regardless of the weight's own magnitude.

**Step 1 — identify salient channels.** Run calibration data through the model; for each input channel `c`, record `s_x = average(|X_{:,c}|)`. Largest-`s_x` channels (often ~0.1–1% of channels) are deemed salient.

**Step 2 — per-channel rescaling.** Choose `α_c ≥ 1` per channel (larger for salient channels, found via small grid search minimizing actual calibration output error):

```
W'_{:,c} = W_{:,c} × α_c        (scale weight UP)
X'_{:,c} = X_{:,c} / α_c        (scale matching activation DOWN)
```

Since `W'·X' = (W·α)·(X/α) = W·X`, the mathematical product — and the layer's actual output — is exactly preserved before any quantization happens.

**Step 3 — quantize as normal.** Standard round-to-nearest (Section 10.1) applied to `W'`. Because salient channels were scaled up first, their values occupy a larger numeric range relative to the fixed step size `s`, so the absolute error bound `≤ s/2` becomes a smaller *relative* error exactly where it matters most, while unimportant channels absorb comparatively more relative error where it matters less.

**Why no Hessian:** AWQ only needs simple activation-magnitude statistics from a calibration pass — cheaper to run than GPTQ, at the cost of relying on "large average activation magnitude" as an importance proxy rather than GPTQ's direct second-order error-minimization objective.

### 10.5 Quantization vs. distillation vs. pruning

| Technique | What it does | Retraining needed? |
|---|---|---|
| Quantization | Same weights, fewer bits | Usually no |
| Distillation | New, smaller model trained to mimic a larger one | Yes, full training run |
| Pruning | Deletes/zeroes weights judged unimportant | Sometimes (fine-tuning helps) |

---

## 11. Compute-Bound vs. Memory-Bound — Formalized via the Roofline Model

### 11.1 The roofline model

```
Arithmetic Intensity (AI) = FLOPs performed / Bytes moved from memory
```

A GPU has peak compute throughput (FLOPs/sec) and peak memory bandwidth (bytes/sec). The **ridge point** is where these intersect: `AI_ridge = peak_FLOPs / peak_bandwidth`. For `AI < AI_ridge`, performance is capped by memory bandwidth (memory-bound); for `AI > AI_ridge`, capped by compute (compute-bound) — regardless of how much faster memory could theoretically be.

### 11.2 Applying it to prefill and decode

- **Prefill's AI:** FLOPs scale `O(n × weight_size)` while bytes moved for weights stay fixed at `O(weight_size)` (loaded once, reused for all `n` tokens) — AI scales upward with `n`, easily exceeding most GPUs' ridge point for reasonable prompt lengths. Hence compute-bound.
- **Decode's AI (single request):** `n=1`, so `O(weight_size)` bytes moved supports only `O(weight_size)` FLOPs — AI collapses to a small constant, far below the ridge point. Hence memory-bound.
- **Decode's AI under batching (`B` requests):** the same weight bytes (loaded once regardless of batch size) now support `O(B × weight_size)` FLOPs, since `B` requests share the one weight load. AI scales linearly with `B` — this is the precise mathematical reason batching raises throughput: it pushes AI rightward toward (or past) the ridge point, until KV-cache reads (which *do* scale with `B`) start to dominate memory traffic instead.

### 11.3 Worked ridge-point calculation

GPU: 3,000 GB/s bandwidth, 300 TFLOPs peak compute → `AI_ridge = 300×10¹² / 3,000×10⁹ = 100 FLOPs/byte`. A single-token decode step's AI is far below 100 (dominated by loading ~14 GB of FP16 weights for comparatively tiny matrix-vector arithmetic) — confirming the memory-bound classification quantitatively, not just qualitatively.

---

## 12. Speculative Decoding — Exact Rejection-Sampling Algorithm and Speedup Formula

### 12.1 Setup

A small **draft model** `q(x)` and the large **target model** `p(x)`, whose distribution you want to sample from exactly, while invoking `p` far less often than once per token.

### 12.2 The algorithm

1. **Draft proposes:** the draft model autoregressively generates `γ` candidates `x_1,...,x_γ`, recording `q(x_i)` for each.
2. **Target verifies in parallel:** the target model processes the entire draft sequence in one forward pass (works because verification, unlike generation, has no sequential dependency once the candidates exist — exactly like prefill), yielding `p(x_i)` for every position from one pass.
3. **Accept/reject in order**, for `i=1..γ`:
   ```
   if p(x_i) ≥ q(x_i): accept unconditionally
   else: accept with probability p(x_i)/q(x_i); otherwise reject and stop
   ```
4. **On rejection, resample from the residual:**
   ```
   p_residual(x) = max(0, p(x) - q(x)) / Σ_x' max(0, p(x') - q(x'))
   ```
5. **Bonus token on full acceptance:** if all `γ` are accepted, sample one more token directly from `p` at position `γ+1`, since the target's step-2 pass already computed that distribution for free.

### 12.3 Proof sketch: why this is exact

Case A (`p(x) ≥ q(x)`): drawn from `q` with probability `q(x)`, accepted unconditionally → `P(accept, output x) = q(x)`. The shortfall `p(x)-q(x)` is made up whenever some *other* drafted token is rejected and residual resampling happens to land on `x`.

Case B (`p(x) < q(x)`): drawn from `q` with probability `q(x)`, accepted with probability `p(x)/q(x)` → `P(accept, output x) = q(x) × p(x)/q(x) = p(x)` exactly — no shortfall to make up, since acceptance was deliberately capped.

Summed across both routes to producing `x` (direct acceptance + residual resampling after some other rejection), by construction of the residual distribution the total works out to exactly `p(x)` for every `x` — the standard rejection-sampling theorem, applied per token position. The practical result: output is statistically indistinguishable from running the target model alone, the slow way — no quality trade-off, only a latency one.

### 12.4 Expected speedup, derived

Let `α` = expected per-token acceptance rate (empirically ~0.6–0.9 for a well-matched pair):

```
E[tokens per target invocation] = (1 - α^(γ+1)) / (1 - α)
```

| α | γ | E[tokens/invocation] | Speedup |
|---|---|---|---|
| 0.5 | 4 | 1.94 | ~1.9x |
| 0.7 | 4 | 2.77 | ~2.8x |
| 0.9 | 4 | 4.10 | ~4.1x |
| 0.9 | 8 | 6.51 | ~6.5x |

Longer `γ` only pays off if `α` is high enough — at low acceptance, longer drafts mostly waste discarded work; at high acceptance, they compound the speedup. Production systems often tune `γ` dynamically from observed recent acceptance rates.

Viewed through the roofline lens (Section 11): the target's single verification pass over `γ` tokens loads its weights once but reuses that load across `γ` tokens' worth of computation — pushing decode's arithmetic intensity up toward the ridge point, exactly like batching does, except the extra "tokens per weight-load" comes from one request's own speculative candidates rather than pooling concurrent users.

---

## 13. Tensor, Pipeline, and Data Parallelism — Exact Sharding Mechanics

### 13.1 Tensor parallelism (Megatron-style)

For a feedforward layer `Y = GeLU(X·A)·B`: matrix `A` (`[d_model, d_ff]`) is split **column-wise** across `p` GPUs — each holds `A_i` (`[d_model, d_ff/p]`) and independently computes `GeLU(X·A_i)`, needing no cross-GPU communication since each GPU has the full `X`. Matrix `B` (`[d_ff, d_model]`) is split **row-wise** to match — each GPU holds `B_i` (`[d_ff/p, d_model]`) and computes a partial output `(GeLU(X·A_i))·B_i`. These partial outputs must then be summed across all `p` GPUs — one **all-reduce** per layer. Attention layers shard analogously by splitting heads across GPUs (each owns a subset end-to-end, avoiding communication *within* attention itself, with an all-reduce needed only after the output projection `W_O`). This is why tensor parallelism needs high-bandwidth, low-latency interconnects (NVLink/NVSwitch) — at least two all-reduces per transformer layer, across potentially over a hundred layers.

### 13.2 Pipeline parallelism

Layers `1..L` are partitioned into `p` contiguous stages, one GPU per stage (GPU 1: layers 1–10, GPU 2: layers 11–20, etc.). A forward pass flows: GPU 1 computes its layers, sends its activation output to GPU 2, which computes its layers, and so on. Communication per stage boundary is just one activation tensor — far less data than tensor parallelism's per-layer all-reduce, but incurred less frequently.

**Pipeline bubbles, precisely:** with only one micro-batch in flight, GPU 2 sits idle while GPU 1 processes it, and GPU 1 sits idle once it's forwarded that micro-batch — utilization is `1/p` in the worst case. Splitting into `m` micro-batches and pipelining them reduces the bubble fraction to approximately:

```
bubble_fraction ≈ (p - 1) / (m + p - 1)
```

Larger `m` amortizes the fixed pipeline-fill/drain overhead across more useful work.

### 13.3 Data parallelism

The entire model (possibly itself tensor/pipeline-sharded internally) is replicated across `p` independent GPU groups. Requests are routed (round-robin or least-loaded) to whichever replica has spare capacity, each running its own independent continuous-batching scheduler (Section 7). No inter-replica communication is needed at inference time (unlike training's gradient synchronization) — the simplest, cheapest way to add serving capacity once a single replica already fits comfortably.

### 13.4 Combining strategies (3D parallelism)

Large-scale multi-node deployments typically combine all three: tensor parallelism within a node (fast NVLink for frequent all-reduces), pipeline parallelism across a small number of nodes (fitting models too large for one node, using the slower inter-node network only for infrequent activation handoffs), and data parallelism across many such node-groups (scaling total throughput with additional full replicas).

---

## 14. Disaggregated Prefill/Decode Serving — The KV-Transfer Mechanics

Once prefill and decode are recognized as having opposite resource profiles (Section 11), running them on separate GPU pools requires physically transferring the freshly-computed KV cache from a "prefill worker" to a "decode worker" after prefill completes.

**Mechanics:** the prefill worker completes its parallel forward pass (Section 2), producing K/V tensors for all `n` prompt tokens across all `L` layers — exactly Section 3.1's bytes. This data transfers over a high-speed interconnect (RDMA over InfiniBand or similar — standard networking would itself become the bottleneck) directly into the decode worker's paged memory pool (Section 5), populating a fresh block table mirroring the prefill worker's blocks. Decode then proceeds entirely locally per Section 7's loop.

**Why the transfer cost is worth paying:** it's a one-time cost per request (e.g., transferring the ~2 GB figure from Section 3.2's worked example, well under the timescale of the many decode iterations that follow), in exchange for eliminating the interference where a large prefill's compute-bound burst would otherwise delay other requests' decode iterations on a *shared* pool — the same problem chunked prefill (Section 8) addresses via scheduling, but disaggregation solves architecturally. This trade-off becomes worthwhile specifically at scale, where the fleet is large enough to dedicate separate hardware pools and request volume is high enough that scheduling interference would otherwise be persistent.

---

## 15. Cross-Technique Interactions — How These Compound

None of the above operate in isolation; production systems compose nearly all of them simultaneously.

- **Paged attention (Sec. 5) is a prerequisite for continuous batching (Sec. 7)** — dynamic per-iteration admission/eviction is only cheap because blocks can be allocated/freed individually rather than requiring contiguous re-layout.
- **Prefix caching (Sec. 6) reduces the effective prefill workload chunked prefill (Sec. 8) needs to schedule** — a request with a fully-cached prefix may need zero prefill chunks at all, going straight to decode.
- **GQA (Sec. 4) and KV-cache quantization (Sec. 10.6-equivalent) both shrink the same term in Section 3.1's formula** — they stack multiplicatively (e.g., GQA's 4x reduction × INT8 KV cache's 2x reduction = 8x smaller cache than FP16 MHA), directly increasing how many concurrent requests' blocks fit in the allocator's pool, raising achievable batch size (Sec. 7), which per Section 11's roofline argument raises achieved arithmetic intensity and therefore throughput.
- **Speculative decoding (Sec. 12) and continuous batching (Sec. 7) optimize different axes** (single-request latency vs. aggregate throughput) and can conflict operationally — speculative decoding's verification pass processes `γ+1` tokens per request per round instead of 1, consuming more of the batch's compute budget per request per iteration, so systems using both must account for this in admission control (Sec. 7.4) rather than treating batch capacity as a fixed token-count budget.
- **Tensor parallelism (Sec. 13.1) changes the constants in Section 11's roofline analysis** — splitting weights across `p` GPUs reduces bytes each individual GPU must move per weight load, but introduces all-reduce communication as a new critical-path term, so the effective ridge point must be reanalyzed per-GPU rather than assuming single-GPU formulas apply unchanged.

---

## 16. Mechanics-Accurate Pseudocode

```python
# ============================================================
# LLM INFERENCE ENGINE — mechanics-accurate pseudo-code
# ============================================================

BLOCK_SIZE = 16          # tokens per KV-cache page (Sec. 5)
GROUP_SIZE = 128         # weights per quantization group (Sec. 10.2)

class Block:
    def __init__(self, physical_id):
        self.physical_id = physical_id
        self.keys = None          # [BLOCK_SIZE, H_kv, d_head], possibly INT8-quantized
        self.values = None
        self.ref_count = 0        # for copy-on-write sharing (Sec. 5.4)

class BlockAllocator:
    def __init__(self, total_blocks):
        self.free_list = list(range(total_blocks))
        self.blocks = [Block(i) for i in range(total_blocks)]

    def allocate(self):
        pid = self.free_list.pop()
        self.blocks[pid].ref_count = 1
        return self.blocks[pid]

    def free(self, block):
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_list.append(block.physical_id)


class Request:
    def __init__(self, prompt_tokens, max_new, stop_seqs, temperature, top_p):
        self.prompt_tokens = prompt_tokens
        self.generated = []
        self.max_new = max_new
        self.stop_seqs = stop_seqs
        self.temperature = temperature
        self.top_p = top_p
        self.block_table = []          # logical -> physical Block (Sec. 5.2)
        self.state = "WAITING"


# ---------------- ATTENTION CORE (Sec. 1) ----------------
def attention(Q, K, V, causal_mask):
    scores = (Q @ K.T) / sqrt(d_k)              # Sec. 1.2, term 1-2
    scores = scores + causal_mask               # Sec. 1.3, -inf above diagonal
    weights = softmax(scores, axis=-1)           # Sec. 1.2, term 3
    return weights @ V                            # Sec. 1.2, term 4


# ---------------- PREFIX-CACHE LOOKUP (Sec. 6) ----------------
def try_prefix_match(request, global_hash_table):
    matched_blocks = []
    running_hash = None
    for chunk in chunk_tokens(request.prompt_tokens, BLOCK_SIZE):
        running_hash = chained_hash(running_hash, chunk)      # Sec. 6.1
        if running_hash in global_hash_table:
            block = global_hash_table[running_hash]
            block.ref_count += 1
            matched_blocks.append(block)
        else:
            break
    request.block_table.extend(matched_blocks)
    tokens_to_prefill = request.prompt_tokens[len(matched_blocks) * BLOCK_SIZE:]
    return tokens_to_prefill


# ---------------- CHUNKED PREFILL (Sec. 8) ----------------
def prefill_chunk(model, request, chunk_tokens, allocator, hash_table):
    hidden, keys, values = model.forward_parallel(chunk_tokens, kv_cache=request.block_table)  # Sec. 2
    for chunk in split_into_blocks(keys, values, BLOCK_SIZE):
        block = allocator.allocate()                          # Sec. 5.3
        block.keys, block.values = chunk.k, chunk.v
        block.keys, block.values = quantize_kv(block.keys, block.values)   # Sec. 10, optional
        request.block_table.append(block)
        hash_table[chained_hash_of(block)] = block             # register for future prefix reuse
    return hidden[-1]


# ---------------- SAMPLING (Sec. 9) ----------------
def sample(logits, temperature, top_p, generated_so_far, rep_penalty=1.2):
    logits = apply_repetition_penalty(logits, generated_so_far, rep_penalty)
    probs = softmax(logits / max(temperature, 1e-5))
    sorted_p, sorted_ids = sort_descending(probs)
    cum = cumulative_sum(sorted_p)
    m_star = first_index_where(cum >= top_p) + 1        # Sec. 9, top-p step 3
    kept_ids, kept_p = sorted_ids[:m_star], renormalize(sorted_p[:m_star])
    return weighted_random_choice(kept_ids, kept_p)


# ---------------- SPECULATIVE DECODING (Sec. 12) ----------------
def speculative_round(draft_model, target_model, request, gamma=4):
    draft_tokens, draft_probs = [], []
    for _ in range(gamma):
        logits = draft_model.forward_single(request)
        tok = sample(logits, request.temperature, request.top_p, request.generated)
        draft_tokens.append(tok)
        draft_probs.append(softmax(logits)[tok])
        request.generated.append(tok)

    target_logits = target_model.forward_parallel_verify(request, draft_tokens)  # Sec. 12.2 step 2
    accepted = []
    for i, (tok, q_prob) in enumerate(zip(draft_tokens, draft_probs)):
        p_prob = softmax(target_logits[i])[tok]
        if p_prob >= q_prob or random() < (p_prob / q_prob):        # Sec. 12.2 step 3
            accepted.append(tok)
        else:
            residual = positive_part(softmax(target_logits[i]) - softmax_full(draft_probs))  # Sec. 12.2 step 4
            replacement = weighted_random_choice_from_distribution(renormalize(residual))
            accepted.append(replacement)
            request.generated = request.generated[: len(request.generated) - gamma + i] + [replacement]
            return accepted
    bonus = sample(target_logits[gamma], request.temperature, request.top_p, request.generated)  # step 5
    accepted.append(bonus)
    return accepted


# ---------------- CONTINUOUS BATCHING SCHEDULER (Sec. 7) ----------------
def scheduler_loop(model, allocator, hash_table, waiting_queue, max_batch):
    running = []
    while True:
        # Step 1: evict finished, free their blocks
        still_running = []
        for r in running:
            if should_stop(r):
                for block in r.block_table:
                    allocator.free(block)                     # Sec. 5.3
                notify_complete(r)
            else:
                still_running.append(r)
        running = still_running

        # Step 2: greedy admission with watermark (Sec. 7.4)
        while len(running) < max_batch and waiting_queue.has_next() \
                and allocator.free_blocks() > RESERVED_WATERMARK:
            r = waiting_queue.pop()
            remaining_tokens = try_prefix_match(r, hash_table)          # Sec. 6
            for chunk in chunk_tokens(remaining_tokens, CHUNK_SIZE):    # Sec. 8
                prefill_chunk(model, r, chunk, allocator, hash_table)
            r.state = "RUNNING"
            running.append(r)

        if not running:
            sleep_briefly(); continue

        # Step 3: one batched decode iteration — amortized weight load (Sec. 11)
        batch_logits = model.forward_batch_single_step(
            [r.block_table for r in running],
            [r.generated[-1] if r.generated else r.prompt_tokens[-1] for r in running]
        )

        # Step 4-5: per-request sampling, cache growth, streaming
        for r, logits in zip(running, batch_logits):
            tok = sample(logits, r.temperature, r.top_p, r.generated)
            r.generated.append(tok)
            append_new_kv_to_block_table(r, allocator, tok)   # allocate new block if current is full
            stream_token_to_user(r, tok)
```
