
## Phase 1 — The Inference Lifecycle and Attention Math

**Goal:** understand exactly what happens between a prompt going in and tokens coming out, down to the equations.

- [ ] **Tokenization** — sub-word tokenization (BPE), why vocabularies are fixed-size, how text becomes integers.
- [ ] **The attention formula itself** — Q, K, V projections, the scaled dot-product formula, why the `√d_k` scaling exists, causal masking, multi-head splitting.
- [ ] **Prefill** — why it's parallel, the FLOP cost derivation, why it's compute-bound.
- [ ] **Decode** — why it's sequential, why each token requires a full pass through the model.
- [ ] **Sampling basics** — greedy, temperature, top-k, top-p, and why the model outputs a distribution rather than a single token.
- [ ] **Stopping criteria** — EOS tokens, max length, stop strings.

**Exercise:** Take your Phase-0 nanoGPT model and write a decode loop entirely by hand — no `model.generate()` shortcuts. Manually implement temperature scaling and top-p filtering as raw NumPy/PyTorch math, not library calls. Confirm your top-p implementation gives a *smaller* candidate pool when you make the model artificially "more confident" (e.g., scale up logits before softmax) and a *larger* pool when you flatten them.

**Reading:** the original "Attention Is All You Need" paper (just the attention section); any FlashAttention blog post/paper for the tiling intuition.

**Checkpoint question:** Given a prompt of length `n`, can you say — without looking it up — why attention FLOPs scale roughly as `O(n²)` and where exactly the `n²` term comes from in the formula?
