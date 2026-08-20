## Phase 5 — Performance Engineering: The Roofline Model

**Goal:** be able to say, with numbers, whether any given operation is compute-bound or memory-bound, and predict how optimizations will actually affect throughput.

- [ ] **Arithmetic intensity** — FLOPs per byte moved.
- [ ] **The roofline model** — peak compute, peak bandwidth, the ridge point.
- [ ] **Why prefill is compute-bound and decode is memory-bound**, formally, not just intuitively.
- [ ] **How batching shifts arithmetic intensity** — the exact reasoning for why batch size raises throughput.
- [ ] **Latency metrics** — TTFT, TPOT, throughput vs. per-user latency trade-offs.
- [ ] **FlashAttention** — the tiling / online-softmax trick, and why it turns a memory-bound sub-operation into a compute-bound one.

**Exercise:** Look up (or benchmark, if you have access) your GPU's peak FLOPs and peak memory bandwidth, and compute its ridge point. Then, for a real model size of your choosing, compute the arithmetic intensity of a single-request decode step vs. a 32-request batched decode step, and check which side of the ridge point each lands on. If you have GPU access, actually benchmark tokens/sec at batch sizes 1, 4, 16, 64 for a real model (Hugging Face `generate()` is fine for this) and see whether your measured throughput curve matches what the roofline math predicts.

**Checkpoint question:** Explain, using the roofline model, exactly why there's a batch size beyond which throughput stops scaling — what's the second-order effect that eventually caps it?
