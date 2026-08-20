## Phase 2 — Memory: The KV Cache and Its Optimizations

**Goal:** understand why memory, not compute, is the primary constraint in decode, and the full family of techniques that shrink or reorganize that memory.

- [ ] **The KV cache** — why it exists, the exact per-token byte formula, and doing the arithmetic yourself for a real model size.
- [ ] **MHA vs. MQA vs. GQA** — the head-sharing math, and why GQA became the default in modern open models.
- [ ] **PagedAttention** — the block/page data structure, the allocator algorithm, fragmentation math.
- [ ] **Prefix caching** — the chained-hash matching mechanism, when it actually helps (shared system prompts, few-shot templates).
- [ ] **Sliding window / cache eviction** — bounding memory in long conversations.
- [ ] **KV-cache quantization** — applying the same quantization math (Phase 4) to cached K/V tensors, not just weights.

**Exercise 1:** Write a function that computes KV cache size in bytes given `L`, `H_kv`, `d_head`, precision, and context length. Run it for 3 real open-weight model configs (look up their actual `config.json` on Hugging Face) and compare cache size at 2K, 8K, and 32K context. Compare against the model's own weight size at the same precision — find the crossover point where cache exceeds weights.

**Exercise 2:** Implement a toy block allocator in plain Python: fixed-size blocks, a free-list, `allocate()`/`free()`, and a per-request block table (list of block IDs). Simulate 20 requests arriving with random lengths and confirm your allocator never wastes more than one block's worth of space per request, versus a naive "reserve max-context contiguous buffer" baseline that you also implement for comparison — print the total wasted memory for both.

**Checkpoint question:** Why does copy-on-write matter specifically for beam search, and what exactly gets duplicated versus shared?
