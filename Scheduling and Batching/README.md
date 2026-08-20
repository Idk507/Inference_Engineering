## Phase 3 — Scheduling and Batching

**Goal:** understand how a server keeps a GPU busy across many concurrent users, and why this is the single highest-leverage lever for throughput.

- [ ] **Static batching** — and precisely why it wastes GPU time.
- [ ] **Continuous (in-flight) batching** — the scheduler state machine (waiting/running states), the per-iteration admission/eviction loop.
- [ ] **Admission control** — watermarks, preventing KV-pool over-commitment.
- [ ] **Chunked prefill** — why a long prompt's prefill needs to be split across iterations, the interleaving pattern.
- [ ] **Disaggregated prefill/decode serving** — separating compute-bound and memory-bound workloads onto different hardware pools, the KV-transfer mechanics.

**Exercise:** Build a toy scheduler simulation (no real model needed — simulate "compute" with `time.sleep()` or a token-count-based fake cost function). Model requests arriving via a Poisson process with random target lengths. Implement both a static-batching scheduler and a continuous-batching scheduler on top of your Phase-2 block allocator, and measure: total GPU-idle-time, average time-to-first-token, and total throughput (tokens/sec) for both. You should see continuous batching win clearly, and be able to explain *why*, quantitatively, from your own simulation's numbers.

**Checkpoint question:** Why does continuous batching depend on paged memory (Phase 2) actually working — what would break if you tried continuous batching on top of naive contiguous pre-allocation?
