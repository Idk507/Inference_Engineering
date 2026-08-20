
## Phase 9 — Cross-Cutting Systems Thinking

**Goal:** stop thinking of these as separate tricks and start thinking of them as one coherent optimization discipline.

- [ ] Be able to explain the entire pipeline (Phase 1) end to end, then explain *why* each technique in Phases 2–7 exists as a targeted fix to either the compute-bound prefill bottleneck or the memory-bound decode bottleneck from Phase 5.
- [ ] Be able to explain how techniques **stack**: e.g., GQA and KV-cache quantization both shrink the same cache-size term and compound multiplicatively; continuous batching depends on paged memory; speculative decoding competes with continuous batching for the same batch compute budget and needs to be accounted for in scheduling.
- [ ] Be able to reason about a **new, unfamiliar optimization technique** you encounter in a paper or blog post by asking: "Is this attacking the compute-bound side or the memory-bound side? What term in the roofline/memory formulas does it change?"

**Exercise:** Pick a recent inference-serving paper or engineering blog post you haven't read yet (vLLM, SGLang, TensorRT-LLM, DeepSpeed-Inference, and similar teams publish these regularly), and before reading the abstract's conclusion, try to predict — from the title and intro alone — whether it's a memory-side or compute-side optimization, and what term in this roadmap's formulas it's likely modifying. Then read it and check yourself.

**Checkpoint question (the real final exam):** Someone hands you a serving workload description — model size, GPU type, expected concurrent users, target latency — and asks you to sketch a serving architecture. Can you walk through, from first principles, which of every technique in this roadmap you'd apply and in what order of priority, and justify each choice with actual numbers rather than "because it's best practice"?

---
