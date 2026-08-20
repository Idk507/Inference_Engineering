
## Phase 6 — Advanced Decoding Strategies

**Goal:** go beyond basic sampling into the algorithms that trade compute for latency or quality.

- [ ] **Sampling strategies in full** — exact top-k/top-p algorithms, beam search's scoring and pruning steps, repetition penalties.
- [ ] **Speculative decoding** — the draft-then-verify architecture, the exact accept/reject/residual-resampling algorithm, and *why it's mathematically exact*, not an approximation.
- [ ] **Expected speedup math** — the acceptance-rate formula, and how draft length interacts with acceptance rate.

**Exercise:** Implement speculative decoding end-to-end using two real Hugging Face models of very different sizes (e.g., a 125M "draft" model and a 1–3B "target" model in the same family/tokenizer). Implement the actual rejection-sampling accept/reject logic yourself (don't just use a library's built-in `assisted_generation` — build it so you *understand* the residual resampling step). Measure real acceptance rate `α` on a handful of prompts, plug it into the expected-speedup formula, and compare the formula's prediction against your actual measured wall-clock speedup.

**Checkpoint question:** Walk through, in your own words, why the residual-resampling step is what makes speculative decoding's output distribution provably identical to the target model's own distribution — not just "close enough."
