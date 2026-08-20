# The LLM Inference Engineering Roadmap

A topic-by-topic curriculum, ordered so each stage builds on the one before it. Each module lists what to learn, why it matters, a hands-on exercise to actually prove you understand it, and how it connects to what comes next. Treat the checkboxes as a progress tracker.

**How to use this:** go top to bottom. Don't skip the hands-on exercises — this field is almost entirely about mechanics and numbers, and reading about a KV cache is not the same as watching one blow past your GPU's memory limit and figuring out why. Rough total time investment for someone comfortable with Python and basic deep learning: **8–14 weeks** at a few hours a day, faster if you already know PyTorch well.

---

## Phase 0 — Prerequisites (skip anything you already know)

You need these before any of the rest of this roadmap will make sense.

- [ ] **Linear algebra refresher**: matrix multiplication, dot products, vector norms. You'll be reading FLOP-cost derivations constantly.
- [ ] **Basic PyTorch fluency**: tensors, `nn.Module`, writing a forward pass by hand, running things on a GPU (`.to("cuda")`), understanding `torch.no_grad()`.
- [ ] **How a transformer works at a high level**: embeddings → transformer blocks → output head. If you've never trained even a tiny transformer, build one first (see exercise).
- [ ] **Basic GPU/CUDA mental model**: what HBM vs. SRAM is, what "kernel launch" means, why GPUs like parallel work. You don't need to write CUDA yet — just know these terms exist and roughly what they mean.

**Exercise:** Implement a minimal decoder-only transformer (Karpathy's nanoGPT is the standard reference) from scratch and train it on a tiny dataset (Shakespeare, tiny Shakespeare, or similar). Get it generating text, even badly. This gives you a real model to optimize later instead of a black box.

**Checkpoint question:** Can you draw, from memory, the shape of every tensor as it flows through one transformer block? If not, stay here.

---

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

---

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

---

## Phase 3 — Scheduling and Batching

**Goal:** understand how a server keeps a GPU busy across many concurrent users, and why this is the single highest-leverage lever for throughput.

- [ ] **Static batching** — and precisely why it wastes GPU time.
- [ ] **Continuous (in-flight) batching** — the scheduler state machine (waiting/running states), the per-iteration admission/eviction loop.
- [ ] **Admission control** — watermarks, preventing KV-pool over-commitment.
- [ ] **Chunked prefill** — why a long prompt's prefill needs to be split across iterations, the interleaving pattern.
- [ ] **Disaggregated prefill/decode serving** — separating compute-bound and memory-bound workloads onto different hardware pools, the KV-transfer mechanics.

**Exercise:** Build a toy scheduler simulation (no real model needed — simulate "compute" with `time.sleep()` or a token-count-based fake cost function). Model requests arriving via a Poisson process with random target lengths. Implement both a static-batching scheduler and a continuous-batching scheduler on top of your Phase-2 block allocator, and measure: total GPU-idle-time, average time-to-first-token, and total throughput (tokens/sec) for both. You should see continuous batching win clearly, and be able to explain *why*, quantitatively, from your own simulation's numbers.

**Checkpoint question:** Why does continuous batching depend on paged memory (Phase 2) actually working — what would break if you tried continuous batching on top of naive contiguous pre-allocation?

---

## Phase 4 — Quantization and Model Compression

**Goal:** understand how models get shrunk, the exact math behind it, and the difference between the major methods.

- [ ] **The affine quantization formula** — scale, zero-point, symmetric vs. asymmetric.
- [ ] **Granularity** — per-tensor vs. per-channel vs. per-group, and why it matters for error.
- [ ] **GPTQ** — the Hessian-based, column-by-column error-compensation algorithm.
- [ ] **AWQ** — activation-aware channel rescaling.
- [ ] **GGUF / llama.cpp-style quantization** — practical, consumer-hardware-oriented formats.
- [ ] **Quantization vs. distillation vs. pruning** — know the difference cold; these get confused constantly.

**Exercise 1:** By hand (NumPy, no library), implement symmetric per-tensor quantization and per-channel quantization for a random weight matrix with a few artificial outlier values injected. Measure and compare reconstruction error (`mean squared error` between original and dequantized) for both granularities. Confirm per-channel wins, and explain why using your own numbers.

**Exercise 2:** Take a real small open-weight model (something that fits on whatever GPU you have, e.g., a 1–3B model) and quantize it with two different libraries/methods — e.g., `bitsandbytes` INT8 and `AutoGPTQ` or `AutoAWQ` INT4. Compare: model file size on disk, VRAM usage at inference, and output quality on a handful of prompts (even just eyeballing coherence). Write down what you'd trade off in a real deployment decision.

**Checkpoint question:** Why does AWQ not need a Hessian, and what's the actual mathematical trick (the scale-up/scale-down identity) that lets it preserve the layer's output while still reducing quantization error on important channels?

---

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

---

## Phase 6 — Advanced Decoding Strategies

**Goal:** go beyond basic sampling into the algorithms that trade compute for latency or quality.

- [ ] **Sampling strategies in full** — exact top-k/top-p algorithms, beam search's scoring and pruning steps, repetition penalties.
- [ ] **Speculative decoding** — the draft-then-verify architecture, the exact accept/reject/residual-resampling algorithm, and *why it's mathematically exact*, not an approximation.
- [ ] **Expected speedup math** — the acceptance-rate formula, and how draft length interacts with acceptance rate.

**Exercise:** Implement speculative decoding end-to-end using two real Hugging Face models of very different sizes (e.g., a 125M "draft" model and a 1–3B "target" model in the same family/tokenizer). Implement the actual rejection-sampling accept/reject logic yourself (don't just use a library's built-in `assisted_generation` — build it so you *understand* the residual resampling step). Measure real acceptance rate `α` on a handful of prompts, plug it into the expected-speedup formula, and compare the formula's prediction against your actual measured wall-clock speedup.

**Checkpoint question:** Walk through, in your own words, why the residual-resampling step is what makes speculative decoding's output distribution provably identical to the target model's own distribution — not just "close enough."

---

## Phase 7 — Distributed Serving: Parallelism

**Goal:** understand how models too large for one GPU get split across many, and the communication cost each strategy introduces.

- [ ] **Tensor parallelism** — column/row weight sharding (Megatron-style), where all-reduces land in the computation graph.
- [ ] **Pipeline parallelism** — layer partitioning across GPUs, the pipeline-bubble formula, micro-batching.
- [ ] **Data parallelism** — full model replication, why it needs zero inter-replica communication at inference time.
- [ ] **Combining strategies** — "3D parallelism" for very large models across many nodes.

**Exercise:** If you have access to 2+ GPUs, actually shard a model's feedforward layer manually across two devices using raw PyTorch (`torch.distributed`), implementing the column-then-row split and the all-reduce yourself, and confirm the output matches an unsharded version numerically. If you don't have multi-GPU access, do this as a written derivation exercise instead: for a model with `d_model=4096, d_ff=16384`, work out the exact shapes each GPU holds under 4-way tensor parallelism, and calculate how many bytes get all-reduced per layer per forward pass.

**Checkpoint question:** Why does tensor parallelism need much faster interconnects than pipeline parallelism, in terms of *frequency* of communication, not just volume?

---

## Phase 8 — Production Serving Systems

**Goal:** connect everything above to the real frameworks people actually deploy, and understand what each one optimizes for.

- [ ] **vLLM** — install it, serve a real model, and specifically go find where in its docs/source PagedAttention and continuous batching show up. Benchmark it against a naive Hugging Face `generate()` loop for a batch of concurrent requests.
- [ ] **NVIDIA TensorRT-LLM** — read about its kernel-level optimization approach and disaggregated serving support, even if you don't have the hardware to run it.
- [ ] **Hugging Face TGI** — understand where it sits relative to vLLM in terms of ecosystem integration.
- [ ] **llama.cpp / GGUF** — run a quantized model locally on CPU or a consumer GPU; this is the fastest way to *feel* what quantization buys you.
- [ ] **Disaggregated serving in practice** — read case studies from teams running prefill/decode-separated fleets at scale.

**Exercise:** Stand up vLLM locally (or on a cloud GPU instance) serving an open-weight model. Hit it with a burst of concurrent requests using a simple load-testing script, and record throughput and latency. Then do the same thing with a naive single-request-at-a-time Hugging Face loop, and compare the numbers directly. This is the exercise that makes Phases 2–5 click as one coherent story instead of separate topics.

**Checkpoint question:** Given a deployment scenario (e.g., "serve a 70B model to 500 concurrent users with sub-second time-to-first-token" vs. "run a 7B model on a single consumer GPU for local use"), which framework and which techniques from this roadmap would you reach for, and why?

---

## Phase 9 — Cross-Cutting Systems Thinking

**Goal:** stop thinking of these as separate tricks and start thinking of them as one coherent optimization discipline.

- [ ] Be able to explain the entire pipeline (Phase 1) end to end, then explain *why* each technique in Phases 2–7 exists as a targeted fix to either the compute-bound prefill bottleneck or the memory-bound decode bottleneck from Phase 5.
- [ ] Be able to explain how techniques **stack**: e.g., GQA and KV-cache quantization both shrink the same cache-size term and compound multiplicatively; continuous batching depends on paged memory; speculative decoding competes with continuous batching for the same batch compute budget and needs to be accounted for in scheduling.
- [ ] Be able to reason about a **new, unfamiliar optimization technique** you encounter in a paper or blog post by asking: "Is this attacking the compute-bound side or the memory-bound side? What term in the roofline/memory formulas does it change?"

**Exercise:** Pick a recent inference-serving paper or engineering blog post you haven't read yet (vLLM, SGLang, TensorRT-LLM, DeepSpeed-Inference, and similar teams publish these regularly), and before reading the abstract's conclusion, try to predict — from the title and intro alone — whether it's a memory-side or compute-side optimization, and what term in this roadmap's formulas it's likely modifying. Then read it and check yourself.

**Checkpoint question (the real final exam):** Someone hands you a serving workload description — model size, GPU type, expected concurrent users, target latency — and asks you to sketch a serving architecture. Can you walk through, from first principles, which of every technique in this roadmap you'd apply and in what order of priority, and justify each choice with actual numbers rather than "because it's best practice"?

---

## Suggested Pacing

| Phase | Focus | Rough time |
|---|---|---|
| 0 | Prerequisites | 1 week (skip if already fluent) |
| 1 | Lifecycle & attention math | 1 week |
| 2 | KV cache & memory | 1.5 weeks |
| 3 | Batching & scheduling | 1.5 weeks |
| 4 | Quantization | 1.5 weeks |
| 5 | Roofline / performance | 1 week |
| 6 | Advanced decoding | 1 week |
| 7 | Parallelism | 1 week |
| 8 | Production frameworks | 1.5 weeks |
| 9 | Synthesis | ongoing |

---

## Where to Go Deeper on Each Topic

Use these as your primary sources once you've done the exercises above — read them *after* attempting the math/code yourself, not before, so you're checking your own derivation against the source rather than just copying it.

- **Attention & Transformers:** "Attention Is All You Need" (original paper); Karpathy's nanoGPT and "Let's build GPT" walkthrough.
- **FlashAttention:** the FlashAttention and FlashAttention-2 papers.
- **PagedAttention / continuous batching:** the vLLM paper ("Efficient Memory Management for Large Language Model Serving with PagedAttention").
- **Quantization:** the GPTQ paper, the AWQ paper, the llama.cpp/GGUF repo documentation.
- **Speculative decoding:** "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al.) and the DeepMind "accelerating LLM decoding" paper.
- **Parallelism:** the Megatron-LM paper for tensor parallelism; the GPipe paper for pipeline parallelism.
- **Roofline model:** the original Williams et al. roofline paper (not LLM-specific, but the model is directly applicable).
- **Production systems:** vLLM, SGLang, and TensorRT-LLM's own documentation and engineering blogs — these move fast, so treat papers as foundational and blogs/docs as current state of the art.

---
