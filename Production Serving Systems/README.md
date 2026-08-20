
## Phase 8 — Production Serving Systems

**Goal:** connect everything above to the real frameworks people actually deploy, and understand what each one optimizes for.

- [ ] **vLLM** — install it, serve a real model, and specifically go find where in its docs/source PagedAttention and continuous batching show up. Benchmark it against a naive Hugging Face `generate()` loop for a batch of concurrent requests.
- [ ] **NVIDIA TensorRT-LLM** — read about its kernel-level optimization approach and disaggregated serving support, even if you don't have the hardware to run it.
- [ ] **Hugging Face TGI** — understand where it sits relative to vLLM in terms of ecosystem integration.
- [ ] **llama.cpp / GGUF** — run a quantized model locally on CPU or a consumer GPU; this is the fastest way to *feel* what quantization buys you.
- [ ] **Disaggregated serving in practice** — read case studies from teams running prefill/decode-separated fleets at scale.

**Exercise:** Stand up vLLM locally (or on a cloud GPU instance) serving an open-weight model. Hit it with a burst of concurrent requests using a simple load-testing script, and record throughput and latency. Then do the same thing with a naive single-request-at-a-time Hugging Face loop, and compare the numbers directly. This is the exercise that makes Phases 2–5 click as one coherent story instead of separate topics.

**Checkpoint question:** Given a deployment scenario (e.g., "serve a 70B model to 500 concurrent users with sub-second time-to-first-token" vs. "run a 7B model on a single consumer GPU for local use"), which framework and which techniques from this roadmap would you reach for, and why?

