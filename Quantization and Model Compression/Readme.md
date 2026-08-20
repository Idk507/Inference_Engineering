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
