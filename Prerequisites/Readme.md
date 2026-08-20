
---

## Phase 0 — Prerequisites (skip anything you already know)

You need these before any of the rest of this roadmap will make sense.

- [ ] **Linear algebra refresher**: matrix multiplication, dot products, vector norms. You'll be reading FLOP-cost derivations constantly.
- [ ] **Basic PyTorch fluency**: tensors, `nn.Module`, writing a forward pass by hand, running things on a GPU (`.to("cuda")`), understanding `torch.no_grad()`.
- [ ] **How a transformer works at a high level**: embeddings → transformer blocks → output head. If you've never trained even a tiny transformer, build one first (see exercise).
- [ ] **Basic GPU/CUDA mental model**: what HBM vs. SRAM is, what "kernel launch" means, why GPUs like parallel work. You don't need to write CUDA yet — just know these terms exist and roughly what they mean.

**Exercise:** Implement a minimal decoder-only transformer (Karpathy's nanoGPT is the standard reference) from scratch and train it on a tiny dataset (Shakespeare, tiny Shakespeare, or similar). Get it generating text, even badly. This gives you a real model to optimize later instead of a black box.

**Checkpoint question:** Can you draw, from memory, the shape of every tensor as it flows through one transformer block? If not, stay here.
