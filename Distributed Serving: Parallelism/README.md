
## Phase 7 — Distributed Serving: Parallelism

**Goal:** understand how models too large for one GPU get split across many, and the communication cost each strategy introduces.

- [ ] **Tensor parallelism** — column/row weight sharding (Megatron-style), where all-reduces land in the computation graph.
- [ ] **Pipeline parallelism** — layer partitioning across GPUs, the pipeline-bubble formula, micro-batching.
- [ ] **Data parallelism** — full model replication, why it needs zero inter-replica communication at inference time.
- [ ] **Combining strategies** — "3D parallelism" for very large models across many nodes.

**Exercise:** If you have access to 2+ GPUs, actually shard a model's feedforward layer manually across two devices using raw PyTorch (`torch.distributed`), implementing the column-then-row split and the all-reduce yourself, and confirm the output matches an unsharded version numerically. If you don't have multi-GPU access, do this as a written derivation exercise instead: for a model with `d_model=4096, d_ff=16384`, work out the exact shapes each GPU holds under 4-way tensor parallelism, and calculate how many bytes get all-reduced per layer per forward pass.

**Checkpoint question:** Why does tensor parallelism need much faster interconnects than pipeline parallelism, in terms of *frequency* of communication, not just volume?

---
