Here's the full picture, start to end — building up from hardware to execution model to performance intuition.

## 1. The chip itself

A GPU is made of many **Streaming Multiprocessors (SMs)** — think of each SM as a mini-processor with its own small pool of cores, registers, and SRAM. A GPU might have anywhere from dozens to over 100 SMs. All SMs share access to the big pool of HBM.

## 2. Memory hierarchy (HBM vs. SRAM, in full)

From slowest/biggest to fastest/smallest:

- **HBM** (a few GB–80+GB) — main memory, shared across the whole GPU, off-chip, relatively slow.
- **L2 cache** (tens of MB) — shared across all SMs, faster than HBM, automatic (you don't control it directly).
- **SRAM / shared memory / L1** (tens of KB to a couple MB *per SM*) — on-chip, very fast, and in CUDA you can explicitly manage it ("shared memory") as a scratchpad for a group of threads.
- **Registers** — fastest, private to each thread, extremely limited.

Data flows down this hierarchy: HBM → L2 → SRAM/shared → registers → the actual arithmetic unit. Every hop costs latency. Good GPU code loads data as few times as possible from HBM and reuses it heavily once it's close to the compute.

# GPU/CUDA Mental Model

**HBM (High Bandwidth Memory)** — the GPU's main memory, a few GB to 80+GB. Big but relatively slow (hundreds of GB/s to a few TB/s), and *far* from the compute cores. Think of it as the GPU's "hard drive-ish" storage layer — every input tensor, weight, and output lives here between operations.

**SRAM** — small, fast on-chip memory sitting right next to the compute cores (registers, L1 cache, "shared memory"). Only a few MB total, but orders of magnitude faster to access than HBM. The catch: it can't hold much data at once.

**Why this distinction matters:** most GPU performance problems are actually about *moving data between HBM and SRAM*, not about raw compute. A kernel that keeps reloading data from HBM repeatedly is "memory-bound" — the compute cores sit idle waiting for data. Good GPU code tries to load a chunk into SRAM once, do as much math on it as possible while it's there, then write the result back. (This is the whole idea behind things like FlashAttention — restructuring the computation so it reuses data in SRAM instead of round-tripping to HBM constantly.)

---

**Kernel launch** — a "kernel" is a function that runs on the GPU (written in CUDA, Triton, etc.). "Launching" it means the CPU tells the GPU: "run this function, across N threads, now." Each launch has overhead — the CPU has to set things up, hand off to the GPU, and there's some latency before the GPU actually starts crunching. That's why people care about *fusing* operations into fewer, bigger kernels rather than launching a separate tiny kernel for every operation — e.g., doing add-then-relu in one kernel launch instead of two, because each launch's overhead can dominate when the actual math is trivial.

---

**Why GPUs like parallel work** — a CPU has a few cores, each very good at doing one thing fast, handling complex logic, branching, etc. A GPU has thousands of small, simple cores, each pretty weak individually, designed to all do the *same* operation on *different* pieces of data at the same time (SIMD-ish). This is why GPUs are great for things like matrix multiplication or applying an activation function across a whole tensor — the same instruction, applied to millions of independent data points, gets spread across thousands of cores simultaneously. GPUs are bad at workloads full of sequential dependencies or heavy branching, because that parallelism goes to waste — you're back to running things mostly one at a time.

---

**Putting it together:** a typical GPU op is: CPU launches a kernel → kernel pulls a chunk of data from HBM into SRAM → thousands of cores crunch on it in parallel → result gets written back to HBM. Performance work is mostly about (1) minimizing kernel launch overhead by fusing ops, and (2) minimizing HBM traffic by reusing data in SRAM as much as possible before writing back.



## 3. The execution model: threads, blocks, warps, grids

This is the piece that was missing before, and it's central:

- **Thread** — the smallest unit of work; runs one instance of the kernel function.
- **Warp** — a group of 32 threads that execute *in lockstep*, literally the same instruction at the same time (this is what makes GPUs SIMD-like under the hood). If threads in a warp disagree (branch differently — an `if` that goes different ways per thread), the warp has to run both paths sequentially and mask off the inactive threads — this is "warp divergence" and it kills parallelism.
- **Block** — a group of threads (made of multiple warps) that run on the *same SM* and can share that SM's shared memory/SRAM and synchronize with each other.
- **Grid** — the full set of blocks launched together, covering the whole problem (e.g., every element of a tensor).

So a **kernel launch** actually specifies: "run this function as a grid of this many blocks, each with this many threads." The GPU scheduler then spreads blocks across available SMs.

## 4. Kernel launch, in full

- The CPU (**host**) issues the launch; the GPU (**device**) executes it. They're physically separate, connected over PCIe (or NVLink), so data often has to be copied host→device before a kernel runs, and results copied back after — this transfer is itself a bottleneck people forget about.
- Each launch has fixed overhead (microseconds), independent of how much work the kernel does. Many tiny launches → overhead-dominated. This is why **kernel fusion** (combining multiple ops into one launch) matters.
- **Streams**: kernels can be queued asynchronously and overlapped (e.g., overlap compute on one chunk with data transfer of the next), instead of running everything strictly one-after-another.

## 5. Why GPUs like parallel work, in full

- Thousands of simple cores grouped into warps of 32 that execute in lockstep → massive throughput *if* every thread is doing the same operation on different data, with no branching disagreement.
- **Latency hiding**: when one warp is stalled waiting on a memory load, the SM just switches to executing a different warp that's ready. This is why GPUs want *many more threads than cores* — it's not just about parallel compute, it's about always having something ready to run while others wait on memory. This is called **occupancy**.
- GPUs are bad at: sequential dependency chains, heavy branching/divergence, and small workloads that can't fill up all those threads.

## 6. Compute-bound vs. memory-bound (the performance lens tying it together)

Every kernel is limited by one of two things:
- **Memory-bound**: cores spend more time waiting on HBM traffic than computing. Fix: reuse data in SRAM, fuse kernels, reduce redundant reads/writes.
- **Compute-bound**: cores are constantly busy computing; memory isn't the bottleneck. Fix: use faster math paths (e.g., **tensor cores**, specialized hardware units for matrix multiply used heavily in deep learning), better numerical precision (fp16/bf16 vs fp32).

The ratio of math-operations-to-bytes-moved is called **arithmetic intensity**, and it's the standard way people reason about which regime a kernel is in (the "roofline model," if you ever see that term).

---

**End-to-end story:** CPU launches a kernel with a grid of blocks/warps/threads → SM schedules warps, switching between them to hide memory latency → each warp pulls data HBM → SRAM/registers → does the same op across 32 threads at once → writes back to HBM → CPU may launch the next (fused, ideally) kernel or transfer results back over PCIe.
Let's trace through a concrete example end-to-end: **vector addition**, `C = A + B`, where A, B, C are each arrays of, say, 1 million floats.

## Setup (host side)

1. A and B start out in **CPU memory** (RAM).
2. You copy A and B from CPU RAM → **GPU HBM** over PCIe. This transfer itself takes time and is often a real cost people forget to account for.
3. You allocate space in HBM for the output C.

## Kernel launch

You launch the kernel and tell the GPU how to divide the work — e.g., "use blocks of 256 threads, and enough blocks to cover all 1,000,000 elements" (so ~3907 blocks). This launch call goes from CPU → GPU with that fixed overhead we talked about.

## Execution on the GPU

- The GPU scheduler hands out blocks to available **SMs**. Let's say your GPU has 100 SMs — it might run ~100 blocks at once initially, then more as blocks finish.
- Inside each block, threads are grouped into **warps** of 32. Each thread gets one index `i` (its own element to compute), and does the exact same instruction: `C[i] = A[i] + B[i]`.
- To do that, each thread needs `A[i]` and `B[i]`. Those values live in **HBM**, so they get pulled up through L2 → SRAM/registers close to the core. Because vector-add does almost no math per element (just one addition) relative to the two reads and one write it needs, this kernel is **heavily memory-bound** — the cores are basically idle, waiting on HBM bandwidth, not on arithmetic. There's very little to "reuse" in SRAM here, since each element is touched once.
- While some warps are stalled waiting on their memory loads, the SM switches to run other ready warps — that's **latency hiding** in action.

## Writing back

Each thread writes its result `C[i]` back to HBM. Once the kernel finishes, you copy C from GPU HBM back to CPU RAM if you need it there.

---

### Now contrast with matrix multiply

Matmul (`C = A @ B`) is the classic **compute-bound** counter-example: to compute one output element, you need a whole row of A and a whole column of B — but critically, *each row and column gets reused across many output elements*. So naive code that re-reads from HBM for every single multiply-add wastes huge amounts of bandwidth. The standard trick ("tiling") is: load a small tile of A and a tile of B into **SRAM/shared memory** once, then reuse that tile for many multiply-adds before moving to the next tile — dramatically cutting HBM traffic per unit of compute. This is why matmul, done well, becomes compute-bound (limited by how fast the cores/tensor cores can multiply) rather than memory-bound. It's also the direct ancestor of the idea behind FlashAttention I mentioned earlier — tile things so you reuse SRAM, don't keep round-tripping to HBM.

---

This is the mental model you'll want in your head before writing actual CUDA or Triton code — every kernel you write, the first question is "is this going to be memory-bound or compute-bound, and what can I keep in SRAM to avoid redundant HBM traffic?"

