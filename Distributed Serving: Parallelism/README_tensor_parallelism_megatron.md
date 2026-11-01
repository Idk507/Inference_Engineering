# Tensor Parallelism: Megatron-Style Column/Row Sharding

## Core idea

Tensor parallelism splits the tensor operations inside a transformer
layer across multiple GPUs. Megatron-style tensor parallelism is
especially important because it chooses complementary matrix partitions
that minimize communication:

``` text
column-parallel linear
        |
local elementwise / head-local computation
        |
row-parallel linear
        |
SUM collective
```

The key question for locating a collective is:

``` text
Does each GPU own a different slice of the answer,
or a partial contribution to the same answer?
```

Different slices can remain sharded. Partial contributions to the same
output coordinates must be summed.

## 1. Ordinary linear layer

Let:

``` text
X : [B,M]
W : [M,N]
Y : [B,N]

Y = XW
```

`B` is the effective token dimension, `M` is the input hidden dimension,
and `N` is the output dimension.

## 2. Column-parallel weight sharding

Split W along its output dimension:

``` text
W = [W0 | W1 | ... | W(P-1)]
```

Each shard has shape:

``` text
Wi : [M,N/P]
```

The input X is replicated across the tensor-parallel ranks:

``` text
GPU 0: Y0 = XW0
GPU 1: Y1 = XW1
...
```

The logical full output is:

``` text
Y = concat(Y0,Y1,...)
```

No SUM is required because every rank computes a different set of
complete output features.

Therefore:

``` text
column-parallel GEMM
-> sharded output
-> no immediate all-reduce
```

## 3. Why column parallelism avoids a reduction

For output feature j:

``` text
Y[:,j] = sum_i X[:,i] * W[i,j]
```

A rank that owns column j owns every weight required to compute that
output feature. It does not produce a partial sum of feature j; it
produces the complete feature.

That is why column sharding creates independent output slices.

## 4. Local nonlinearities

Suppose:

``` text
H = GELU(XA)
```

and:

``` text
A = [A0 | A1]
```

Then:

``` text
XA = [XA0 | XA1]
```

Since GELU is elementwise:

``` text
GELU(XA)
=
[GELU(XA0) | GELU(XA1)]
```

Each GPU can apply the activation to its own shard without
communicating.

This is what makes the column-parallel first MLP projection useful.

## 5. Row-parallel weight sharding

Now suppose the input is already sharded:

``` text
X = [X0 | X1 | ... | X(P-1)]
```

Partition W along its input dimension:

``` text
W =
[W0
 W1
 ...
 W(P-1)]
```

Then:

``` text
Y
=
XW
=
sum_i XiWi
```

Each GPU computes:

``` text
Yi_partial = XiWi
```

But every `Yi_partial` contributes to the same output coordinates.

Therefore:

``` text
Y = sum_i Yi_partial
```

requires a SUM collective.

If every rank needs the completed Y, the natural abstraction is an
all-reduce.

## 6. Column versus row parallel

  Property         Column parallel          Row parallel
  ---------------- ------------------------ ----------------------------
  Weight split     Output dimension         Input dimension
  Typical input    Replicated               Sharded
  Local result     Different output slice   Partial full-width output
  Immediate SUM?   No                       Yes
  Typical use      MLP up, QKV              MLP down, attention output

Memorize:

``` text
COLUMN:
different output features

ROW:
different contributions to the same output features
```

## 7. Megatron MLP

A simplified transformer MLP is:

``` text
X
|
Linear up
|
GELU
|
Linear down
|
Y
```

with:

``` text
W_up   : [H,F]
W_down : [F,H]
```

Megatron pairs:

``` text
W_up   -> column parallel
W_down -> row parallel
```

For two GPUs:

``` text
                         X [B,H]
                        replicated
                      /            \
                     v              v

GPU 0                                   GPU 1
X @ W_up_0                              X @ W_up_1
    |                                       |
    v                                       v
Z0 [B,F/2]                              Z1 [B,F/2]
    |                                       |
  GELU                                    GELU
    |                                       |
    v                                       v
H0 [B,F/2]                              H1 [B,F/2]
    |                                       |
H0 @ W_down_0                         H1 @ W_down_1
    |                                       |
    v                                       v
Y0_partial [B,H]                       Y1_partial [B,H]
        \                                  /
         +------------ SUM ---------------+
                        |
                        v
                    Y [B,H]
                   replicated
```

Mathematically:

``` text
H0 = GELU(XW_up_0)
H1 = GELU(XW_up_1)

Y
=
H0 W_down_0
+
H1 W_down_1
```

The SUM is exactly where the all-reduce lands.

## 8. Why Megatron does not gather after the up projection

After the column-parallel up projection:

``` text
GPU 0 owns H0
GPU 1 owns H1
```

A naive design could all-gather the full H before the next matrix
multiplication.

Megatron avoids that because the row-parallel down projection can
consume the shards directly.

So instead of:

``` text
column linear
-> all-gather
-> activation
-> linear
-> communication
```

we get:

``` text
column linear
-> local activation
-> row linear
-> all-reduce
```

This delays communication until the algebra genuinely requires
cross-rank summation.

## 9. SwiGLU / gated MLPs

Modern LLMs commonly use a gated MLP:

``` text
gate = X W_gate
up   = X W_up

hidden = SiLU(gate) * up

Y = hidden W_down
```

Both expansion projections can be column-sharded:

``` text
W_gate -> column parallel
W_up   -> column parallel
```

Each rank computes matching local shards:

``` text
gate_i
up_i
```

and locally performs:

``` text
hidden_i
=
SiLU(gate_i) * up_i
```

Then:

``` text
W_down -> row parallel
```

Each rank produces a partial output and the partial outputs are reduced.

The pattern remains:

``` text
column expansion
-> local nonlinear/gating work
-> row contraction
-> SUM
```

## 10. Tensor parallelism in attention

A simplified attention block contains:

``` text
X
|
QKV projection
|
attention heads
|
output projection
|
Y
```

Megatron-style TP uses:

``` text
QKV projection
-> column parallel

attention heads
-> local

output projection
-> row parallel

then
-> all-reduce
```

## 11. Column-parallel QKV

Conceptually:

``` text
QKV = X W_QKV
```

Partition output channels:

``` text
W_QKV
=
[W_QKV_0 | W_QKV_1 | ...]
```

Each rank receives replicated X:

``` text
QKV_i = X W_QKV_i
```

For ordinary head-parallel multi-head attention, this can correspond to
assigning subsets of attention heads to ranks.

Example:

``` text
16 heads
TP = 4

GPU 0 -> heads 0-3
GPU 1 -> heads 4-7
GPU 2 -> heads 8-11
GPU 3 -> heads 12-15
```

No reduction is required after QKV because each rank owns different
Q/K/V channels or heads.

## 12. Local attention heads

Each rank computes attention for its own heads:

``` text
scores_i
=
Q_i K_i^T / sqrt(d_head)

P_i
=
softmax(scores_i)

O_i
=
P_i V_i
```

Ordinary attention heads are independent at this stage, so this work
remains local to each tensor-parallel rank.

## 13. Row-parallel attention output projection

The head outputs form a hidden-dimension partition:

``` text
O =
[O0 | O1 | ...]
```

The output projection is:

``` text
Y = O W_O
```

Row-partition `W_O` compatibly:

``` text
W_O =
[W_O_0
 W_O_1
 ...]
```

Each rank computes:

``` text
Yi_partial = Oi W_O_i
```

and:

``` text
Y = sum_i Yi_partial
```

Therefore the attention output projection creates the other major
all-reduce boundary.

## 14. Attention graph

``` text
                    X
               replicated
                    |
                    v
        COLUMN-PARALLEL QKV
                    |
           +--------+--------+
           |                 |
           v                 v
      local heads        local heads
        GPU 0              GPU 1
           |                 |
           v                 v
          O0                O1
           |                 |
           v                 v
     O0 @ W_O0         O1 @ W_O1
           |                 |
           v                 v
      partial Y0         partial Y1
            \             /
             +---- SUM ---+
                    |
                    v
                    Y
               replicated
```

## 15. Where all-reduces land in a transformer block

A simplified pre-norm block:

``` text
X
|
LayerNorm
|
ColumnParallel(QKV)
|
Local Attention
|
RowParallel(Output Projection)
|
ALL-REDUCE
|
Residual Add
|
LayerNorm
|
ColumnParallel(MLP Up / Gate)
|
Local GELU / SwiGLU
|
RowParallel(MLP Down)
|
ALL-REDUCE
|
Residual Add
|
next block
```

Canonical Megatron-style tensor parallelism therefore has two major
forward reduction boundaries per transformer block:

``` text
1. after attention output projection
2. after MLP down/output projection
```

The exact collective form can differ in newer layouts, but this is the
fundamental graph.

## 16. Why the reductions align with residual boundaries

Residual addition requires:

``` text
Y = X + F(X)
```

If X is replicated but `F(X)` is only a local partial contribution, the
residual is mathematically incorrect.

The row-parallel SUM reconstructs the complete `F(X)`:

``` text
partial F_i
|
SUM
|
complete F(X)
|
X + F(X)
```

Thus the collective naturally appears before the residual boundary in
the straightforward layout.

## 17. Why LayerNorm can remain local

After the row-parallel reduction, each rank has the complete hidden
vector for its tokens.

Therefore ordinary LayerNorm can run independently on each rank.

No TP communication is required for LayerNorm in this
replicated-hidden-state layout.

## 18. Four-GPU MLP example

Suppose:

``` text
H  = 4096
F  = 16384
TP = 4
```

First MLP matrix:

``` text
W_up : [4096,16384]
```

Column shards:

``` text
GPU 0: [4096,4096]
GPU 1: [4096,4096]
GPU 2: [4096,4096]
GPU 3: [4096,4096]
```

Each rank produces:

``` text
[B,4096]
```

intermediate features.

Down matrix:

``` text
W_down : [16384,4096]
```

Row shards:

``` text
GPU 0: [4096,4096]
GPU 1: [4096,4096]
GPU 2: [4096,4096]
GPU 3: [4096,4096]
```

Each rank computes:

``` text
[B,4096] @ [4096,4096]
```

and obtains a partial:

``` text
[B,4096]
```

Then all four partial tensors are SUM-reduced.

## 19. All-reduce semantics

If ranks begin with:

``` text
rank 0 -> Y0
rank 1 -> Y1
...
rank P-1 -> Y(P-1)
```

all-reduce SUM produces:

``` text
Y
=
Y0 + Y1 + ... + Y(P-1)
```

and makes Y available to every participating rank.

Conceptually:

``` text
reduce
+
broadcast
```

Efficient implementations use ring, tree, hierarchical, or
topology-aware algorithms.

## 20. Ring all-reduce traffic intuition

For P ranks and an S-byte tensor, an idealized ring all-reduce can be
viewed as:

``` text
reduce-scatter
+
all-gather
```

Approximate communicated bytes per rank:

``` text
2 * (P-1)/P * S
```

As P grows this approaches:

``` text
2S
```

per rank, while the number of communication stages and synchronization
behavior still matter.

## 21. All-reduce versus reduce-scatter

A row-parallel operation mathematically requires a reduction.

It does not always require that the result immediately become fully
replicated.

A modern implementation may use:

``` text
reduce-scatter
```

to produce a reduced but sharded tensor and keep the following
operations sharded.

Later:

``` text
all-gather
```

may restore a representation when required.

This can reduce activation replication and improve overlap.

But the canonical Megatron mental model remains:

``` text
row-parallel partial contributions
-> cross-rank SUM
```

## 22. Weight-memory benefit

For a TP-sharded weight:

``` text
weight memory per rank
≈
full weight memory / TP degree
```

This is a major reason TP allows large models to fit across
accelerators.

Not every parameter or runtime allocation is necessarily sharded in
exactly this way, so total device memory does not always scale perfectly
as `1/P`.

## 23. Compute scaling

Ideally, partitioning a dense matrix across P ranks gives each rank
approximately:

``` text
1/P
```

of the matrix computation.

But actual layer latency behaves more like:

``` text
T_layer
≈
T_local_compute
+
T_collective
+
T_synchronization
-
T_overlap
```

As P grows, local computation shrinks while communication becomes
increasingly important.

## 24. Why fast interconnects matter

Tensor-parallel communication sits inside every transformer block.

During autoregressive generation, those communication boundaries are
traversed for every generated token.

Therefore high-bandwidth, low-latency links such as NVLink/NVSwitch can
materially improve TP efficiency.

Cross-node TP over slower networking is much harder to scale
efficiently.

A common systems preference is therefore to keep a TP group within a
high-bandwidth connectivity domain when practical.

## 25. Forward/backward duality

Training introduces the backward graph.

For a column-parallel layer:

``` text
forward:
Y_i = X W_i
```

No forward reduction is needed.

But:

``` text
dX_i
=
dY_i W_i^T
```

is only a partial contribution to the input gradient.

Therefore:

``` text
dX
=
sum_i dX_i
```

requires a reduction in backward.

For a row-parallel layer:

``` text
forward:
Y
=
sum_i X_i W_i
```

so forward requires the reduction.

Its local input-gradient shard can then be computed from the completed
output gradient.

A useful simplified view is:

``` text
COLUMN PARALLEL

forward:
replicated X
-> local GEMM
-> sharded Y

backward:
sharded dY
-> local GEMM
-> partial dX
-> SUM
```

``` text
ROW PARALLEL

forward:
sharded X
-> local GEMM
-> partial Y
-> SUM

backward:
replicated dY
-> local GEMM
-> sharded dX
```

## 26. TP versus data parallelism

Data parallelism:

``` text
different examples
full model replica per rank
```

Its major training collective is typically gradient synchronization.

Tensor parallelism:

``` text
same layer execution
different tensor shards
```

Its communication occurs inside the layer's computation graph.

That makes TP much more latency-sensitive.

## 27. TP versus pipeline parallelism

Pipeline parallelism splits layers:

``` text
GPU 0 -> early layers
GPU 1 -> later layers
```

Tensor parallelism splits operations within a layer:

``` text
one GEMM
|
+-> GPU 0 shard
+-> GPU 1 shard
+-> GPU 2 shard
+-> GPU 3 shard
```

Large training and inference systems can combine:

``` text
data parallelism
tensor parallelism
pipeline parallelism
sequence/context parallelism
expert parallelism
```

## 28. Decode implications

Autoregressive decode uses a small token dimension unless many requests
are batched.

Tensor parallelism reduces per-rank weight memory and compute, but the
collectives still occur repeatedly:

``` text
every transformer block
x
every generated token
```

At low batch sizes, collective latency can become highly visible in
TPOT.

This is why increasing TP degree is not automatically an inference
speedup.

## 29. Prefill implications

Prefill processes many prompt tokens together.

Its GEMMs are larger and generally utilize GPUs better.

There is more compute available to amortize collective overhead.

Therefore TP scaling can look much better for prefill than for low-batch
decode.

Always profile the two phases separately.

## 30. Distributed roofline intuition

A normal GPU roofline considers:

``` text
compute throughput
HBM bandwidth
```

Tensor parallelism adds another potential bottleneck:

``` text
interconnect bandwidth / latency
```

So a TP operation can be:

``` text
locally compute-efficient
```

while the distributed layer remains:

``` text
communication-bound
```

## 31. Strong-scaling model

Let:

``` text
T1 = single-GPU runtime
TP = tensor-parallel degree
```

A simplified model is:

``` text
T_TP
≈
T_compute/TP
+
T_collective(TP)
+
T_sync(TP)
```

Speedup:

``` text
S_TP
=
T1/T_TP
```

Parallel efficiency:

``` text
E_TP
=
S_TP/TP
```

Ideal scaling has:

``` text
E_TP = 1
```

Real efficiency declines as communication and smaller local kernels
become significant.

## 32. Communication model

A rough collective latency model is:

``` text
T_comm
≈
startup_latency
+
message_bytes/effective_bandwidth
```

For repeated small decode collectives, startup/synchronization latency
can matter substantially.

For large prefill collectives, bandwidth can dominate.

## 33. Why collectives are on the critical path

If the next operation needs the completed hidden state:

``` text
local row-parallel GEMM
|
SUM collective
|
next dependent operation
```

The next operation cannot correctly proceed before the required
reduction completes.

The collective is therefore a true computation-graph dependency, not
merely background synchronization.

## 34. Communication/computation overlap

Optimized systems try to overlap independent communication and
computation.

Conceptually:

``` text
compute chunk 1
|
launch communication for chunk 1
|
compute chunk 2 while communication progresses
```

For the overlappable portion, runtime can move from:

``` text
T_compute + T_comm
```

toward:

``` text
max(T_compute,T_comm)
```

Perfect overlap is uncommon because dependency boundaries and resource
contention remain.

## 35. Tiny numerical column example

Let:

``` text
X = [x1 x2]
```

and:

``` text
W =
[a b c d
 e f g h]
```

Column shards:

``` text
W0 =
[a b
 e f]

W1 =
[c d
 g h]
```

GPU 0:

``` text
Y0 =
[
x1*a + x2*e,
x1*b + x2*f
]
```

GPU 1:

``` text
Y1 =
[
x1*c + x2*g,
x1*d + x2*h
]
```

Full output:

``` text
Y = [Y0 | Y1]
```

No SUM is necessary.

## 36. Tiny numerical row example

Suppose:

``` text
X = [u v w z]
```

Split:

``` text
X0 = [u v]
X1 = [w z]
```

and:

``` text
W =
[a b
 c d
 e f
 g h]
```

Row shards:

``` text
W0 =
[a b
 c d]

W1 =
[e f
 g h]
```

GPU 0:

``` text
P0 =
[
u*a + v*c,
u*b + v*d
]
```

GPU 1:

``` text
P1 =
[
w*e + z*g,
w*f + z*h
]
```

Full result:

``` text
Y = P0 + P1
```

The cross-rank SUM is mathematically unavoidable because both ranks
contribute to the same output coordinates.

## 37. PyTorch simulation

``` python
from __future__ import annotations

import torch


def column_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Simulate a column-parallel linear layer."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("Expected matrices.")

    if x.shape[1] != weight.shape[0]:
        raise ValueError("Incompatible dimensions.")

    if weight.shape[1] % world_size:
        raise ValueError(
            "Output dimension must divide world size."
        )

    shards = torch.chunk(
        weight,
        world_size,
        dim=1,
    )

    outputs = [
        x @ shard
        for shard in shards
    ]

    return torch.cat(
        outputs,
        dim=1,
    )


def row_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Simulate row parallelism and its SUM reduction."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("Expected matrices.")

    if x.shape[1] != weight.shape[0]:
        raise ValueError("Incompatible dimensions.")

    if x.shape[1] % world_size:
        raise ValueError(
            "Input dimension must divide world size."
        )

    x_shards = torch.chunk(
        x,
        world_size,
        dim=1,
    )

    w_shards = torch.chunk(
        weight,
        world_size,
        dim=0,
    )

    partials = [
        x_i @ w_i
        for x_i, w_i in zip(
            x_shards,
            w_shards,
            strict=True,
        )
    ]

    return torch.stack(
        partials,
        dim=0,
    ).sum(dim=0)


def main() -> None:
    torch.manual_seed(7)

    x = torch.randn(3, 8)
    weight = torch.randn(8, 12)
    reference = x @ weight

    column_result = column_parallel_linear(
        x,
        weight,
        world_size=2,
    )

    row_result = row_parallel_linear(
        x,
        weight,
        world_size=2,
    )

    print(
        "Column correct:",
        torch.allclose(
            reference,
            column_result,
            atol=1e-5,
        ),
    )

    print(
        "Row correct:",
        torch.allclose(
            reference,
            row_result,
            atol=1e-5,
        ),
    )


if __name__ == "__main__":
    main()
```

Expected:

``` text
Column correct: True
Row correct: True
```

## 38. Distributed row-parallel skeleton

The essential forward operation resembles:

``` python
local_partial = local_x @ local_weight

torch.distributed.all_reduce(
    local_partial,
    op=torch.distributed.ReduceOp.SUM,
    group=tensor_parallel_group,
)

output = local_partial
```

After the collective, each participating rank owns the completed output.

Real implementations use optimized communication wrappers and custom
autograd behavior.

## 39. What to inspect in a profiler

A simplified transformer trace may contain:

``` text
QKV GEMM
attention kernels
output-projection GEMM
NCCL reduction
LayerNorm
MLP up/gate GEMM
activation
MLP down GEMM
NCCL reduction
```

Sweep:

``` text
TP = 1,2,4,8
```

and record:

``` text
local GEMM time
collective time
collective bytes
GPU utilization
SM utilization
HBM bandwidth
interconnect bandwidth
prefill latency
TPOT
throughput
```

If NCCL occupies an increasing fraction of token latency as TP grows,
the system is becoming communication-limited.

## 40. Common mistakes

### Mistake 1: column parallel means splitting the input

No. The weight is split along its output-feature columns.

### Mistake 2: row-parallel output is complete locally

No. Each rank owns only a partial contribution to the same full output.

### Mistake 3: every TP GEMM needs all-reduce

No. Column-parallel output slices are independent.

### Mistake 4: gather immediately after every column projection

This often destroys the communication-saving benefit. Pair the column
projection with a compatible row projection when possible.

### Mistake 5: TP gives perfect P-times speedup

Collectives, synchronization, topology, and smaller local GEMMs prevent
ideal scaling.

### Mistake 6: TP communication resembles data-parallel gradient sync

TP communication is inside the forward layer dependency graph and can
directly affect inference latency.

## 41. Interview-ready answer

``` text
Megatron tensor parallelism pairs column-parallel and row-parallel
linear layers.

In a column-parallel layer, the weight matrix is split along its
output dimension. Every rank receives the same input and computes
different complete output features. Because the ranks own different
slices rather than partial sums, no immediate all-reduce is needed.

The activation can remain sharded through elementwise operations or
independent attention heads.

The next linear layer is row-parallel. Its input dimension and weight
rows are partitioned compatibly. Each rank then computes a partial
contribution to the same final output tensor. Those partial outputs
must be summed, so this is where the tensor-parallel all-reduce lands.

For an MLP the pattern is column-parallel up projection, local
activation, row-parallel down projection, then reduction.

For attention it is column-parallel QKV, local attention heads,
row-parallel output projection, then reduction.

Thus the canonical transformer block has major TP reduction
boundaries after the attention output projection and after the MLP
down projection.
```

## 42. Equations to memorize

Column parallel:

``` text
W =
[W1 | W2 | ... | WP]

Yi = XWi

Y =
concat(Y1,Y2,...,YP)
```

No SUM.

Row parallel:

``` text
X =
[X1 | X2 | ... | XP]

W =
[W1
 W2
 ...
 WP]

Yi_partial = XiWi

Y =
sum_i Yi_partial
```

SUM required.

MLP:

``` text
Hi
=
activation(X W_up_i)

Y
=
sum_i Hi W_down_i
```

Attention:

``` text
QKV_i
=
X W_QKV_i

Oi
=
Attention(Qi,Ki,Vi)

Y
=
sum_i Oi W_O_i
```

## 43. One-line rule for locating the collective

Ask:

``` text
different slices of the answer?
-> keep sharded / concatenate logically

partial contributions to the same answer?
-> SUM reduction
```

That rule is enough to derive where most Megatron-style tensor-parallel
collectives belong.

## 44. Complete transformer mental model

``` text
REPLICATED HIDDEN STATE
        |
     LayerNorm
        |
COLUMN-PARALLEL QKV
        |
 SHARDED HEADS
        |
 LOCAL ATTENTION
        |
ROW-PARALLEL OUTPUT
        |
  PARTIAL RESULTS
        |
   SUM COLLECTIVE
        |
 COMPLETE HIDDEN
        |
  RESIDUAL ADD
        |
     LayerNorm
        |
COLUMN-PARALLEL UP/GATE
        |
SHARDED INTERMEDIATE
        |
 LOCAL GELU/SwiGLU
        |
ROW-PARALLEL DOWN
        |
  PARTIAL RESULTS
        |
   SUM COLLECTIVE
        |
 COMPLETE HIDDEN
        |
  RESIDUAL ADD
        |
    NEXT BLOCK
```

## Conclusion

Megatron-style tensor parallelism is best understood as
communication-aware matrix algebra.

Column sharding creates different output features:

``` text
replicated input
-> independent output shards
```

so no immediate SUM is necessary.

Row sharding creates different contributions to the same output:

``` text
sharded input
-> partial full-width results
```

so a cross-rank SUM is mathematically required.

Megatron deliberately pairs these two layouts:

``` text
column-parallel projection
-> useful local computation
-> row-parallel projection
-> reduction
```

For the MLP:

``` text
column up/gate
-> local activation
-> row down
-> reduction
```

For attention:

``` text
column QKV
-> local heads
-> row output projection
-> reduction
```

This places the major tensor-parallel reductions after the attention
output projection and MLP down projection rather than after every
sharded matrix multiplication.

The systems trade-off is then clear:

``` text
higher TP
-> less weight memory and local compute per GPU

but
-> more communication sensitivity
-> smaller local GEMMs
-> collectives on the token critical path
```

The best TP degree is therefore the degree that satisfies memory and
compute requirements while keeping distributed communication efficient.
