# 3D Parallelism for Very Large Model Inference

## Combining Tensor, Pipeline, and Data Parallelism Across Many Nodes

## 1. Core architecture

3D parallelism combines three orthogonal dimensions:

``` text
Tensor Parallelism (TP)
= split tensor/layer computation

Pipeline Parallelism (PP)
= split the layer stack

Data Parallelism (DP)
= replicate the complete logical model and split requests
```

For a regular topology:

``` text
Total GPUs = TP * PP * DP
```

The dimensions solve different problems. TP and PP primarily make a
logical model replica fit and provide compute for that replica. DP
primarily creates more independent serving capacity.

The production objective is not to maximize any one parallelism degree.
It is to find the topology that fits the model, maps communication onto
the physical interconnect, leaves KV-cache headroom, satisfies TTFT/TPOT
SLOs, and maximizes SLO-compliant throughput per dollar.

## 2. A 64-GPU example

Assume:

``` text
8 nodes
8 GPUs/node
64 GPUs total
```

One possible topology is:

``` text
TP = 4
PP = 4
DP = 4

4 * 4 * 4 = 64 GPUs
```

One logical replica consumes:

``` text
TP * PP = 16 GPUs
```

and the cluster contains four independent replicas.

``` text
64 GPUs
|
+-- DP Replica 0: 16 GPUs
+-- DP Replica 1: 16 GPUs
+-- DP Replica 2: 16 GPUs
+-- DP Replica 3: 16 GPUs
```

Inside each replica, four pipeline stages each contain a four-GPU
tensor-parallel group.

## 3. The 3D coordinate system

Each GPU can be described by:

``` text
(dp_rank, pp_rank, tp_rank)
```

For `DP=D`, `PP=P`, and `TP=T`:

``` text
dp_rank = 0 ... D-1
pp_rank = 0 ... P-1
tp_rank = 0 ... T-1
```

For example:

``` text
(2, 1, 3)
```

means DP replica 2, pipeline stage 1, tensor-parallel rank 3.

A simple linear rank mapping is:

``` text
global_rank
=
((dp_rank * PP) + pp_rank) * TP + tp_rank
```

Frameworks can use different layouts, but the coordinate model is useful
for understanding communication groups.

## 4. One TP x PP logical replica

With `TP=4` and `PP=4`:

``` text
                     Pipeline direction

Stage 0          Stage 1          Stage 2          Stage 3

 TP0              TP0              TP0              TP0
 TP1      ->      TP1      ->      TP1      ->      TP1
 TP2              TP2              TP2              TP2
 TP3              TP3              TP3              TP3
```

Within each stage, four GPUs jointly execute layer operations. Between
stages, activation tensors move through the pipeline.

DP then replicates this complete structure.

## 5. communication domains

3D parallelism becomes much easier to understand when communication is
separated into domains.

### Tensor-parallel communication

Ranks with the same DP and PP coordinates but different TP coordinates
form a TP group:

``` text
(d=0, p=2, t=0)
(d=0, p=2, t=1)
(d=0, p=2, t=2)
(d=0, p=2, t=3)
```

Depending on the sharding scheme, layer execution may require:

``` text
all-reduce
all-gather
reduce-scatter
```

TP communication is frequent and often latency-sensitive.

### Pipeline-parallel communication

Adjacent pipeline stages transfer activations:

``` text
Stage 0 -> Stage 1 -> Stage 2 -> Stage 3
```

The transfer cost depends on activation size, sequence length,
micro-batching, precision, and network bandwidth.

### Data-parallel communication

For ordinary inference, different DP replicas serve independent
requests.

``` text
required model-forward communication between DP replicas = 0
```

The serving system can still communicate for routing, health, metrics,
deployment, cache locality, or shared application state.

## 6. The physical-topology rule

Not all links are equal.

A common hierarchy is:

``` text
intra-node GPU fabric
>
high-performance inter-node fabric
>
general-purpose network
```

TP performs frequent collectives, so it usually belongs on the fastest
available links.

A strong starting heuristic is:

``` text
Fastest links
-> TP

Next communication tier
-> PP

Weakest coupling
-> DP
```

This is not a universal law. The actual optimum depends on
NVLink/NVSwitch, InfiniBand/RoCE, NIC topology, collective libraries,
model dimensions, and workload. Benchmark the real system.

## 7. Why giant TP groups can fail

Consider:

``` text
TP=64
PP=1
DP=1
```

on 64 GPUs.

Potential problems include:

``` text
cross-node collectives
large synchronization domains
small local matrix shards
collective latency
poor strong scaling
communication dominating compute
large failure domain
```

Increasing TP reduces local computation, but it does not make
communication free.

Eventually:

``` text
compute saved by additional sharding
<
additional communication/synchronization cost
```

and more TP makes performance worse.

## 8. Why PP can be preferable to cross-node TP

Instead of:

``` text
TP=64
PP=1
```

a model might use:

``` text
TP=8
PP=8
```

The same 64 GPUs are used, but TP groups can remain inside eight-GPU
nodes while pipeline stages span nodes.

The communication pattern changes from a huge collective domain to:

``` text
small fast TP collectives
+
pipeline-boundary transfers
```

For many physical topologies this is a much better mapping.

## 9. Why DP should enter once a good replica exists

Suppose the model runs efficiently with:

``` text
TP=8
PP=2
```

One replica requires:

``` text
16 GPUs
```

On 64 GPUs:

``` text
DP = 64 / 16 = 4
```

so:

``` text
TP=8
PP=2
DP=4
```

creates four independent serving replicas.

Compare this with:

``` text
TP=8
PP=8
DP=1
```

Both use 64 GPUs, but the second topology spends every GPU on one
logical model replica.

If the 16-GPU replica already satisfies latency requirements, using the
remaining GPUs for DP can provide much more aggregate serving capacity.

## 10. Model memory

Let:

``` text
W = total model-weight bytes
```

A first-order evenly sharded estimate is:

``` text
WeightMemory_per_GPU
≈
W / (TP * PP)
```

Example:

``` text
W  = 800 GB
TP = 8
PP = 4

WeightMemory_per_GPU
≈
800 / 32
=
25 GB
```

Real HBM usage is higher because it also contains:

``` text
KV cache
runtime workspaces
communication buffers
CUDA graphs
temporary tensors
allocator fragmentation
quantization metadata
```

## 11. DP does not reduce logical-replica model size

With `DP=4`, the logical model is replicated four times.

Approximate aggregate model-weight storage becomes:

``` text
ClusterWeightMemory
≈
DP * W
```

Thus:

``` text
TP + PP
-> model sharding

DP
-> model replication
```

DP does not solve a model-fit problem.

## 12. Minimum model-parallel degree

If the safe per-GPU weight budget is `M_weight`, a crude lower bound is:

``` text
TP * PP
>=
W / M_weight
```

For:

``` text
W = 640 GB
M_weight = 60 GB/GPU
```

we need at least:

``` text
640 / 60
≈
10.67
```

model-parallel GPUs by this simplified bound.

Topology-friendly candidates might instead be:

``` text
TP=8, PP=2 -> 16 GPUs
TP=4, PP=4 -> 16 GPUs
TP=8, PP=4 -> 32 GPUs
```

Memory fit only eliminates impossible configurations. It does not
determine the fastest one.

## 13. Tensor-parallel cost

A simplified communication model is:

``` text
T_collective
≈
T_latency
+
Bytes / EffectiveBandwidth
```

Total TP communication approximately accumulates across collectives:

``` text
T_TP_comm
≈
sum(
    latency
    +
    bytes / bandwidth
)
```

Increasing TP reduces local compute but can increase the relative cost
of communication.

Decode is especially sensitive because local GEMMs become small while
synchronization remains.

## 14. Pipeline communication

A simplified pipeline transfer is:

``` text
T_PP_transfer
≈
T_network_latency
+
ActivationBytes / EffectiveBandwidth
```

PP introduces another cost: pipeline idle time.

For an idealized balanced forward pipeline with:

``` text
P = pipeline stages
M = micro-batches
```

the classic utilization is:

``` text
PipelineEfficiency
=
M / (M + P - 1)
```

and bubble fraction is:

``` text
BubbleFraction
=
(P - 1) / (M + P - 1)
```

Example:

``` text
P=8
M=8

Efficiency = 8/15 ≈ 53.3%
```

With:

``` text
M=64

Efficiency = 64/71 ≈ 90.1%
```

More independent in-flight work amortizes fill/drain bubbles.

Online autoregressive serving is more complex than this idealized
formula because continuous batching, KV state, prefill, decode, and
request completion interact with the pipeline scheduler.

## 15. Data-parallel throughput

If one TP x PP logical replica sustains:

``` text
Q_replica
```

tokens/s under the required SLO, then ideal aggregate throughput is:

``` text
Q_total
≈
DP * Q_replica
```

A more realistic model is:

``` text
Q_total
=
eta_DP * DP * Q_replica
```

where:

``` text
0 < eta_DP <= 1
```

accounts for load imbalance, batch fragmentation, and shared
infrastructure bottlenecks.

## 16. The topology optimization problem

Given:

``` text
G = total GPUs
```

choose positive integers:

``` text
TP=T
PP=P
DP=D
```

such that:

``` text
T * P * D <= G
```

and usually:

``` text
T * P * D = G
```

while satisfying memory, topology, latency, throughput, availability,
and cost constraints.

## 17. Constraint: model fit

A real replica must satisfy:

``` text
WeightMemory
+
KVMemory
+
RuntimeMemory
<=
UsableHBM
```

Do not allocate almost all HBM to weights and then discover that
realistic concurrency cannot fit its KV cache.

## 18. Constraint: topology

For an eight-GPU node, natural initial TP candidates are often:

``` text
TP=1
TP=2
TP=4
TP=8
```

because they can map cleanly inside the node.

Cross-node TP can still be appropriate on fast fabrics, but it must earn
its cost in benchmarks.

## 19. Constraint: latency

Higher TP may reduce local computation but increase collectives.

Higher PP may reduce layers per stage but add stage transfers and
pipeline scheduling overhead.

Higher DP normally does not accelerate an isolated request, but it can
dramatically reduce queueing under load.

Therefore single-request latency and aggregate serving throughput often
prefer different factorizations.

## 20. Constraint: KV-cache capacity

For autoregressive serving:

``` text
HBM
=
weights
+
KV cache
+
runtime
```

A rough KV scaling law is:

``` text
KV bytes
proportional to
layers
*
sequence length
*
KV heads
*
head dimension
*
2
*
bytes per element
```

The factor `2` represents keys and values.

Exact distribution depends on architecture, GQA/MQA, PP partitioning, TP
implementation, and serving engine.

Sometimes extra model sharding beyond the minimum weight-fit requirement
is worthwhile because it leaves enough HBM for useful concurrency.

## 21. Nested mental model

Think of 3D parallelism as:

``` text
Outer:
Data Parallelism

Inside each DP replica:
Pipeline Parallelism

Inside each PP stage:
Tensor Parallelism
```

or:

``` text
DP[
    PP[
        TP[model computation]
    ]
]
```

## 22. Node-aware example

Assume:

``` text
8 GPUs/node
TP=8
PP=2
DP=4
```

One replica uses two nodes.

A clean mapping is:

``` text
Replica 0
Node 0 -> PP stage 0, TP ranks 0-7
Node 1 -> PP stage 1, TP ranks 0-7

Replica 1
Node 2 -> stage 0
Node 3 -> stage 1

Replica 2
Node 4 -> stage 0
Node 5 -> stage 1

Replica 3
Node 6 -> stage 0
Node 7 -> stage 1
```

TP remains intra-node. Only the PP boundary crosses nodes. DP replicas
are independent.

## 23. Compare with cross-node TP

Consider instead:

``` text
TP=16
PP=1
DP=4
```

Each TP group spans two eight-GPU nodes.

Now frequent tensor-parallel collectives cross the node boundary.

If cross-node communication is materially slower than intra-node GPU
fabric, this topology may lose badly even though the GPU count is
identical.

## 24. Communication frequency matters

Do not optimize only total bytes.

A workload containing:

``` text
many small latency-sensitive synchronization points
```

can behave very differently from one containing:

``` text
fewer larger transfers
```

TP collectives appear repeatedly inside transformer execution.

PP communication appears primarily at stage boundaries.

Topology design therefore depends on:

``` text
bytes
frequency
latency sensitivity
synchronization
overlap
```

## 25. Pipeline balance

Suppose:

``` text
Stage 0 = 4 ms
Stage 1 = 5 ms
Stage 2 = 11 ms
Stage 3 = 5 ms
```

The pipeline cadence is constrained by approximately:

``` text
max(stage_time) = 11 ms
```

The faster stages wait.

PP partitions must therefore balance actual compute and memory, not
merely layer counts.

Potential asymmetry includes:

``` text
embedding
LM head
large vocabulary projection
MoE layers
multimodal encoders
cross-attention
special attention patterns
```

## 26. Prefill versus decode

Prefill processes many prompt tokens together and often creates large
GEMMs with relatively high arithmetic intensity.

Decode processes only the next token for each active sequence and is
often more sensitive to:

``` text
HBM bandwidth
KV-cache traffic
collective latency
scheduler overhead
```

A topology that benchmarks well on prefill can perform poorly on decode.

Always benchmark both.

## 27. TP during decode

As TP rises:

``` text
local matrix dimensions shrink
```

while collective synchronization remains.

At some point the local work becomes too small to amortize
communication.

This can worsen TPOT even though more GPUs are participating.

## 28. PP during decode

Each generated token must logically traverse all pipeline stages.

Enough independent active sequences are needed to keep stages useful.

Low concurrency can expose bubbles.

Very high concurrency can increase:

``` text
KV pressure
queueing
tail latency
```

so the scheduler and topology must be tuned together.

## 29. DP during decode

DP is clean because replicas independently advance their active
sequences:

``` text
Replica 0:
A B C D

Replica 1:
E F G

Replica 2:
H I J K
```

No global decode-token barrier is required between replicas.

This makes DP an attractive scale-out dimension once one logical model
replica is efficient.

## 30. A 128-GPU example

Assume:

``` text
16 nodes
8 GPUs/node
128 GPUs
```

Suppose one large model replica works efficiently with:

``` text
TP=8
PP=4
```

Then:

``` text
GPUs/replica = 32
DP = 128 / 32 = 4
```

Final topology:

``` text
TP=8
PP=4
DP=4
```

Physical placement:

``` text
Replica 0:
Node 0 -> Stage 0, TP=8
Node 1 -> Stage 1, TP=8
Node 2 -> Stage 2, TP=8
Node 3 -> Stage 3, TP=8

Replica 1:
Nodes 4-7

Replica 2:
Nodes 8-11

Replica 3:
Nodes 12-15
```

This keeps TP collectives intra-node and primarily uses cross-node
networking for pipeline boundaries.

## 31. Candidate topology search

For 64 GPUs, examples include:

``` text
TP=8, PP=1, DP=8
TP=8, PP=2, DP=4
TP=8, PP=4, DP=2
TP=8, PP=8, DP=1

TP=4, PP=2, DP=8
TP=4, PP=4, DP=4
TP=4, PP=8, DP=2
```

The correct process is:

``` text
enumerate
-> reject memory failures
-> reject bad physical mappings
-> benchmark survivors
-> compare SLO-compliant efficiency
```

## 32. Python candidate enumerator

``` python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    '''A regular TP/PP/DP topology.'''

    tp: int
    pp: int
    dp: int

    @property
    def total_gpus(self) -> int:
        '''Return total GPUs required.'''
        return self.tp * self.pp * self.dp

    @property
    def gpus_per_replica(self) -> int:
        '''Return GPUs consumed by one logical replica.'''
        return self.tp * self.pp


def enumerate_topologies(
    total_gpus: int,
) -> list[Topology]:
    '''Enumerate exact integer factorizations of the GPU count.

    Args:
        total_gpus: Number of available GPUs.

    Returns:
        Candidate regular topologies.

    Raises:
        ValueError: If total_gpus is not positive.
    '''
    if total_gpus <= 0:
        raise ValueError(
            "total_gpus must be positive."
        )

    candidates: list[Topology] = []

    for tp in range(1, total_gpus + 1):
        if total_gpus % tp != 0:
            continue

        remaining = total_gpus // tp

        for pp in range(1, remaining + 1):
            if remaining % pp != 0:
                continue

            dp = remaining // pp

            candidates.append(
                Topology(
                    tp=tp,
                    pp=pp,
                    dp=dp,
                )
            )

    return candidates


def main() -> None:
    '''Display topology-friendly candidates for 64 GPUs.'''
    try:
        for topology in enumerate_topologies(64):
            if topology.tp in {1, 2, 4, 8}:
                print(
                    f"TP={topology.tp:2d} "
                    f"PP={topology.pp:2d} "
                    f"DP={topology.dp:2d} "
                    f"GPUs/replica="
                    f"{topology.gpus_per_replica:2d}"
                )
    except ValueError as exc:
        print(f"Configuration error: {exc}")


if __name__ == "__main__":
    main()
```

This only enumerates possibilities. It cannot select a production
topology without hardware and workload measurements.

## 33. Simplified memory filter

``` python
from __future__ import annotations


def fits_weight_budget(
    model_weight_gb: float,
    tp: int,
    pp: int,
    weight_budget_per_gpu_gb: float,
) -> bool:
    '''Check a first-order per-GPU model-weight budget.

    This intentionally excludes KV cache, runtime workspaces,
    fragmentation, replicated parameters, and pipeline-stage
    imbalance.

    Raises:
        ValueError: If any supplied capacity or degree is invalid.
    '''
    if model_weight_gb <= 0:
        raise ValueError(
            "model_weight_gb must be positive."
        )

    if tp <= 0 or pp <= 0:
        raise ValueError(
            "tp and pp must be positive."
        )

    if weight_budget_per_gpu_gb <= 0:
        raise ValueError(
            "weight_budget_per_gpu_gb must be positive."
        )

    estimated_weight_gb = (
        model_weight_gb / (tp * pp)
    )

    return (
        estimated_weight_gb
        <= weight_budget_per_gpu_gb
    )
```

Use this to reject obviously impossible candidates, not as a substitute
for runtime profiling.

## 34. SLO-aware optimization

The highest raw tokens/s topology may still be wrong.

Example:

``` text
Topology A:
120k tokens/s
p99 TTFT = 8 s

Topology B:
105k tokens/s
p99 TTFT = 1.5 s
```

If:

``` text
p99 TTFT SLO <= 2 s
```

Topology A's extra raw throughput is not useful.

Prefer metrics such as:

``` text
SLO-compliant tokens/s
SLO-compliant requests/s
SLO-compliant tokens/dollar
```

## 35. Why the dimensions are not interchangeable

Increasing TP buys smaller tensor shards and more devices for one layer,
but costs collectives.

Increasing PP buys fewer layers per stage, but costs activation
transfers, stage balancing, and bubbles.

Increasing DP buys independent replicas, but duplicates the model and
can fragment batches.

Therefore:

``` text
TP=8, PP=1, DP=8
```

and:

``` text
TP=8, PP=8, DP=1
```

both use 64 GPUs but represent completely different serving systems.

## 36. Strong scaling versus horizontal scaling

TP and PP mostly attempt to strong-scale one logical model:

``` text
same model execution
more devices
```

Strong scaling eventually encounters communication and synchronization
limits.

DP scales independent work:

``` text
more devices
more independent requests
```

and often scales more cleanly once one replica is efficient.

## 37. The smallest efficient replica principle

Suppose:

``` text
TP=8, PP=2
```

already fits the model and meets latency SLOs.

Increasing to:

``` text
TP=8, PP=4
```

doubles GPUs per replica.

If this produces only a modest performance gain, those GPUs may create
more value as another DP replica.

The central production question is:

``` text
Does putting the next GPU inside this replica
create more useful SLO-compliant capacity than
using it toward another replica?
```

## 38. Global router versus local scheduler

The global router chooses:

``` text
which DP replica?
```

The local scheduler chooses:

``` text
which requests run now?
how many prefill tokens?
how many decode tokens?
which KV blocks?
which requests complete?
```

The router should treat each TP x PP topology as one logical serving
endpoint.

## 39. Continuous batching

Each DP replica runs its own continuous batch.

``` text
Replica 0:
A B C D E

Replica 1:
F G H

Replica 2:
I J K L
```

The replicas do not need synchronized iteration boundaries.

Inside each replica, the scheduler feeds the TP/PP execution topology.

## 40. Chunked prefill

A long prompt can monopolize compute.

Chunked prefill divides it into smaller scheduling units so the local
scheduler can interleave:

``` text
long prompt prefill
short prompt prefill
decode tokens
new admissions
```

This becomes particularly important when a large TP/PP replica serves
heterogeneous online traffic.

## 41. Admission control

Each logical replica has finite:

``` text
KV capacity
active sequence slots
scheduled-token budget
GPU memory
compute capacity
```

The global router can use signals such as:

``` text
KV watermark
queued tokens
active tokens
estimated remaining work
TTFT pressure
```

to avoid overloaded replicas.

## 42. Failure domains

If one required TP rank fails, the entire logical TP/PP replica may
become unavailable.

Other DP replicas can continue.

Therefore DP is also a fault-isolation boundary.

Larger logical replicas have a larger failure surface, which is another
reason not to over-shard one model unnecessarily.

## 43. Shared network contention

Even independent DP replicas can contend physically if their internal PP
or cross-node TP traffic shares network links.

Consider:

``` text
NIC locality
GPU-to-NIC affinity
NUMA
switch oversubscription
network rails
```

Logical independence does not imply physical network isolation.

## 44. Hierarchical collectives

If TP must cross nodes, collective implementations can exploit
hierarchy:

``` text
intra-node reduction
-> inter-node exchange
-> intra-node distribution
```

This can improve performance, but it does not make cross-node TP free.

Measure collective latency and overlap on the real fabric.

## 45. Communication overlap

Some communication can overlap useful computation:

``` text
compute A
-> launch communication A
-> compute B while A transfers
```

Whether overlap actually occurs depends on dependencies, streams,
collective implementation, tensor size, and network behavior.

Use a profiler rather than assuming overlap.

## 46. TP bottleneck signatures

Watch for:

``` text
small local GEMMs
low tensor-core utilization
large collective fraction
collective synchronization gaps
cross-node saturation
TPOT worsening as TP increases
```

A possible response is:

``` text
reduce TP
use PP for remaining model fit
use saved GPUs for DP
```

if benchmarks support it.

## 47. PP bottleneck signatures

Watch for:

``` text
one slow stage
large stage idle gaps
activation-transfer stalls
pipeline bubbles
poor low-concurrency utilization
```

Possible responses:

``` text
rebalance layers
change PP degree
increase useful in-flight work
change placement
improve scheduler
```

## 48. DP bottleneck signatures

Watch for:

``` text
some replicas idle while others queue
tiny batches on every replica
poor cache locality
adapter thrashing
gateway saturation
```

Possible responses:

``` text
token-aware routing
cache-aware routing
adapter-aware routing
fewer replicas
better autoscaling
```

## 49. Production tuning workflow

``` text
1. Measure model memory.

2. Determine the minimum TP*PP needed for weights plus
   realistic KV/runtime headroom.

3. Keep TP aligned with the fastest physical links.

4. Add PP when the model must span additional nodes.

5. Balance pipeline stages by measured runtime, not just
   layer count.

6. Benchmark one logical replica:
   TTFT
   TPOT
   tokens/s
   KV capacity
   collective time
   stage idle time

7. If latency SLOs are satisfied, allocate remaining GPUs
   to DP.

8. Add token-aware replica routing.

9. Sweep concurrency and prompt/output distributions.

10. Compare SLO-compliant throughput per dollar.

11. Profile communication and idle time.

12. Retune TP/PP/DP based on evidence.
```

## 50. Security

A distributed inference deployment expands the trust boundary.

Production controls should include:

``` text
least-privilege workload identity
authenticated service-to-service communication
network segmentation
signed/verified model artifacts
verified runtime images
secret isolation
restricted administrative interfaces
encryption according to the threat model
```

Do not expose internal distributed-runtime or collective-communication
ports to untrusted networks.

## 51. Model integrity

Every DP replica should load the intended configuration.

Track:

``` text
model artifact hash
tokenizer version/hash
quantization configuration
adapter version
runtime version
TP/PP/DP topology
generation configuration
```

A mismatched replica can silently create inconsistent outputs.

## 52. Observability

A request trace should ideally identify:

``` text
request ID
DP replica
PP stage
TP group
node
GPU
model version
queue delay
prefill time
decode time
TTFT
TPOT
output tokens
KV utilization
```

Operators need to distinguish:

``` text
router delay
scheduler delay
TP collective delay
PP transfer delay
GPU compute
```

otherwise 3D-parallel performance failures are difficult to diagnose.

## 53. Metrics by dimension

Tensor parallelism:

``` text
collective latency
collective bytes
collective % of step
synchronization wait
local GEMM efficiency
```

Pipeline parallelism:

``` text
per-stage latency
stage imbalance
bubble/idle fraction
activation-transfer time
```

Data parallelism:

``` text
per-replica queue depth
active tokens
load imbalance
batch size
replica utilization
```

Global service:

``` text
TTFT
TPOT
E2E latency
tokens/s
requests/s
SLO violation rate
cost/token
```

## 54. Capacity planning

Do not size the cluster from weight memory alone.

Include:

``` text
weights
KV cache
peak concurrency
prompt-length distribution
output-length distribution
TTFT target
TPOT target
batching efficiency
failure headroom
autoscaling delay
```

A topology that barely fits weights but cannot hold realistic KV state
is not production-capable.

## 55. Availability headroom

If four DP replicas are required at peak load, exactly four replicas
provide no replica-loss headroom.

Production capacity may need:

``` text
required serving capacity
+
failure reserve
```

depending on availability targets.

DP therefore participates in reliability architecture as well as
throughput scaling.

## 56. Beyond 3D

Modern large models can add dimensions such as:

``` text
Expert Parallelism (EP)
Context/Sequence Parallelism
```

MoE models distribute experts, while long-context systems may shard
sequence-related state.

The enduring principle is not the number of dimensions. It is:

``` text
identify what is sharded
identify which tensors communicate
measure how often they communicate
measure how many bytes move
map the most expensive communication to the best links
```

## 57. Interview-ready explanation

``` text
3D parallelism combines tensor, pipeline, and data parallelism.

Tensor parallelism shards operations inside transformer layers, so
TP ranks jointly execute a layer and require collectives such as
all-reduce or all-gather.

Pipeline parallelism partitions the layer stack. A request passes
through multiple stages, so stage-boundary activations must move
between those stages. Sufficient in-flight work is needed to reduce
pipeline bubbles.

Data parallelism replicates the complete logical TP x PP model.
Different replicas serve independent requests, so inference requires
no model-forward collective between DP replicas.

For a regular topology:

Total GPUs = TP * PP * DP.

In a multi-node system, TP is generally kept on the fastest links
because its collectives are frequent and latency-sensitive. PP can
span layer partitions across nodes. Once one TP x PP replica fits the
model and satisfies latency requirements, remaining GPUs are often
more valuable as DP replicas for throughput.

The goal is therefore not maximum TP or PP. It is the smallest
efficient model-parallel replica that satisfies memory and latency,
followed by enough DP to meet traffic and availability requirements.
```

## 58. Formula sheet

``` text
Total GPUs
=
TP * PP * DP
```

``` text
GPUs per logical replica
=
TP * PP
```

``` text
Approximate weight memory/GPU
≈
ModelWeightBytes / (TP * PP)
```

``` text
Approximate aggregate model-weight storage
≈
DP * ModelWeightBytes
```

``` text
Ideal aggregate throughput
≈
DP * ThroughputPerReplica
```

``` text
Real aggregate throughput
=
eta_DP * DP * ThroughputPerReplica
```

``` text
PipelineEfficiency
=
M / (M + PP - 1)
```

``` text
BubbleFraction
=
(PP - 1) / (M + PP - 1)
```

``` text
Communication time
≈
latency
+
bytes / effective_bandwidth
```

``` text
global_rank
=
((dp_rank * PP) + pp_rank) * TP + tp_rank
```

## 59. Decision tree

``` text
Does the model fit on one GPU?
|
+-- YES
|   |
|   +-- Does one GPU meet isolated-request latency?
|       |
|       +-- YES -> prefer DP for additional throughput
|       |
|       +-- NO  -> benchmark TP/model optimization
|
+-- NO
    |
    +-- Can intra-node TP make it fit efficiently?
        |
        +-- YES
        |   |
        |   +-- build efficient TP replica
        |   +-- use DP for remaining capacity
        |
        +-- NO
            |
            +-- add PP across stages/nodes
            +-- keep TP on fastest links where practical
            +-- balance stages
            +-- once replica is efficient, scale with DP
```

This is a starting heuristic, not a replacement for profiling.

## 60. Complete mental model

``` text
                           GLOBAL ROUTER
                   /           |                             v            v            v

             DP Replica 0  DP Replica 1  DP Replica 2
                  |
          +-------+-------+
          |               |
          v               v
      PP Stage 0      PP Stage 1
          |               |
      +---+---+       +---+---+
      |   |   |       |   |   |
     TP0 TP1 TP2     TP0 TP1 TP2
      |   |   |       |   |   |
      +-collective     +-collective
          |
          +---- activation ---->
```

Interpret it as:

``` text
TP:
GPUs cooperate inside layers.

PP:
TP groups cooperate across layer partitions.

DP:
complete TP x PP logical models independently serve traffic.
```

## 61. The most important production rule

Do not begin with:

``` text
How can I put all available GPUs into one model replica?
```

Begin with:

``` text
What is the smallest efficient TP x PP topology that:

1. fits the model,
2. leaves realistic KV-cache headroom,
3. meets TTFT and TPOT requirements,
4. maps well to the physical interconnect,
5. and has acceptable failure characteristics?
```

Then ask:

``` text
How many copies of that logical replica should I deploy
with DP to satisfy throughput and availability?
```

This avoids over-sharding simply because GPUs are available.

# Conclusion

3D parallelism is a hierarchy for assigning different scaling problems
to the appropriate dimension.

``` text
TP
= split layer computation

PP
= split the layer stack

DP
= replicate the complete logical TP x PP model
```

For a regular topology:

``` text
Total GPUs = TP * PP * DP
```

but the best factorization depends on model size, HBM, KV-cache
requirements, GPU topology, collective latency, network bandwidth, stage
balance, prefill behavior, decode behavior, traffic, TTFT, TPOT,
availability, and cost.

A strong production heuristic is:

``` text
1. Keep TP on the fastest links whenever practical.

2. Use enough TP and PP to build an efficient logical
   model replica with realistic memory headroom.

3. Do not increase TP or PP merely because GPUs remain.

4. Once the replica meets latency requirements, use
   remaining capacity for DP.

5. Benchmark prefill and decode separately.

6. Optimize SLO-compliant throughput per dollar rather
   than raw GPU utilization.
```

The final communication hierarchy is:

``` text
TP:
frequent collectives

PP:
stage-boundary activation transfers

DP inference:
no required model-forward synchronization between replicas
```

That is the foundation of 3D parallel inference for very large models
across many GPUs and nodes.
