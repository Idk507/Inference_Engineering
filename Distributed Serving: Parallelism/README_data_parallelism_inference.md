# Data Parallelism for LLM Inference

## Full Model Replication and Why Inference Replicas Need No Inter-Replica Model Communication

## 1. Core idea

Data parallelism (DP) replicates the complete logical model and
distributes independent requests across those replicas.

``` text
                    Requests
                       |
                       v
                 Load Balancer
              /        |        \
             v         v         v
        Replica 0  Replica 1  Replica 2
        full model full model full model
```

If model parameters are `theta`, every replica begins with the same
parameters:

``` text
theta_0 = theta_1 = ... = theta_(D-1) = theta
```

A request assigned to replica i is evaluated as:

``` text
y_i = f(x_i; theta_i)
```

For independent requests, no term in that equation depends on another
replica.

Therefore ordinary data-parallel inference requires:

``` text
zero inter-replica communication
for the neural-network forward computation itself
```

This does **not** mean the overall serving system has zero network
traffic. Routing, responses, health checks, metrics, deployment, shared
application state, and optional cache migration can still communicate.

------------------------------------------------------------------------

## 2. One replica

A single complete model computes:

``` text
Request A
   |
   v
[ complete model theta ]
   |
   v
Output A
```

The device or device group owns every layer required to evaluate:

``` text
f(x_A; theta)
```

If request volume exceeds the capacity of this replica, one way to scale
is to create another complete replica.

------------------------------------------------------------------------

## 3. Two-replica inference

``` text
Replica 0:
theta_0 = complete model

Replica 1:
theta_1 = complete model
```

Route different requests:

``` text
Request A -> Replica 0
Request B -> Replica 1
```

Then:

``` text
Replica 0:
y_A = f(x_A; theta_0)

Replica 1:
y_B = f(x_B; theta_1)
```

Nothing in the calculation of `y_A` requires an activation, partial sum,
KV tensor, or logit from Replica 1.

Likewise, Replica 1 does not require Replica 0.

That is the mathematical reason inference DP does not require an
inter-replica collective.

------------------------------------------------------------------------

## 4. The simplest proof

For two independent inputs:

``` text
y_1 = f(x_1; theta)
y_2 = f(x_2; theta)
```

We can evaluate:

``` text
GPU 0:
y_1 = f(x_1; theta)

GPU 1:
y_2 = f(x_2; theta)
```

There is no equation such as:

``` text
y_1 = partial_0 + partial_1
```

and no equation such as:

``` text
hidden_next = Stage1(hidden_previous)
```

across the two replicas.

Each replica computes a complete result.

------------------------------------------------------------------------

## 5. General case

With D replicas:

``` text
Replica i owns complete theta
for i = 0 ... D-1
```

The request stream is partitioned:

``` text
R
=
R_0 union R_1 union ... union R_(D-1)
```

Each replica independently executes:

``` text
for request r in R_i:
    output_r = f(input_r; theta_i)
```

There is no cross-replica reduction in the inference equation.

------------------------------------------------------------------------

## 6. Compare with tensor parallelism

Tensor parallelism (TP) splits computation **inside a model**.

``` text
                  Request A
                      |
              one model execution
                 /          \
                v            v
             GPU 0        GPU 1
             shard        shard
                \            /
                 \          /
                  collective
                      |
                      v
                   Output A
```

For a row-parallel linear layer, ranks may compute:

``` text
Y_0_partial
Y_1_partial
```

where:

``` text
Y = Y_0_partial + Y_1_partial
```

The SUM is mathematically required.

So TP often requires collectives such as:

``` text
all-reduce
all-gather
reduce-scatter
```

Data parallelism instead does:

``` text
Request A -> Replica 0 -> complete Output A
Request B -> Replica 1 -> complete Output B
```

No reconstruction across replicas is required.

------------------------------------------------------------------------

## 7. Compare with pipeline parallelism

Pipeline parallelism (PP) splits the layer stack:

``` text
Request A
   |
   v
Stage 0
layers 0-7
   |
activation transfer
   |
   v
Stage 1
layers 8-15
   |
activation transfer
   |
   v
Stage 2
   |
   v
Output A
```

One request depends on multiple stages.

Data parallelism instead replicates every layer:

``` text
Replica 0:
layers 0 ... L-1

Replica 1:
layers 0 ... L-1
```

A request can remain within its assigned logical replica for the entire
model execution.

------------------------------------------------------------------------

## 8. TP versus PP versus DP

  ------------------------------------------------------------------------
  Parallelism      What is divided? Does one request span Main inference
                                                 workers? communication
  ---------------- ---------------- --------------------- ----------------
  Data parallelism Request stream      No, across logical None required
                                                 replicas between replicas
                                                          for model
                                                          evaluation

  Tensor           Tensors within                     Yes Collectives
  parallelism      layers                                 

  Pipeline         Layer stack                        Yes Stage-boundary
  parallelism                                             activation
                                                          transfers
  ------------------------------------------------------------------------

Memorize:

``` text
DP:
parallelize ACROSS independent requests

TP:
parallelize WITHIN layers

PP:
parallelize ACROSS layer groups
```

------------------------------------------------------------------------

## 9. Full model replication

Suppose model weights require:

``` text
W bytes
```

Ordinary DP uses approximately:

``` text
W bytes of model weights per replica
```

With D replicas:

``` text
aggregate cluster weight memory
≈
D * W
```

The memory is distributed across separate serving resources, but every
replica contains a complete logical model.

This is the price paid for independent execution.

------------------------------------------------------------------------

## 10. Why DP alone does not make a model fit

Suppose:

``` text
model weights = 80 GB
GPU memory = 40 GB
```

Adding four one-GPU DP replicas would conceptually require:

``` text
GPU 0 -> 80 GB model
GPU 1 -> 80 GB model
GPU 2 -> 80 GB model
GPU 3 -> 80 GB model
```

Each still fails to fit.

DP scales the number of complete model instances. It does not inherently
reduce model memory per logical replica.

To solve model fit, use techniques such as:

``` text
tensor parallelism
pipeline parallelism
quantization
other model sharding
```

------------------------------------------------------------------------

## 11. A DP replica can itself contain multiple GPUs

"Full model replication" means a complete **logical model replica**, not
necessarily a single GPU.

Suppose one model requires:

``` text
TP = 4
```

Then:

``` text
DP Replica 0:
GPU 0 + GPU 1 + GPU 2 + GPU 3
        TP group

DP Replica 1:
GPU 4 + GPU 5 + GPU 6 + GPU 7
        TP group
```

Inside each replica:

``` text
TP communication is required
```

Between Replica 0 and Replica 1:

``` text
no communication is required
to evaluate independent requests
```

This distinction is essential in large-model serving.

------------------------------------------------------------------------

## 12. DP + TP

Suppose:

``` text
TP = 2
DP = 4
```

Total GPUs:

``` text
TP * DP = 8
```

Topology:

``` text
                    Router
          /           |           \
         v            v            v

     Replica 0    Replica 1    Replica 2 ...
       /  \         /  \         /  \
     G0   G1      G2   G3      G4   G5
       TP            TP            TP
```

A request assigned to Replica 1 uses its two TP GPUs.

Those GPUs communicate with each other because they jointly evaluate the
model.

They do not need Replica 0 or Replica 2 to complete that request.

------------------------------------------------------------------------

## 13. DP + PP

Suppose:

``` text
PP = 4
DP = 2
```

Then:

``` text
Replica 0:
GPU 0 -> GPU 1 -> GPU 2 -> GPU 3

Replica 1:
GPU 4 -> GPU 5 -> GPU 6 -> GPU 7
```

Within each replica:

``` text
PP activation communication occurs
```

Across the two DP replicas:

``` text
no model-forward synchronization is required
```

------------------------------------------------------------------------

## 14. DP + TP + PP

Suppose:

``` text
TP = 2
PP = 4
DP = 8
```

GPUs per logical model replica:

``` text
TP * PP
=
8
```

Total GPUs:

``` text
TP * PP * DP
=
64
```

Each DP replica is a complete logical model distributed across eight
GPUs.

Within a replica:

``` text
TP collectives
+
PP activation transfers
```

Between DP replicas:

``` text
no neural-network synchronization
for independent inference requests
```

------------------------------------------------------------------------

## 15. Why data-parallel training communicates

Training is different.

Replica 0 processes one data subset:

``` text
B_0
```

Replica 1 processes another:

``` text
B_1
```

They compute different local gradients:

``` text
g_0
g_1
```

If each independently updates:

``` text
theta_0 <- theta_0 - learning_rate * g_0
theta_1 <- theta_1 - learning_rate * g_1
```

then generally:

``` text
theta_0 != theta_1
```

The replicas diverge.

Synchronous data-parallel training therefore combines gradients.

------------------------------------------------------------------------

## 16. Gradient synchronization

With D training replicas:

``` text
g
=
(1/D) * sum_i g_i
```

Then every replica applies the same update:

``` text
theta
<-
theta - learning_rate * g
```

The aggregation is commonly implemented with an all-reduce or related
collective.

Therefore:

``` text
DP training:
gradient synchronization required

DP inference:
no gradient synchronization
```

------------------------------------------------------------------------

## 17. Why inference removes that communication

Ordinary inference has:

``` text
no backward pass
no parameter gradients
no optimizer step
```

The parameters are read-only:

``` text
theta_after_request
=
theta_before_request
```

Therefore replicas do not need to exchange gradients or parameter
updates after serving requests.

------------------------------------------------------------------------

## 18. Training versus inference

  Property                                   Data-parallel training   Data-parallel inference
  ------------------------------------------ ------------------------ -------------------------
  Complete logical model per replica         Yes                      Yes
  Different data per replica                 Yes                      Yes
  Backpropagation                            Yes                      No
  Gradients                                  Yes                      No
  Gradient synchronization                   Yes                      No
  Optimizer update                           Yes                      No
  Mandatory inter-replica model collective   Yes, typically           No

This is the main distinction to remember.

------------------------------------------------------------------------

## 19. KV cache is request-local state

Autoregressive inference creates KV-cache state for each active
sequence.

For request r:

``` text
KV_r
```

contains attention keys and values for that request.

If request r runs on Replica 2:

``` text
KV_r -> Replica 2
```

Other DP replicas do not need `KV_r` to serve their own requests.

So the KV cache does not create a normal inter-replica dependency.

------------------------------------------------------------------------

## 20. Why request affinity matters

After a request begins decoding on one replica, later decode iterations
usually stay there:

``` text
Request A
-> Replica 2

token 1 -> Replica 2
token 2 -> Replica 2
token 3 -> Replica 2
...
```

Replica 2 may hold:

``` text
KV cache
scheduler state
sampling state
prefix-cache references
adapter state
```

Moving the request to another replica may require state transfer or
recomputation.

That is a migration operation, not ordinary DP synchronization.

------------------------------------------------------------------------

## 21. Serving architecture

``` text
Client
  |
  v
API Gateway
  |
  v
Request Router
  |
  +-------------+-------------+
  |             |             |
  v             v             v
Replica 0    Replica 1    Replica 2
  |             |             |
Scheduler     Scheduler     Scheduler
  |             |             |
Full Model    Full Model    Full Model
  |             |             |
KV Cache      KV Cache      KV Cache
```

Each replica can run its own scheduler and continuous batch
independently.

------------------------------------------------------------------------

## 22. Routing is not model communication

A router may inspect:

``` text
queue depth
active sequences
estimated outstanding tokens
KV-cache utilization
health status
GPU load
prefix-cache locality
adapter locality
```

This is serving/control-plane coordination.

It is not a tensor dependency needed to evaluate:

``` text
f(x; theta)
```

Therefore:

``` text
routing coordination
!=
inter-replica neural-network communication
```

------------------------------------------------------------------------

## 23. Does DP mean zero network traffic?

No.

The precise claim is:

``` text
Independent DP replicas require zero inter-replica
communication to mathematically evaluate independent
inference requests.
```

The service may still communicate for:

``` text
requests and responses
load balancing
health checks
metrics
logging
tracing
configuration
model deployment
shared databases
cache coordination
autoscaling
```

These are not mandatory model-forward collectives.

------------------------------------------------------------------------

## 24. Why inference DP scales cleanly

TP and PP place communication on a request's model-execution path.

DP does not require a cross-replica model collective.

If one replica provides sustainable throughput:

``` text
Q
```

then D ideal replicas provide:

``` text
Q_total
≈
D * Q
```

This is why horizontal replication can approach linear throughput
scaling when shared infrastructure and routing remain efficient.

------------------------------------------------------------------------

## 25. Real scaling efficiency

Real throughput is better modeled as:

``` text
Q_D
=
eta(D) * D * Q_1
```

where:

``` text
0 < eta(D) <= 1
```

`eta(D)` captures losses from:

``` text
load imbalance
batch fragmentation
shared CPU bottlenecks
network bottlenecks
heterogeneous request lengths
cache locality
```

These losses do not arise from a mandatory DP inference all-reduce.

------------------------------------------------------------------------

## 26. Scaling-efficiency formula

Define:

``` text
Efficiency_D
=
Q_D / (D * Q_1)
```

Example:

``` text
Q_1 = 1,000 tokens/s
D   = 4
Q_4 = 3,600 tokens/s
```

Then:

``` text
Efficiency_4
=
3600 / 4000
=
0.90
=
90%
```

------------------------------------------------------------------------

## 27. Throughput versus single-request latency

Suppose one request requires:

``` text
2 seconds
```

on one complete replica.

Adding another independent replica does not normally make the same
request take one second.

Instead:

``` text
Replica 0 -> Request A
Replica 1 -> Request B
```

can execute simultaneously.

DP primarily increases:

``` text
aggregate throughput
capacity
availability
```

and can reduce queueing latency under load.

------------------------------------------------------------------------

## 28. Queueing benefit

With one saturated replica:

``` text
requests
   |
   v
 queue
   |
   v
 GPU
```

Adding replicas creates more service capacity:

``` text
                Router
              /   |   \
             v    v    v
            R0   R1   R2
```

This can reduce:

``` text
queue time
TTFT
tail latency
```

even though the neural-network execution time of an isolated request is
unchanged.

------------------------------------------------------------------------

## 29. DP and TTFT

A useful decomposition is:

``` text
TTFT
≈
queue delay
+
prefill
+
first decode step
```

Adding DP replicas can reduce:

``` text
queue delay
```

when traffic is high.

It does not inherently divide one request's prefill across replicas.

------------------------------------------------------------------------

## 30. DP and TPOT

Once a request is assigned, TPOT is primarily governed by that
replica's:

``` text
decode scheduler
batch size
KV-cache behavior
GPU
memory bandwidth
model parallelism
```

DP can indirectly improve TPOT by reducing contention.

But excessive replication can also fragment batches and reduce GPU
efficiency.

------------------------------------------------------------------------

## 31. Batching versus replication

Suppose there are 64 active requests.

Option A:

``` text
1 replica
~64 active requests
```

Option B:

``` text
8 replicas
~8 active requests each
```

More replicas give more independent capacity, but smaller per-replica
batches may reduce:

``` text
weight reuse
GEMM size
arithmetic intensity
GPU utilization
```

So:

``` text
more DP replicas
```

does not automatically imply proportional token-throughput improvement.

------------------------------------------------------------------------

## 32. Continuous batching is replica-local

Each replica can maintain its own in-flight batch.

``` text
Replica 0:
A C D E

Replica 1:
F G H

Replica 2:
I J K L M
```

Their decode iterations do not need to align.

There is no global inference DP step barrier.

This asynchronous independence is very different from synchronous DP
training.

------------------------------------------------------------------------

## 33. Scale up versus scale out

### Scale up

Use more GPUs inside one logical replica:

``` text
TP
PP
```

Useful when:

``` text
the model does not fit
or one logical model execution needs multiple GPUs
```

### Scale out

Create more independent logical replicas:

``` text
DP
```

Useful when:

``` text
one replica is already viable
but more serving capacity is needed
```

A strong serving heuristic is:

``` text
use the smallest efficient TP/PP configuration
needed for one replica,
then scale throughput with DP
```

subject to actual benchmarks.

------------------------------------------------------------------------

## 34. Why DP may beat unnecessary TP

Suppose the entire model fits efficiently on one GPU.

Two GPUs could be configured as:

``` text
TP = 2
DP = 1
```

or:

``` text
TP = 1
DP = 2
```

For throughput-heavy independent requests, DP=2 may be attractive:

``` text
GPU 0 -> Request A
GPU 1 -> Request B
```

without an intra-request collective.

TP=2 makes both GPUs cooperate on every request and introduces
communication.

Which is faster depends on the model, batch, latency target, and
hardware, but this illustrates why unnecessary model parallelism can be
costly.

------------------------------------------------------------------------

## 35. Per-replica memory

Each DP replica independently allocates resources such as:

``` text
model weights
KV-cache pool
runtime workspaces
CUDA graphs
adapter memory
scheduler state
```

So DP increases total cluster memory consumption while also increasing
total capacity.

------------------------------------------------------------------------

## 36. Failure isolation

If:

``` text
Replica 2 fails
```

other replicas can continue serving their own requests.

The router can stop sending new work to Replica 2.

``` text
R0 -> healthy
R1 -> healthy
R2 -> failed
R3 -> healthy
```

Requests whose KV state existed only on the failed replica may fail
unless recovery mechanisms exist, but the other replicas do not depend
on it for their forward computations.

------------------------------------------------------------------------

## 37. Rolling deployments

Independent replicas also make gradual upgrades possible.

``` text
R0 -> v2
R1 -> v1
R2 -> v1
R3 -> v1
```

Drain and replace replicas incrementally until all use v2.

Deployment systems should verify:

``` text
model checkpoint
tokenizer
quantization configuration
adapter version
runtime configuration
```

This is deployment consistency, not per-request model synchronization.

------------------------------------------------------------------------

## 38. LoRA and adapter routing

A base model may be replicated while adapters are loaded selectively.

``` text
Request requiring Adapter A
-> prefer replica where A is resident
```

Adapter-aware routing can improve performance.

The replicas still do not require an all-reduce with each other to
evaluate independent requests.

------------------------------------------------------------------------

## 39. Prefix-cache locality

Suppose Replica 0 has a useful cached prefix.

A cache-aware router may send a matching request to Replica 0.

``` text
request
  |
cache-aware router
  |
Replica 0
```

Sharing or migrating prefix-cache state is optional system optimization.

It is not a fundamental DP requirement.

------------------------------------------------------------------------

## 40. Disaggregated serving nuance

Some systems separate:

``` text
prefill workers
decode workers
KV storage
```

and transfer KV state across machines.

That network traffic exists because the serving architecture
intentionally disaggregates phases.

It should not be confused with communication required by ordinary DP
replication.

------------------------------------------------------------------------

## 41. Shared application state

An AI application may share:

``` text
RAG databases
conversation state
agent memory
rate-limit state
tool state
session metadata
```

across replicas.

Those systems may require networking.

Still:

``` text
shared application state
!=
inter-replica model-forward collective
```

------------------------------------------------------------------------

## 42. Round-robin routing

Simple routing:

``` text
Request 1 -> R0
Request 2 -> R1
Request 3 -> R2
Request 4 -> R0
```

This is easy but can be poor for LLM traffic because requests vary
greatly in:

``` text
prompt length
output length
KV memory
adapter needs
prefix-cache locality
```

------------------------------------------------------------------------

## 43. Least-loaded routing

A better router can estimate:

``` text
load_i
```

and choose:

``` text
argmin_i(load_i)
```

Possible load signals:

``` text
queued tokens
active sequences
scheduled tokens
KV utilization
estimated completion work
```

Token-aware routing is often more meaningful than request-count routing.

------------------------------------------------------------------------

## 44. Why request count can mislead

Compare:

``` text
Replica A:
2 requests with 32k-token prompts

Replica B:
8 requests with 200-token prompts
```

Replica A may have substantially more work despite having fewer
requests.

A useful router should account for work, not merely connection count.

------------------------------------------------------------------------

## 45. Power-of-two choices

For many replicas:

``` text
1. randomly select two replicas
2. inspect their load
3. choose the less-loaded replica
```

This can achieve much better balance than random placement without
querying every replica on every arrival.

------------------------------------------------------------------------

## 46. Autoscaling

DP maps naturally to horizontal autoscaling:

``` text
load increases
-> add replicas

load decreases
-> drain replicas
-> remove them
```

Useful signals include:

``` text
queueing delay
TTFT
pending tokens
active sequences
KV pressure
SLO violation rate
tokens/s
```

GPU utilization alone is not always sufficient.

------------------------------------------------------------------------

## 47. Cold-start cost

A new LLM replica may need to:

``` text
acquire GPU capacity
load weights
allocate KV cache
initialize kernels
compile graphs
warm caches
register with router
pass health checks
```

Large models can have substantial startup latency.

Production systems may therefore use:

``` text
minimum warm replicas
predictive scaling
warm pools
careful draining
```

------------------------------------------------------------------------

## 48. Multi-node advantage

DP replicas can naturally live on different nodes:

``` text
Node A -> Replica 0
Node B -> Replica 1
Node C -> Replica 2
```

Since they do not need per-layer collectives with one another, DP does
not require the same tightly coupled high-bandwidth interconnect that TP
often benefits from.

The network still carries normal serving traffic, but not mandatory
cross-replica tensor synchronization.

------------------------------------------------------------------------

## 49. Offline batch inference

Suppose one million documents must be processed.

With four replicas:

``` text
R0 -> documents 0-249999
R1 -> documents 250000-499999
R2 -> documents 500000-749999
R3 -> documents 750000-999999
```

Every replica independently runs the full model.

Outputs can later be written to shared storage or assembled by the
application.

No gradient synchronization is necessary.

This is close to an embarrassingly parallel workload.

------------------------------------------------------------------------

## 50. Online LLM example

Requests:

``` text
A B C D E F
```

Router:

``` text
A -> R0
B -> R1
C -> R2
D -> R0
E -> R2
F -> R1
```

Local continuous batches:

``` text
R0:
A D

R1:
B F

R2:
C E
```

Each replica has independent:

``` text
KV cache
scheduler
decode iterations
```

No global token-by-token synchronization is required across R0, R1, and
R2.

------------------------------------------------------------------------

## 51. Minimal PyTorch demonstration

``` python
from __future__ import annotations

import copy

import torch
from torch import nn


class TinyModel(nn.Module):
    """Small model for demonstrating inference replication."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


def main() -> None:
    torch.manual_seed(7)

    model = TinyModel()
    model.eval()

    # Two complete model replicas.
    replica_0 = copy.deepcopy(model)
    replica_1 = copy.deepcopy(model)

    request_a = torch.randn(3, 8)
    request_b = torch.randn(5, 8)

    with torch.inference_mode():
        output_a = replica_0(request_a)
        output_b = replica_1(request_b)

        reference_a = model(request_a)
        reference_b = model(request_b)

    torch.testing.assert_close(
        output_a,
        reference_a,
    )

    torch.testing.assert_close(
        output_b,
        reference_b,
    )

    print(
        "Replica 0 independently served A:",
        output_a.shape,
    )

    print(
        "Replica 1 independently served B:",
        output_b.shape,
    )

    print(
        "No cross-replica tensor operation required."
    )


if __name__ == "__main__":
    main()
```

The important property is that each replica has all parameters required
to evaluate its own inputs.

------------------------------------------------------------------------

## 52. Token-aware routing example

``` python
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush


@dataclass(frozen=True)
class Request:
    """Simplified inference request."""

    request_id: int
    estimated_tokens: int


def assign_least_loaded(
    requests: list[Request],
    replica_count: int,
) -> list[list[Request]]:
    """Assign requests using estimated token work."""
    if replica_count <= 0:
        raise ValueError(
            "replica_count must be positive."
        )

    assignments: list[list[Request]] = [
        []
        for _ in range(replica_count)
    ]

    heap = [
        (0, replica_id)
        for replica_id in range(replica_count)
    ]

    for request in requests:
        load, replica_id = heappop(heap)

        assignments[replica_id].append(
            request
        )

        heappush(
            heap,
            (
                load + request.estimated_tokens,
                replica_id,
            ),
        )

    return assignments


def main() -> None:
    requests = [
        Request(0, 500),
        Request(1, 8_000),
        Request(2, 300),
        Request(3, 2_000),
        Request(4, 6_000),
        Request(5, 700),
    ]

    assignments = assign_least_loaded(
        requests,
        replica_count=3,
    )

    for replica_id, assigned in enumerate(
        assignments
    ):
        total = sum(
            request.estimated_tokens
            for request in assigned
        )

        print(
            f"Replica {replica_id}: "
            f"{[r.request_id for r in assigned]} "
            f"estimated_tokens={total}"
        )


if __name__ == "__main__":
    main()
```

The router coordinates placement, while the replicas remain independent
model executors.

------------------------------------------------------------------------

## 53. What to benchmark

Sweep:

``` text
DP = 1, 2, 4, 8, ...
```

Measure:

``` text
aggregate tokens/s
requests/s
TTFT p50/p95/p99
TPOT p50/p95/p99
queueing delay
per-replica batch size
active sequences
GPU utilization
HBM bandwidth
KV-cache utilization
load imbalance
cost per SLO-compliant token
```

The important question is:

``` text
How much useful SLO-compliant serving capacity
does each additional replica add?
```

------------------------------------------------------------------------

## 54. Detecting poor load balance

Suppose:

``` text
R0 -> 96% busy
R1 -> 93% busy
R2 -> 43% busy
R3 -> 37% busy
```

The cluster has unused capacity.

Potential causes:

``` text
request-length variance
sticky sessions
round-robin routing
adapter affinity
prefix-cache affinity
heterogeneous GPUs
KV pressure
```

Adding more replicas without fixing routing may accomplish little.

------------------------------------------------------------------------

## 55. Shared bottlenecks

Even independent replicas can hit common infrastructure limits:

``` text
tokenization CPU
API gateway
network ingress/egress
RAG service
database
model storage
logging
shared filesystem
```

Thus:

``` text
no DP model collective
```

does not imply:

``` text
unlimited linear cluster scaling
```

------------------------------------------------------------------------

## 56. Cost efficiency

If one replica costs:

``` text
C dollars/hour
```

then D replicas cost approximately:

``` text
D*C
```

plus shared infrastructure.

Useful production metrics include:

``` text
SLO-compliant tokens/dollar
SLO-compliant requests/dollar
```

rather than only peak throughput.

Too many replicas can reduce batching efficiency and increase cost per
token.

------------------------------------------------------------------------

## 57. Admission control

Each replica can maintain local safety limits:

``` text
KV-cache high watermark
maximum active sequences
maximum scheduled tokens
memory reserve
```

The router can avoid overloaded replicas.

``` text
R0:
KV = 93%
-> avoid

R1:
KV = 52%
-> eligible

R2:
KV = 61%
-> eligible
```

This is coordination, not model-forward synchronization.

------------------------------------------------------------------------

## 58. Heterogeneous replicas

Suppose:

``` text
R0 -> faster GPU
R1 -> faster GPU
R2 -> slower GPU
```

Equal request counts do not imply equal load.

Routing weights can reflect measured sustainable capacity:

``` text
routing_weight_i
proportional to
capacity_i
```

This is especially useful for token-heavy LLM workloads.

------------------------------------------------------------------------

## 59. When DP is ideal

DP is especially attractive when:

``` text
one complete logical replica fits
traffic contains independent requests
more horizontal throughput is needed
minimal inter-replica coupling is desirable
```

Examples:

``` text
LLM serving
embedding services
reranking
classification
vision inference
offline inference
```

------------------------------------------------------------------------

## 60. When DP alone is insufficient

DP does not directly solve:

``` text
model too large for one logical replica
single-request compute must span devices
single-request latency is too high
```

Use model optimization or model parallelism inside a replica:

``` text
quantization
TP
PP
other sharding
```

Then replicate that optimized logical model with DP if more throughput
is needed.

------------------------------------------------------------------------

## 61. Production design sequence

A useful design process:

``` text
1. Make one logical replica fit.

2. Minimize TP/PP overhead while satisfying memory.

3. Optimize the replica:
   kernels
   batching
   KV cache
   scheduler
   chunked prefill

4. Measure sustainable SLO-compliant capacity.

5. Scale out using DP replicas.

6. Add load-aware routing.

7. Add autoscaling and failure handling.

8. Re-benchmark batching efficiency as DP increases.
```

------------------------------------------------------------------------

## 62. Common mistakes

### Mistake 1: DP means each GPU stores only part of the model

No.

Ordinary DP replicates the complete logical model.

### Mistake 2: DP inference needs gradient all-reduce

No.

Inference has no training-gradient synchronization.

### Mistake 3: zero inter-replica model communication means zero networking

No.

Serving infrastructure still communicates.

### Mistake 4: DP makes one isolated request D times faster

No.

DP mainly increases independent serving capacity.

### Mistake 5: DP solves a model-fit problem

No.

Each logical replica must still fit.

### Mistake 6: decode replicas synchronize every token

No.

Each replica advances its own active sequences.

### Mistake 7: more replicas always increase tokens/s linearly

No.

Batch fragmentation and shared bottlenecks can reduce scaling
efficiency.

### Mistake 8: request count is enough for load balancing

LLM work varies with prompt tokens, generated tokens, KV usage, cache
locality, and adapters.

------------------------------------------------------------------------

## 63. Interview-ready explanation

``` text
Data parallelism replicates the complete logical model and sends
different requests to different replicas.

At inference time, each replica contains every parameter required to
evaluate its assigned request from input to output. Request A on one
replica has no mathematical dependency on Request B on another, so
the replicas do not need to exchange activations, partial outputs, or
model collectives.

This differs from tensor parallelism, where multiple GPUs jointly
compute one layer and may need all-reduces, and pipeline parallelism,
where one request passes through layer groups on different devices.

Data-parallel training does require communication because each
replica computes different gradients. Those gradients must be
synchronized so replicas apply the same parameter update.

Inference has no backward pass or optimizer update, so that gradient
synchronization disappears. The weights are read-only.

The overall serving system can still communicate for routing,
monitoring, shared caches, deployment, and request migration, but
none of those is a required inter-replica collective for the neural
network forward computation.
```

------------------------------------------------------------------------

## 64. Formula sheet

Replication:

``` text
theta_i = theta
for i = 0 ... D-1
```

Independent inference:

``` text
y_i = f(x_i; theta_i)
```

Ideal throughput:

``` text
Throughput_D
≈
D * Throughput_1
```

Real throughput:

``` text
Throughput_D
=
eta(D) * D * Throughput_1
```

Scaling efficiency:

``` text
Efficiency_D
=
Throughput_D
/
(D * Throughput_1)
```

Cluster model-weight memory:

``` text
WeightMemory_cluster
≈
D * WeightMemory_replica
```

Simple multidimensional GPU count:

``` text
Total GPUs
=
DP * TP * PP
```

Training gradient:

``` text
g
=
(1/D) * sum_i g_i
```

Inference has no corresponding gradient aggregation.

------------------------------------------------------------------------

## 65. The three-parallelism mental model

``` text
DATA PARALLELISM

Request A -> [FULL LOGICAL MODEL] -> Output A
Request B -> [FULL LOGICAL MODEL] -> Output B
Request C -> [FULL LOGICAL MODEL] -> Output C

Parallelism:
across independent requests

Required model communication between replicas:
NONE
```

``` text
TENSOR PARALLELISM

              Request A
                  |
        +---------+---------+
        |                   |
    tensor shard        tensor shard
        |                   |
        +---- collective ---+
                  |
               Output A

Parallelism:
inside layer operations

Communication:
collectives
```

``` text
PIPELINE PARALLELISM

Request A
   |
Stage 0
   |
activation
   |
Stage 1
   |
activation
   |
Stage 2
   |
Output A

Parallelism:
across layer groups and multiple in-flight work items

Communication:
stage-boundary activations
```

------------------------------------------------------------------------

## 66. Full production hierarchy

``` text
                            ROUTER
               +-------------+-------------+
               |             |             |
               v             v             v
          DP Replica 0  DP Replica 1  DP Replica 2
               |             |             |
           PP stages      PP stages      PP stages
               |             |             |
          TP groups       TP groups       TP groups
         inside stages   inside stages   inside stages
```

Communication hierarchy:

``` text
Inside a TP group:
collectives

Between PP stages:
activation transfers

Between DP replicas:
no communication required for independent
neural-network inference
```

------------------------------------------------------------------------

## 67. Connection to the inference-engineering sequence

The distributed-inference picture is now:

``` text
Tensor Parallelism
-> split tensor computation
-> intra-layer communication

Pipeline Parallelism
-> split layer stack
-> activation communication
-> pipeline bubbles

Data Parallelism
-> replicate complete logical model
-> split request stream
-> no required inter-replica model communication
```

This naturally leads to the topology-design problem:

``` text
Given:
model size
GPU memory
GPU topology
traffic
TTFT SLO
TPOT SLO
throughput target

How should we choose:
TP
PP
DP
?
```

That is the foundation of distributed LLM serving architecture.

------------------------------------------------------------------------

# Conclusion

Data parallelism scales inference by replicating a complete logical
model:

``` text
Replica 0 -> complete theta
Replica 1 -> complete theta
Replica 2 -> complete theta
```

Independent requests are routed to independent replicas:

``` text
A -> Replica 0
B -> Replica 1
C -> Replica 2
```

Each computes:

``` text
y = f(x; theta)
```

without obtaining intermediate tensors from another DP replica.

That is why ordinary inference DP requires:

``` text
zero inter-replica communication
for the neural-network computation itself
```

Training is fundamentally different.

Different training replicas compute different gradients, so those
gradients must be synchronized:

``` text
g
=
average(g_0, g_1, ..., g_(D-1))
```

Inference removes this path because it has:

``` text
no backward pass
no optimizer update
read-only parameters
```

The cost of DP is replication:

``` text
every replica needs a complete logical model
```

while the benefit is clean horizontal capacity:

``` text
more replicas
-> more independent requests in parallel
-> less queueing under load
-> greater aggregate throughput
-> better failure isolation
```

The precise statement worth memorizing is:

``` text
Data-parallel inference replicas do not need to communicate
with one another to mathematically evaluate independent requests.

The serving system may still coordinate routing, health, caching,
metrics, deployment, and state, but those are not model-forward
collectives.
```
