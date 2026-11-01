# Pipeline Parallelism: Layer Partitioning, Pipeline Bubbles, and Micro-Batching

## 1. What pipeline parallelism solves

Large transformer models have many sequential layers:

``` text
Embedding
   |
Layer 0
   |
Layer 1
   |
Layer 2
   |
...
   |
Layer L-1
   |
LM Head
```

If the whole model cannot fit efficiently on one GPU, one option is
**pipeline parallelism (PP)**.

Instead of splitting an individual matrix operation across GPUs, as
tensor parallelism does, pipeline parallelism splits the **layer
stack**:

``` text
GPU 0:
Layers 0-7

GPU 1:
Layers 8-15

GPU 2:
Layers 16-23

GPU 3:
Layers 24-31
```

A hidden activation moves from one stage to the next:

``` text
input
  |
  v
GPU 0
  |
activation transfer
  |
  v
GPU 1
  |
activation transfer
  |
  v
GPU 2
  |
activation transfer
  |
  v
GPU 3
  |
  v
output
```

The central systems problem is that transformer layers are sequential.

GPU 1 cannot process a particular input until GPU 0 has produced the
activation for that input.

If only one batch is processed at a time, most pipeline stages sit idle.

**Micro-batching** creates multiple independent pieces of work that can
occupy different stages simultaneously.

That converts:

``` text
serial stage execution
```

into:

``` text
overlapped pipeline execution
```

but introduces an unavoidable startup/drain period called the **pipeline
bubble**.

------------------------------------------------------------------------

# 2. Pipeline parallelism versus tensor parallelism

Tensor parallelism splits an operation inside a layer:

``` text
             one layer
                |
       +--------+--------+
       |        |        |
     GPU 0    GPU 1    GPU 2
      shard    shard    shard
```

Pipeline parallelism splits layers:

``` text
GPU 0       GPU 1       GPU 2       GPU 3
layers      layers      layers      layers
0-7         8-15        16-23       24-31
   |           |           |           |
   +---------->+---------->+---------->+
```

A useful distinction is:

``` text
Tensor parallelism:
multiple GPUs cooperate on the SAME layer.

Pipeline parallelism:
different GPUs execute DIFFERENT layers.
```

Large systems commonly combine both.

------------------------------------------------------------------------

# 3. Pipeline stages

Suppose a model has:

``` text
L = 32 transformer layers
P = 4 pipeline stages
```

A simple balanced partition is:

``` text
Stage 0 / GPU 0:
layers 0-7

Stage 1 / GPU 1:
layers 8-15

Stage 2 / GPU 2:
layers 16-23

Stage 3 / GPU 3:
layers 24-31
```

Each stage owns approximately:

``` text
L/P
```

layers.

If all layers had identical compute and memory cost, this would be
balanced.

Real models may also contain:

``` text
embedding layers
LM head
different attention implementations
Mixture-of-Experts layers
different sequence-dependent costs
```

so production partitioning should balance measured stage time rather
than merely layer count.

------------------------------------------------------------------------

# 4. The naive pipeline problem

Consider four stages:

``` text
S0 -> S1 -> S2 -> S3
```

and one batch B1.

Execution is:

``` text
time --->

S0: [ B1 ][    ][    ][    ]

S1: [    ][ B1 ][    ][    ]

S2: [    ][    ][ B1 ][    ]

S3: [    ][    ][    ][ B1 ]
```

Only one GPU is doing useful work at each time slot.

For four GPUs, average utilization is only about:

``` text
1/4 = 25%
```

in this idealized example.

Pipeline parallelism by itself therefore does not create parallel
execution for a single sequential item.

We need multiple items in flight.

------------------------------------------------------------------------

# 5. Micro-batching

Suppose the global batch contains:

``` text
B
```

examples.

Instead of sending the entire batch through all stages as one unit,
divide it into:

``` text
M
```

micro-batches.

If evenly divided:

``` text
micro_batch_size
=
B/M
```

Example:

``` text
global batch = 64
micro-batches = 8

micro-batch size = 8
```

The pipeline now receives:

``` text
MB1
MB2
MB3
...
MB8
```

Each stage can work on a different micro-batch at the same time.

------------------------------------------------------------------------

# 6. Pipeline filling

With four stages and four micro-batches:

``` text
time --->

       t0    t1    t2    t3    t4    t5    t6

S0    MB1   MB2   MB3   MB4
S1          MB1   MB2   MB3   MB4
S2                MB1   MB2   MB3   MB4
S3                      MB1   MB2   MB3   MB4
```

At `t3`:

``` text
S0 -> MB4
S1 -> MB3
S2 -> MB2
S3 -> MB1
```

All four GPUs are busy simultaneously.

This is the steady-state pipeline.

------------------------------------------------------------------------

# 7. Fill, steady state, and drain

Pipeline execution has three conceptual phases.

## Fill

The first micro-batch moves through early stages while later stages are
still idle.

``` text
S0: busy
S1: waiting
S2: waiting
S3: waiting
```

Then:

``` text
S0: busy
S1: busy
S2: waiting
S3: waiting
```

until all stages are occupied.

## Steady state

Every stage has useful work:

``` text
S0: MB4
S1: MB3
S2: MB2
S3: MB1
```

## Drain

The final micro-batches leave early stages, causing them to become idle
before later stages finish.

These fill and drain idle regions form the **pipeline bubble**.

------------------------------------------------------------------------

# 8. Ideal balanced-stage model

Assume:

``` text
P = number of pipeline stages
M = number of micro-batches
t = time for one stage to process one micro-batch
```

Assume every stage takes exactly t.

For a forward-only pipeline, the first micro-batch requires:

``` text
P * t
```

to cross all P stages.

After the pipeline is full, each additional micro-batch completes every:

``` text
t
```

Therefore total time is:

``` text
T_pipeline
=
(P + M - 1) * t
```

or equivalently:

``` text
T_pipeline
=
(M + P - 1) * t
```

This is the fundamental forward-pipeline timing formula.

------------------------------------------------------------------------

# 9. Deriving the formula

Without overlap, M micro-batches through P stages would require:

``` text
M * P
```

stage-time units.

With ideal pipelining, the first result appears after:

``` text
P
```

time units.

The remaining:

``` text
M - 1
```

micro-batches emerge one per time unit.

So:

``` text
total slots
=
P + (M - 1)
=
M + P - 1
```

Hence:

``` text
T_pipeline
=
(M + P - 1)t
```

------------------------------------------------------------------------

# 10. Ideal useful work

Each pipeline stage processes:

``` text
M
```

micro-batches.

So each stage performs:

``` text
M * t
```

of useful computation.

But the wall-clock schedule lasts:

``` text
(M + P - 1)t
```

Therefore ideal stage utilization is:

``` text
U
=
M/(M + P - 1)
```

------------------------------------------------------------------------

# 11. Pipeline bubble fraction

The fraction of schedule capacity not doing useful stage work is:

``` text
Bubble Fraction
=
1 - U
```

Substitute utilization:

``` text
Bubble Fraction
=
1 - M/(M + P - 1)
```

Therefore:

``` text
Bubble Fraction
=
(P - 1)/(M + P - 1)
```

This is the standard forward-pipeline bubble formula.

------------------------------------------------------------------------

# 12. Pipeline efficiency

Pipeline efficiency is:

``` text
Efficiency
=
M/(M + P - 1)
```

Bubble fraction:

``` text
Bubble
=
(P - 1)/(M + P - 1)
```

and naturally:

``` text
Efficiency + Bubble = 1
```

------------------------------------------------------------------------

# 13. Example: P=4, M=4

``` text
P = 4
M = 4
```

Total slots:

``` text
M + P - 1
=
4 + 4 - 1
=
7
```

Efficiency:

``` text
4/7
≈
57.14%
```

Bubble:

``` text
3/7
≈
42.86%
```

So even though four GPUs are available, only about 57% of the ideal
stage capacity is used under this simple schedule.

------------------------------------------------------------------------

# 14. Increase micro-batches

Keep:

``` text
P = 4
```

but increase:

``` text
M = 16
```

Efficiency:

``` text
16/(16+4-1)
=
16/19
≈
84.21%
```

Bubble:

``` text
3/19
≈
15.79%
```

Increasing the number of micro-batches dramatically reduces the relative
pipeline bubble.

------------------------------------------------------------------------

# 15. Why micro-batching reduces the bubble

The bubble overhead is associated primarily with the pipeline depth:

``` text
P - 1
```

Increasing M does not remove the startup/drain cost.

Instead, it amortizes that cost over more useful work.

Conceptually:

``` text
fixed-ish pipeline fill/drain overhead
--------------------------------------
more useful micro-batches
```

Therefore:

``` text
M >> P
```

is desirable for high pipeline utilization.

------------------------------------------------------------------------

# 16. Bubble table

For four stages:

    Micro-batches M   Efficiency   Bubble
  ----------------- ------------ --------
                  1       25.00%   75.00%
                  2       40.00%   60.00%
                  4       57.14%   42.86%
                  8       72.73%   27.27%
                 16       84.21%   15.79%
                 32       91.43%    8.57%
                 64       95.52%    4.48%

The trend is:

``` text
M increases
-> bubble fraction decreases
-> utilization approaches 100%
```

but there are practical limits to making M arbitrarily large.

------------------------------------------------------------------------

# 17. Required M for a target efficiency

Starting from:

``` text
E
=
M/(M + P - 1)
```

solve for M.

``` text
E(M + P - 1)
=
M
```

``` text
EM + E(P-1)
=
M
```

``` text
M(1-E)
=
E(P-1)
```

Therefore:

``` text
M
=
E(P-1)/(1-E)
```

For:

``` text
P = 8
target efficiency = 0.90
```

``` text
M
=
0.90 * 7 / 0.10
=
63
```

So roughly 63 micro-batches are required to reach 90% ideal pipeline
efficiency under the balanced forward-only model.

------------------------------------------------------------------------

# 18. Why deeper pipelines require more micro-batches

For fixed M:

``` text
Bubble
=
(P-1)/(M+P-1)
```

Increasing P increases the bubble.

Example with:

``` text
M = 16
```

For P=2:

``` text
Bubble
=
1/17
≈
5.88%
```

For P=4:

``` text
Bubble
=
3/19
≈
15.79%
```

For P=8:

``` text
Bubble
=
7/23
≈
30.43%
```

For P=16:

``` text
Bubble
=
15/31
≈
48.39%
```

Therefore a deeper pipeline requires more micro-batches to remain
efficient.

------------------------------------------------------------------------

# 19. Why not use infinitely many micro-batches?

Because micro-batching creates trade-offs.

As M increases for a fixed global batch:

``` text
micro_batch_size
=
global_batch_size / M
```

So each micro-batch becomes smaller.

Eventually:

``` text
smaller GEMMs
lower GPU occupancy
lower arithmetic intensity
more kernel-launch overhead
more communication messages
more scheduler overhead
```

can reduce per-stage efficiency.

Thus:

``` text
more micro-batches
-> smaller bubble

but potentially
-> less efficient individual stage execution
```

The optimal M balances these effects.

------------------------------------------------------------------------

# 20. A more realistic stage-time model

The simple formula assumes:

``` text
t0 = t1 = ... = t(P-1) = t
```

Real pipelines have different stage times:

``` text
t0
t1
...
t(P-1)
```

The slowest stage determines steady-state throughput.

Define:

``` text
t_max
=
max(t0,t1,...,t(P-1))
```

A useful forward-only approximation is:

``` text
T
≈
sum_i ti
+
(M-1) * t_max
```

Why?

The first micro-batch must traverse every stage:

``` text
sum_i ti
```

Afterward, completions are limited by the bottleneck stage:

``` text
t_max
```

------------------------------------------------------------------------

# 21. Example of stage imbalance

Suppose:

``` text
S0 = 8 ms
S1 = 8 ms
S2 = 15 ms
S3 = 8 ms
```

The first micro-batch latency is:

``` text
8 + 8 + 15 + 8
=
39 ms
```

Steady-state throughput cannot exceed approximately one micro-batch
every:

``` text
15 ms
```

For M=16:

``` text
T
≈
39 + 15*(16-1)
=
264 ms
```

Stage 2 is the bottleneck.

Other stages periodically wait for it.

This creates **imbalance bubbles** in addition to fill/drain bubbles.

------------------------------------------------------------------------

# 22. The slowest-stage law

Once the pipeline reaches steady state:

``` text
throughput
≈
1/t_max
```

micro-batches per unit time.

This is one of the most important pipeline-parallel principles.

Adding more stages does not help if one stage remains substantially
slower than the others.

Pipeline partitioning is therefore a load-balancing problem.

------------------------------------------------------------------------

# 23. Layer count is not enough for partitioning

Suppose a 32-layer model is divided:

``` text
8 / 8 / 8 / 8
```

layers per GPU.

This does not guarantee equal stage time.

Potential causes include:

``` text
embedding cost
LM-head cost
different layer types
MoE routing
communication boundaries
different memory pressure
sequence-dependent attention cost
hardware heterogeneity
```

A production partitioner should aim for:

``` text
approximately equal measured stage latency
```

not merely equal numbers of layers.

------------------------------------------------------------------------

# 24. Inter-stage communication

Pipeline stages exchange activations.

For:

``` text
Stage i -> Stage i+1
```

the producer sends the boundary hidden state.

If the boundary tensor has:

``` text
micro_batch_size = b
sequence length = S
hidden size = H
element size = s bytes
```

then a rough activation payload is:

``` text
Activation Bytes
≈
b * S * H * s
```

ignoring additional metadata and layout effects.

Pipeline communication therefore depends strongly on micro-batch size
and sequence length.

------------------------------------------------------------------------

# 25. Communication versus tensor parallelism

Tensor parallelism typically performs collective communication inside
layers.

Pipeline parallelism typically performs point-to-point activation
transfer between neighboring stages.

Conceptually:

``` text
TP:
GPU 0 <---- collective ----> GPU 1

PP:
Stage 0 ---- activation ---> Stage 1
```

This difference matters for network topology.

------------------------------------------------------------------------

# 26. Forward-only inference pipeline

For offline inference where many independent requests or batches are
available, pipeline parallelism can keep stages busy:

``` text
time --->

S0: R1 R2 R3 R4 R5 R6
S1:    R1 R2 R3 R4 R5 R6
S2:       R1 R2 R3 R4 R5 R6
S3:          R1 R2 R3 R4 R5 R6
```

After filling:

``` text
one result can complete every bottleneck-stage interval
```

This can improve aggregate throughput.

But the first request still traverses all stages sequentially.

Pipeline parallelism therefore improves throughput much more naturally
than single-request latency.

------------------------------------------------------------------------

# 27. Autoregressive decoding makes PP harder

LLM decode introduces another dependency:

``` text
token t+1
cannot be generated until
token t is known
```

A single request must pass through all pipeline stages to produce one
token before its next token can begin normally.

For one request:

``` text
token 1:
S0 -> S1 -> S2 -> S3

token 2:
S0 -> S1 -> S2 -> S3

...
```

This creates poor utilization unless multiple sequences are interleaved.

Serving systems therefore need sufficient request concurrency or
specialized scheduling to keep pipeline stages occupied during decode.

------------------------------------------------------------------------

# 28. Pipeline parallelism and TTFT

A new request's prefill must traverse the stages.

Ignoring queueing and communication details:

``` text
TTFT contribution
≈
sum_i stage_prefill_time_i
```

The first request does not magically become P times faster because its
layers live on P GPUs.

The stages are sequential for that request.

The throughput benefit comes from processing other
requests/micro-batches concurrently at other stages.

------------------------------------------------------------------------

# 29. Pipeline parallelism and TPOT

During decode, each generated token must traverse the model.

For a single isolated sequence:

``` text
token latency
≈
sum_i decode_stage_time_i
+
communication
```

With many concurrent sequences, different requests can occupy different
stages.

Thus PP can improve aggregate decode throughput while still imposing
stage-to-stage latency on each token.

This is another example of:

``` text
throughput != per-user latency
```

------------------------------------------------------------------------

# 30. Training adds backward propagation

Training is more complex because each micro-batch needs:

``` text
forward pass
+
backward pass
```

The backward pass travels through pipeline stages in the reverse
direction.

Conceptually:

``` text
Forward:
S0 -> S1 -> S2 -> S3

Backward:
S3 -> S2 -> S1 -> S0
```

The scheduling policy determines:

``` text
bubble size
activation memory
communication overlap
optimizer timing
```

------------------------------------------------------------------------

# 31. GPipe schedule

A classic GPipe-style schedule performs:

``` text
all forwards
then
all backwards
```

for the micro-batches.

Conceptually:

``` text
Forward:
MB1
MB2
MB3
...
MBM

then

Backward:
MBM
...
MB3
MB2
MB1
```

This is often called a flush-style schedule because the forward pipeline
is filled and drained before the backward wave completes.

------------------------------------------------------------------------

# 32. GPipe timeline intuition

For a small pipeline:

``` text
time --->

S0: F1 F2 F3 F4 ......... B4 B3 B2 B1
S1:    F1 F2 F3 F4 ... B4 B3 B2 B1
S2:       F1 F2 F3 F4 B4 B3 B2 B1
S3:          F1 F2 F3 F4 B4 B3 B2 B1
```

This diagram is schematic rather than cycle-exact, but it highlights the
two large waves:

``` text
forward wave
backward wave
```

------------------------------------------------------------------------

# 33. GPipe bubble fraction

Under the simplified balanced model where forward and backward stage
slots have equal duration, useful work per stage is:

``` text
2M
```

slots:

``` text
M forward
+
M backward
```

The flush schedule incurs pipeline startup/drain overhead.

Its idealized total slot count is:

``` text
2(M + P - 1)
```

Therefore utilization is:

``` text
U_GPipe
=
2M / [2(M+P-1)]
```

which simplifies to:

``` text
U_GPipe
=
M/(M+P-1)
```

and the bubble fraction is:

``` text
Bubble_GPipe
=
(P-1)/(M+P-1)
```

This familiar expression therefore also appears in the balanced GPipe
analysis.

Be careful: papers and systems sometimes define "bubble" or normalize
forward/backward time differently, so always check the exact schedule
and denominator.

------------------------------------------------------------------------

# 34. Activation-memory problem in GPipe

If all M forward micro-batches are processed before their backwards,
stages may need to retain activations for many outstanding
micro-batches.

Activation memory can therefore grow approximately with the number of
in-flight micro-batches:

``` text
activation memory
proportional to M
```

unless activation checkpointing/recomputation is used.

This is a major motivation for alternative schedules.

------------------------------------------------------------------------

# 35. 1F1B schedule

A widely used training schedule is:

``` text
one forward
one backward
```

or **1F1B** after warmup.

A stage first performs enough forward work to fill dependencies.

Then it alternates:

``` text
Forward micro-batch
Backward micro-batch
Forward micro-batch
Backward micro-batch
...
```

This reduces the number of outstanding activations compared with the
all-forward-then-all-backward approach.

------------------------------------------------------------------------

# 36. 1F1B phases

A 1F1B pipeline has:

``` text
warmup
steady state
cooldown
```

During warmup, a stage performs forward operations until backward work
becomes available.

During steady state:

``` text
1 forward
1 backward
```

operations are interleaved.

During cooldown, remaining backwards are completed.

The exact number of warmup forwards depends on the stage's position in
the pipeline.

------------------------------------------------------------------------

# 37. Why 1F1B helps memory

Suppose a stage has already forwarded many micro-batches but none has
completed backward.

It must retain their activations.

1F1B starts backward work earlier, allowing activations to be released
sooner.

Thus:

``` text
GPipe:
potentially many outstanding forward activations

1F1B:
bounded outstanding activation window
```

This can significantly reduce peak activation memory.

------------------------------------------------------------------------

# 38. Does 1F1B eliminate the pipeline bubble?

No.

It improves scheduling and memory behavior, but pipeline dependencies
still require warmup and cooldown.

The pipeline cannot begin in a fully occupied forward/backward steady
state at time zero.

Nor can all stages remain busy forever after the final micro-batch.

So:

``` text
pipeline bubble != scheduling bug
```

Some bubble is structurally caused by pipeline depth and finite work.

------------------------------------------------------------------------

# 39. Micro-batching and gradient accumulation

Pipeline micro-batching is closely related to gradient accumulation.

Suppose:

``` text
micro-batch size = b
number of micro-batches = M
data-parallel degree = D
```

An approximate global batch is:

``` text
Global Batch
=
b * M * D
```

assuming one optimizer update after those M micro-batches and no
additional accumulation factor.

The gradients from micro-batches can be accumulated before the optimizer
step.

This lets pipeline scheduling use small units of work without changing
the desired global optimization batch.

------------------------------------------------------------------------

# 40. Micro-batch size versus number of micro-batches

These are different variables.

``` text
b = examples/tokens in one micro-batch
M = number of micro-batches per pipeline batch
```

For fixed global batch:

``` text
B = bM
```

Increasing M means decreasing b.

This creates the key optimization trade-off:

``` text
larger M
-> smaller bubble

smaller b
-> potentially worse GPU kernel efficiency
```

You optimize both, not just one.

------------------------------------------------------------------------

# 41. The micro-batch trade-off

Large micro-batch:

``` text
Pros:
larger GEMMs
better GPU utilization
fewer messages
less scheduling overhead

Cons:
fewer micro-batches
larger pipeline bubble
more activation memory per micro-batch
```

Small micro-batch:

``` text
Pros:
more pipeline slots
smaller relative bubble
finer scheduling granularity

Cons:
smaller GEMMs
more kernel overhead
more communication events
possibly lower arithmetic intensity
```

The optimal point is hardware- and workload-dependent.

------------------------------------------------------------------------

# 42. A simple throughput model

Let:

``` text
t_stage(b)
```

be the bottleneck stage time as a function of micro-batch size b.

For a forward-only balanced pipeline:

``` text
T
≈
(M + P - 1) * t_stage(b)
```

with:

``` text
M = B/b
```

So:

``` text
T(b)
≈
(B/b + P - 1) * t_stage(b)
```

Reducing b increases M and shrinks the relative bubble, but `t_stage(b)`
does not necessarily scale linearly.

Very small b can make each stage inefficient.

This is why micro-batch tuning should be measured.

------------------------------------------------------------------------

# 43. Pipeline throughput

Under balanced forward-only steady state:

``` text
throughput
≈
micro_batch_size / t_stage
```

in examples per second, or:

``` text
tokens_per_micro_batch / t_stage
```

for token throughput.

The fill/drain bubble lowers finite-batch average throughput.

For long-running streams with abundant work, the steady-state bottleneck
stage dominates.

------------------------------------------------------------------------

# 44. Pipeline latency versus throughput

Consider four balanced stages with:

``` text
t = 10 ms
```

A single micro-batch latency is:

``` text
4 * 10
=
40 ms
```

Once the pipeline is full, one micro-batch can complete every:

``` text
10 ms
```

So:

``` text
latency per individual micro-batch
≈ 40 ms

steady-state completion interval
≈ 10 ms
```

This distinction is fundamental.

Pipeline parallelism overlaps different work items; it does not remove
the sequential stage path for one work item.

------------------------------------------------------------------------

# 45. Bubble as lost capacity

For:

``` text
P stages
M micro-batches
```

total stage-slot capacity is:

``` text
P * (M + P - 1)
```

Useful stage slots are:

``` text
P * M
```

Idle stage slots are:

``` text
P(P-1)
```

The fraction idle is:

``` text
P(P-1)
----------------
P(M+P-1)
```

which simplifies to:

``` text
(P-1)/(M+P-1)
```

This gives another derivation of the forward-only bubble formula.

------------------------------------------------------------------------

# 46. Visual example: P=4, M=4

``` text
       t0   t1   t2   t3   t4   t5   t6

S0     M1   M2   M3   M4   --   --   --
S1     --   M1   M2   M3   M4   --   --
S2     --   --   M1   M2   M3   M4   --
S3     --   --   --   M1   M2   M3   M4
```

Useful slots:

``` text
4 stages * 4 micro-batches
=
16
```

Total slots:

``` text
4 stages * 7 time slots
=
28
```

Utilization:

``` text
16/28
=
4/7
≈
57.14%
```

Bubble:

``` text
12/28
=
3/7
≈
42.86%
```

------------------------------------------------------------------------

# 47. Visual example: P=4, M=8

``` text
       0  1  2  3  4  5  6  7  8  9 10

S0     1  2  3  4  5  6  7  8
S1        1  2  3  4  5  6  7  8
S2           1  2  3  4  5  6  7  8
S3              1  2  3  4  5  6  7  8
```

Total time slots:

``` text
8 + 4 - 1
=
11
```

Efficiency:

``` text
8/11
≈
72.73%
```

The fixed fill/drain region is now a smaller fraction of the schedule.

------------------------------------------------------------------------

# 48. Virtual pipeline stages and interleaving

Some systems divide each physical pipeline rank into multiple **virtual
stages** or model chunks.

Instead of:

``` text
GPU 0 -> one contiguous layer block
GPU 1 -> one contiguous layer block
```

a rank can own multiple chunks and execute an interleaved schedule.

The goal is to create finer scheduling granularity and reduce exposed
bubble time.

Conceptually:

``` text
physical GPU 0:
chunk A
chunk E

physical GPU 1:
chunk B
chunk F

...
```

Interleaving can improve utilization, but it increases scheduling and
communication complexity.

------------------------------------------------------------------------

# 49. Activation checkpointing

Pipeline training can consume substantial activation memory.

Activation checkpointing trades compute for memory:

``` text
forward:
store selected checkpoints
discard some intermediate activations

backward:
recompute discarded forward activations
then compute gradients
```

This can allow:

``` text
larger micro-batches
more micro-batches
larger models
```

at the cost of additional computation.

Pipeline scheduling and activation checkpointing are therefore often
tuned together.

------------------------------------------------------------------------

# 50. Communication overlap

A stage may be able to overlap:

``` text
sending activation for MB1
```

with:

``` text
computing MB2
```

when dependencies and hardware permit.

Likewise during training:

``` text
activation send/receive
gradient send/receive
compute
```

may partially overlap.

A simplified stage time becomes:

``` text
T_stage
≈
compute
+
non-overlapped communication
```

rather than always:

``` text
compute + all communication
```

------------------------------------------------------------------------

# 51. Pipeline boundary placement

Not every layer boundary is equally good.

A boundary should consider:

``` text
stage compute balance
parameter memory
activation size
communication cost
special layers
embedding/head placement
hardware topology
```

If a boundary produces an unusually large activation transfer,
communication may offset the balancing benefit.

------------------------------------------------------------------------

# 52. Memory distribution

Pipeline parallelism reduces parameter memory per GPU because each stage
owns only a subset of layers.

Idealized:

``` text
parameter memory per stage
≈
total parameter memory / P
```

if layers are balanced.

But stage memory also includes:

``` text
activations
optimizer state
gradients
temporary buffers
communication buffers
KV cache during inference
```

so total memory is not simply total model bytes divided by P.

------------------------------------------------------------------------

# 53. PP and KV cache

During autoregressive inference, each pipeline stage owns the KV cache
corresponding to the attention layers assigned to that stage.

So KV memory is naturally partitioned by layer:

``` text
Stage 0:
KV for early layers

Stage 1:
KV for middle layers

...

Stage P-1:
KV for late layers
```

This differs from tensor parallelism, where the KV representation may
also be sharded within a layer depending on the attention/TP layout.

------------------------------------------------------------------------

# 54. Pipeline parallelism with tensor parallelism

Large systems often create a grid.

Example:

``` text
PP = 4
TP = 2
```

Total GPUs per model replica:

``` text
4 * 2
=
8 GPUs
```

Conceptually:

``` text
Pipeline Stage 0:
GPU 0 + GPU 1
(tensor-parallel group)

Pipeline Stage 1:
GPU 2 + GPU 3

Pipeline Stage 2:
GPU 4 + GPU 5

Pipeline Stage 3:
GPU 6 + GPU 7
```

Within a stage:

``` text
TP shards layers
```

Between stages:

``` text
PP transfers activations
```

This is a common way to scale very large transformers.

------------------------------------------------------------------------

# 55. Add data parallelism

Suppose:

``` text
TP = 2
PP = 4
DP = 8
```

Total accelerator count:

``` text
2 * 4 * 8
=
64
```

Each data-parallel replica contains:

``` text
TP * PP
=
8 GPUs
```

The full system has eight such model replicas.

This gives the classic multidimensional-parallelism picture:

``` text
Tensor Parallel
x
Pipeline Parallel
x
Data Parallel
```

------------------------------------------------------------------------

# 56. Pipeline bubble versus data-parallel efficiency

Pipeline bubble is specific to pipeline-stage dependencies.

Data parallelism has different inefficiencies, such as:

``` text
gradient synchronization
load imbalance
stragglers
optimizer communication
```

Tensor parallelism has:

``` text
intra-layer collectives
```

Pipeline parallelism has:

``` text
fill/drain bubbles
stage imbalance
activation transfers
```

When parallelism dimensions are combined, all of these costs interact.

------------------------------------------------------------------------

# 57. Choosing pipeline depth

Increasing P can:

``` text
reduce parameters per GPU
reduce compute per stage
allow a larger model to fit
```

but also:

``` text
increase bubble overhead
increase stage boundaries
increase activation communication
require more micro-batches
increase scheduling complexity
```

Therefore the largest possible P is rarely automatically optimal.

------------------------------------------------------------------------

# 58. A practical selection heuristic

Start from memory.

Ask:

``` text
How many layers can fit comfortably per GPU/stage?
```

This gives a minimum PP degree.

Then measure stage times.

Adjust layer assignment so:

``` text
t0 ≈ t1 ≈ ... ≈ t(P-1)
```

Then choose M large enough that:

``` text
M >> P
```

while keeping micro-batch kernels efficient.

Finally benchmark:

``` text
throughput
latency
bubble fraction
stage idle time
communication
memory
```

------------------------------------------------------------------------

# 59. Pipeline bubble versus imbalance bubble

It is useful to separate two causes of idle time.

## Structural bubble

Created by finite pipeline fill and drain:

``` text
approximately P-1 slots
```

in the balanced forward-only model.

## Imbalance bubble

Created when one stage is slower:

``` text
t_i < t_max
```

Fast stages repeatedly wait for the bottleneck.

Increasing M helps amortize structural fill/drain overhead.

It does **not** fix a severely imbalanced stage partition.

------------------------------------------------------------------------

# 60. Pipeline bubble versus communication stalls

A third source of idle time is communication.

A stage may finish compute but wait for:

``` text
activation receive
activation send completion
gradient receive
network synchronization
```

This is not the ideal mathematical pipeline bubble.

When profiling, distinguish:

``` text
fill/drain idle
stage imbalance idle
communication idle
runtime/scheduler idle
```

because each needs a different optimization.

------------------------------------------------------------------------

# 61. Python bubble calculator

``` python
from __future__ import annotations


def pipeline_efficiency(
    stages: int,
    micro_batches: int,
) -> float:
    """Return ideal balanced forward-pipeline utilization."""
    if stages <= 0:
        raise ValueError("stages must be positive.")

    if micro_batches <= 0:
        raise ValueError(
            "micro_batches must be positive."
        )

    return (
        micro_batches
        /
        (micro_batches + stages - 1)
    )


def bubble_fraction(
    stages: int,
    micro_batches: int,
) -> float:
    """Return ideal fill/drain bubble fraction."""
    return 1.0 - pipeline_efficiency(
        stages,
        micro_batches,
    )


def total_forward_time(
    stages: int,
    micro_batches: int,
    stage_time_ms: float,
) -> float:
    """Return ideal balanced forward pipeline time."""
    if stage_time_ms <= 0:
        raise ValueError(
            "stage_time_ms must be positive."
        )

    return (
        micro_batches + stages - 1
    ) * stage_time_ms


def main() -> None:
    stages = 4
    stage_time_ms = 10.0

    print(
        "M | Efficiency | Bubble | Time (ms)"
    )
    print("-" * 44)

    for micro_batches in (
        1,
        2,
        4,
        8,
        16,
        32,
    ):
        efficiency = pipeline_efficiency(
            stages,
            micro_batches,
        )

        bubble = bubble_fraction(
            stages,
            micro_batches,
        )

        runtime = total_forward_time(
            stages,
            micro_batches,
            stage_time_ms,
        )

        print(
            f"{micro_batches:>2} | "
            f"{efficiency:>9.2%} | "
            f"{bubble:>6.2%} | "
            f"{runtime:>9.1f}"
        )


if __name__ == "__main__":
    main()
```

Expected trend:

``` text
M increases
-> efficiency increases
-> bubble decreases
```

------------------------------------------------------------------------

# 62. Uneven-stage calculator

``` python
from __future__ import annotations

from collections.abc import Sequence


def estimate_pipeline_time(
    stage_times_ms: Sequence[float],
    micro_batches: int,
) -> float:
    """
    Estimate forward-only pipeline time for fixed stage times.

    T ≈ sum(stage times)
        + (M-1) * max(stage time)
    """
    if not stage_times_ms:
        raise ValueError(
            "At least one stage is required."
        )

    if micro_batches <= 0:
        raise ValueError(
            "micro_batches must be positive."
        )

    if any(
        value <= 0
        for value in stage_times_ms
    ):
        raise ValueError(
            "All stage times must be positive."
        )

    first_item_latency = sum(
        stage_times_ms
    )

    bottleneck_time = max(
        stage_times_ms
    )

    return (
        first_item_latency
        +
        (micro_batches - 1)
        * bottleneck_time
    )


def main() -> None:
    balanced = [
        10.0,
        10.0,
        10.0,
        10.0,
    ]

    imbalanced = [
        8.0,
        8.0,
        15.0,
        8.0,
    ]

    micro_batches = 16

    print(
        "Balanced:",
        estimate_pipeline_time(
            balanced,
            micro_batches,
        ),
        "ms",
    )

    print(
        "Imbalanced:",
        estimate_pipeline_time(
            imbalanced,
            micro_batches,
        ),
        "ms",
    )


if __name__ == "__main__":
    main()
```

This demonstrates why the slowest stage controls steady-state
throughput.

------------------------------------------------------------------------

# 63. Simple pipeline simulator

The following simulation assigns each micro-batch to each stage as soon
as both the stage and the incoming activation are ready.

``` python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    """One forward pipeline operation."""

    stage: int
    micro_batch: int
    start_ms: float
    end_ms: float


def simulate_forward_pipeline(
    stage_times_ms: list[float],
    micro_batches: int,
) -> list[Operation]:
    """Simulate a forward-only pipeline."""
    if not stage_times_ms:
        raise ValueError(
            "stage_times_ms cannot be empty."
        )

    if micro_batches <= 0:
        raise ValueError(
            "micro_batches must be positive."
        )

    if any(t <= 0 for t in stage_times_ms):
        raise ValueError(
            "Stage times must be positive."
        )

    stages = len(stage_times_ms)

    stage_available = [
        0.0
        for _ in range(stages)
    ]

    completion = [
        [0.0 for _ in range(micro_batches)]
        for _ in range(stages)
    ]

    operations: list[Operation] = []

    for micro_batch in range(
        micro_batches
    ):
        for stage in range(stages):
            dependency_ready = (
                0.0
                if stage == 0
                else completion[
                    stage - 1
                ][micro_batch]
            )

            start = max(
                stage_available[stage],
                dependency_ready,
            )

            end = (
                start
                + stage_times_ms[stage]
            )

            completion[stage][
                micro_batch
            ] = end

            stage_available[stage] = end

            operations.append(
                Operation(
                    stage=stage,
                    micro_batch=micro_batch,
                    start_ms=start,
                    end_ms=end,
                )
            )

    return operations


def main() -> None:
    operations = simulate_forward_pipeline(
        stage_times_ms=[
            8.0,
            8.0,
            15.0,
            8.0,
        ],
        micro_batches=4,
    )

    for op in operations:
        print(
            f"stage={op.stage} "
            f"mb={op.micro_batch} "
            f"start={op.start_ms:.1f} "
            f"end={op.end_ms:.1f}"
        )


if __name__ == "__main__":
    main()
```

This can be extended to visualize stage idle intervals and experiment
with different partitions.

------------------------------------------------------------------------

# 64. What to profile

For every stage, measure:

``` text
forward compute time
backward compute time
activation send time
activation receive time
gradient send/receive time
idle time
memory usage
activation memory
GPU utilization
```

At the pipeline level:

``` text
micro-batches per second
tokens per second
pipeline bubble
stage imbalance
end-to-end step time
TTFT
TPOT
```

The most useful visualization is often a timeline showing each GPU's
compute and communication regions.

------------------------------------------------------------------------

# 65. Reading a pipeline trace

A healthy balanced forward pipeline looks roughly like:

``` text
GPU 0: C C C C C C C C
GPU 1:   C C C C C C C C
GPU 2:     C C C C C C C C
GPU 3:       C C C C C C C C
```

An imbalanced pipeline may look like:

``` text
GPU 0: C C - C - C - C -
GPU 1:   C - C - C - C -
GPU 2:     LONG LONG LONG LONG
GPU 3:          C    C    C
```

The gaps after filling indicate the bottleneck stage is starving or
backpressuring other stages.

------------------------------------------------------------------------

# 66. Optimization decision tree

If bubble is high:

``` text
Is M small relative to P?
    |
    +-- yes -> increase micro-batch count if feasible
    |
    +-- no --> inspect stage imbalance and communication
```

If one stage is slower:

``` text
rebalance layers
move expensive modules
split virtual stages
optimize kernels
```

If communication dominates:

``` text
reduce boundary traffic
improve overlap
use faster topology
reconsider PP placement
```

If micro-batches are too small:

``` text
increase micro-batch size
or reduce pipeline depth
```

The optimization is multi-dimensional.

------------------------------------------------------------------------

# 67. Common mistakes

## Mistake 1: saying PP makes one request P times faster

It usually does not.

One request still traverses stages sequentially.

PP primarily creates throughput parallelism across multiple work items.

## Mistake 2: confusing global batch with micro-batch

A global batch can be divided into many micro-batches for pipeline
scheduling.

## Mistake 3: thinking more micro-batches are always better

More micro-batches shrink the relative bubble but may make each GPU
kernel too small.

## Mistake 4: balancing by layer count only

Balance measured stage latency and memory.

## Mistake 5: using the bubble formula without stating assumptions

``` text
(P-1)/(M+P-1)
```

assumes a simplified balanced pipeline schedule.

Real training schedules, unequal forward/backward times, interleaving,
communication, and stage imbalance modify observed utilization.

## Mistake 6: treating all idle time as pipeline bubble

Distinguish:

``` text
fill/drain bubble
stage imbalance
communication stalls
scheduler/runtime stalls
```

## Mistake 7: mixing GPipe and 1F1B formulas

Always identify the schedule before deriving utilization.

## Mistake 8: ignoring activation memory

Micro-batching and scheduling directly affect how many activations must
remain live.

------------------------------------------------------------------------

# 68. Interview-ready derivation

Suppose there are:

``` text
P pipeline stages
M micro-batches
```

and each stage takes one equal time slot per micro-batch.

The first micro-batch takes:

``` text
P
```

slots to traverse the pipeline.

After that, each additional micro-batch finishes one slot later.

Therefore:

``` text
total slots
=
P + M - 1
```

Each stage performs useful work for:

``` text
M
```

slots.

Thus:

``` text
pipeline efficiency
=
M/(M+P-1)
```

and:

``` text
bubble fraction
=
(P-1)/(M+P-1)
```

Increasing M amortizes the fixed fill/drain cost, but if the global
batch is fixed it also reduces micro-batch size, which can hurt GPU
efficiency.

------------------------------------------------------------------------

# 69. Formula sheet

Layer partition:

``` text
L layers
P stages

ideal layers/stage
≈
L/P
```

Micro-batch size:

``` text
b
=
B/M
```

Balanced forward pipeline time:

``` text
T
=
(M+P-1)t
```

Efficiency:

``` text
U
=
M/(M+P-1)
```

Bubble fraction:

``` text
Bubble
=
(P-1)/(M+P-1)
```

Micro-batches for target efficiency E:

``` text
M
=
E(P-1)/(1-E)
```

Uneven-stage forward approximation:

``` text
T
≈
sum_i ti
+
(M-1)max_i(ti)
```

Steady-state completion rate:

``` text
micro-batches/second
≈
1/t_max
```

Global training batch, simplified:

``` text
B_global
=
micro_batch_size
*
micro_batches
*
data_parallel_degree
```

------------------------------------------------------------------------

# 70. Complete mental model

``` text
                    MODEL
                      |
              split layer stack
                      |
      +---------------+---------------+
      |               |               |
      v               v               v
   Stage 0          Stage 1         Stage 2 ...
      |               |               |
      +---- activation+---- activation+
                      |
                      v
              sequential dependency

Single work item:
poor cross-stage utilization

                      |
                      v
                MICRO-BATCH
                      |
        MB1 MB2 MB3 MB4 ... MBM
                      |
                      v
          multiple items in flight
                      |
                      v
               PIPELINE OVERLAP
                      |
          +-----------+-----------+
          |                       |
          v                       v
   higher throughput         fill/drain bubble
                                  |
                                  v
                  Bubble=(P-1)/(M+P-1)

Increase M:
bubble fraction decreases

But:
micro-batch size may decrease
        |
        v
smaller GEMMs / more overhead

And:
slowest stage controls throughput
        |
        v
partition layers by measured time,
not merely layer count
```

------------------------------------------------------------------------

# 71. Connection to tensor parallelism

The previous tensor-parallel guide established:

``` text
TP:
split matrices inside a layer
-> intra-layer collective communication
```

Pipeline parallelism instead does:

``` text
PP:
split layers across devices
-> inter-stage activation communication
```

Together:

``` text
                    MODEL
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Pipeline Stage 0        Pipeline Stage 1
          |                       |
    TP GPU 0 + 1             TP GPU 2 + 3
          |                       |
  sharded layer compute   sharded layer compute
          |                       |
          +---- activation ------>+
```

This distinction is essential for understanding large-model distributed
inference and training.

------------------------------------------------------------------------

# 72. Production inference lesson

For LLM serving, pipeline parallelism should not be judged only by:

``` text
Can the model fit?
```

Measure:

``` text
prefill throughput
decode throughput
TTFT
TPOT
stage utilization
stage imbalance
activation transfer
request concurrency
KV-cache distribution
```

A pipeline with excellent aggregate throughput can still have poor
per-user token latency.

Likewise, a deeper pipeline can reduce per-device memory while
increasing the concurrency required to keep all stages occupied.

The production objective is therefore not:

``` text
maximize PP degree
```

but:

``` text
choose the smallest/most efficient combination
of PP, TP, and other parallel dimensions
that satisfies memory, throughput, and latency SLOs.
```

------------------------------------------------------------------------

# Conclusion

Pipeline parallelism partitions a model by **layers**:

``` text
Stage 0 -> early layers
Stage 1 -> middle layers
Stage 2 -> later layers
...
```

A single input still moves sequentially through those stages, so PP does
not inherently accelerate the latency of one item.

The parallelism comes from **micro-batching**:

``` text
MB1 on Stage 3
MB2 on Stage 2
MB3 on Stage 1
MB4 on Stage 0
```

allowing multiple stages to perform useful work simultaneously.

For a balanced forward-only pipeline with:

``` text
P stages
M micro-batches
stage time t
```

the total time is:

``` text
T
=
(M+P-1)t
```

giving:

``` text
Efficiency
=
M/(M+P-1)
```

and:

``` text
Bubble
=
(P-1)/(M+P-1)
```

The bubble shrinks as M increases because pipeline fill/drain overhead
is amortized across more useful micro-batches.

But M cannot be increased blindly. Smaller micro-batches can reduce GEMM
efficiency and increase communication/scheduling overhead.

Real performance is also governed by the slowest stage:

``` text
T
≈
sum_i ti
+
(M-1)max_i(ti)
```

so stage balancing is just as important as bubble reduction.

For training, schedules such as GPipe and 1F1B determine how forward and
backward micro-batches are interleaved, trading pipeline utilization
against activation memory.

The systems-level mental model is therefore:

``` text
Pipeline parallelism
=
layer partitioning
+
micro-batch scheduling
+
stage balancing
+
activation communication
+
bubble management
```

and the key engineering goal is not simply to divide layers evenly, but
to keep every stage doing useful work for as much of the schedule as
possible.
