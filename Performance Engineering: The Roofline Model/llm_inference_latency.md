# LLM Inference Latency Metrics: TTFT, TPOT, Throughput, and Per-User Latency

## Introduction

Large language model inference is not adequately described by a single number such as "latency" or "tokens per second." An LLM request has multiple phases, and each phase affects the user differently.

A request normally enters a queue, runs through **prefill**, produces its first output token, and then enters an autoregressive **decode** loop that emits additional tokens one at a time. Because of this structure, production LLM serving systems usually need several metrics rather than one.

The most important are:

```text
TTFT        = Time To First Token
TPOT        = Time Per Output Token
ITL         = Inter-Token Latency
E2E latency = End-to-End request latency
Throughput  = completed work per unit time
```

Understanding these metrics also explains one of the central trade-offs in inference engineering:

```text
Increasing batching
        |
        +--> usually improves GPU utilization
        +--> usually improves system throughput
        |
        +--> may increase queueing
        +--> may increase TTFT
        +--> may increase TPOT / inter-token latency
        |
        +--> therefore may worsen per-user latency
```

This chapter derives these relationships formally and connects them to the Roofline model, arithmetic intensity, continuous batching, KV-cache behavior, and production service-level objectives.

---

# 1. The lifecycle of an LLM request

Consider a request containing:

```text
Input tokens  = P
Output tokens = O
```

A simplified request lifecycle is:

```text
Client
  |
  | request arrives
  v
+--------------------+
| Request queue      |
+--------------------+
  |
  | scheduler admits request
  v
+--------------------+
| Prefill            |
| process P tokens   |
+--------------------+
  |
  | first token becomes available
  v
==============================
          TTFT
==============================
  |
  v
+--------------------+
| Decode iteration 1 |
+--------------------+
  |
  | token
  v
+--------------------+
| Decode iteration 2 |
+--------------------+
  |
  | token
  v
        ...
  |
+--------------------+
| Decode iteration O |
+--------------------+
  |
  v
Request complete
```

This naturally creates two latency regimes.

Prefill determines how long the user waits before generation begins.

Decode determines how quickly the answer appears once generation has started.

That is why TTFT and TPOT should not be collapsed into a single latency number.

---

# 2. Time To First Token

**TTFT**, or Time To First Token, measures the elapsed time from the arrival of a request until the first generated token becomes available to the client.

Conceptually:

```text
TTFT =
    queueing time
  + scheduling/admission overhead
  + input preprocessing
  + prefill execution
  + first decode step
  + relevant serving/network overhead
```

Depending on the measurement boundary used by a serving system, tokenization or network transfer may be included or excluded. Therefore every benchmark should document where its timer starts and stops.

A simplified server-side approximation is:

```text
TTFT ≈ T_queue + T_prefill + T_first_decode
```

Suppose:

```text
Queueing             = 25 ms
Prefill              = 130 ms
First decode step    = 15 ms
```

Then:

```text
TTFT ≈ 25 + 130 + 15
     ≈ 170 ms
```

The user waits approximately 170 milliseconds before seeing the first generated token.

---

# 3. Why prompt length affects TTFT

During prefill, the model processes the prompt tokens.

For a transformer projection, the operation resembles:

```text
X @ W
```

where:

```text
X = [P, hidden_size]
W = [hidden_size, output_size]
```

As prompt length `P` increases, the amount of prefill computation increases.

For a simplified linear projection:

```text
Prefill FLOPs ≈ 2 * P * K * N
```

A longer prompt therefore generally increases prefill work.

Attention introduces additional sequence-length-dependent work. For standard dense self-attention, portions of prefill attention scale approximately quadratically with prompt length because prompt tokens attend across the prompt.

Therefore:

```text
larger prompt
     |
     +--> more projection work
     +--> more attention work
     +--> larger KV cache construction
     |
     v
higher prefill latency
     |
     v
potentially higher TTFT
```

The relationship is not perfectly linear because kernels, batching, hardware utilization, attention implementation, prefix caching, and chunked prefill all affect the result.

---

# 4. TTFT is not just prefill time

A common mistake is:

```text
TTFT = prefill latency
```

This is incomplete.

Imagine that the GPU could execute the request's prefill in 100 ms, but the request waits 400 ms in the scheduler queue.

Then:

```text
Prefill execution = 100 ms
Queueing           = 400 ms

TTFT ≈ 500 ms
```

Optimizing the prefill kernel by 20% gives:

```text
100 ms -> 80 ms
```

but TTFT only changes from roughly:

```text
500 ms -> 480 ms
```

That is just a 4% TTFT improvement.

This is why production inference optimization must measure queueing separately from GPU execution.

---

# 5. Time Per Output Token

After the first token is produced, the model enters autoregressive decoding.

For output tokens:

```text
y1, y2, y3, ..., yO
```

each new token depends on the previously generated context.

Conceptually:

```text
token 1
   |
   v
token 2
   |
   v
token 3
   |
   v
...
```

These tokens cannot normally all be generated simultaneously because token `i + 1` depends on token `i`.

**TPOT**, or Time Per Output Token, measures the average amount of time required for subsequent output-token generation.

A common approximation is:

```text
TPOT =
    time spent generating tokens after the first
    ------------------------------------------------
             number of tokens after the first
```

If:

```text
Generation after first token = 900 ms
Remaining output tokens      = 45
```

then:

```text
TPOT = 900 / 45
     = 20 ms/token
```

---

# 6. TPOT and tokens per second

TPOT can be converted approximately into per-request generation speed:

```text
tokens_per_second ≈ 1000 / TPOT_ms
```

For example:

```text
TPOT = 100 ms/token
Generation rate ≈ 10 tokens/s

TPOT = 50 ms/token
Generation rate ≈ 20 tokens/s

TPOT = 20 ms/token
Generation rate ≈ 50 tokens/s

TPOT = 10 ms/token
Generation rate ≈ 100 tokens/s
```

But this is a **per-request generation rate**.

It must not be confused with server-wide throughput.

A server might generate:

```text
50 tokens/s for each active request
```

while simultaneously serving 100 requests.

Its aggregate token throughput can therefore be far higher than 50 tokens/s.

---

# 7. Inter-Token Latency

TPOT is an average.

Users experience individual gaps between streamed tokens.

Those gaps are described by **Inter-Token Latency**, often abbreviated ITL.

Suppose tokens arrive at:

```text
Token 1: 200 ms
Token 2: 218 ms
Token 3: 237 ms
Token 4: 257 ms
Token 5: 300 ms
```

Then the gaps are:

```text
Token 1 -> 2 = 18 ms
Token 2 -> 3 = 19 ms
Token 3 -> 4 = 20 ms
Token 4 -> 5 = 43 ms
```

The average may still look reasonable, but the 43 ms stall can be noticeable.

Therefore:

```text
TPOT = average decode-generation behavior
ITL  = individual spacing between streamed tokens
```

For interactive systems, the distribution of ITL can matter as much as average TPOT.

---

# 8. End-to-end latency

End-to-end latency measures the complete request:

```text
request sent
    ->
first token
    ->
remaining output tokens
    ->
final token
```

A useful approximation is:

```text
E2E latency ≈ TTFT + (O - 1) * TPOT
```

where:

```text
O = number of output tokens
```

Suppose:

```text
TTFT = 300 ms
TPOT = 25 ms/token
O    = 101 tokens
```

Then:

```text
E2E ≈ 300 + (101 - 1) * 25

E2E ≈ 300 + 2500

E2E ≈ 2800 ms
```

So the complete response takes approximately:

```text
2.8 seconds
```

The formula is an approximation because real inter-token latency is not necessarily constant.

---

# 9. Why TTFT and TPOT affect users differently

Consider two systems.

System A:

```text
TTFT = 200 ms
TPOT = 40 ms
```

System B:

```text
TTFT = 700 ms
TPOT = 15 ms
```

For a short response of 10 tokens:

```text
System A:
E2E ≈ 200 + 9 * 40
    ≈ 560 ms

System B:
E2E ≈ 700 + 9 * 15
    ≈ 835 ms
```

System A finishes sooner.

Now consider a 200-token response:

```text
System A:
E2E ≈ 200 + 199 * 40
    ≈ 8160 ms

System B:
E2E ≈ 700 + 199 * 15
    ≈ 3685 ms
```

System B becomes much faster overall.

This illustrates why there is no universally sufficient "latency" metric.

Short interactive requests are highly sensitive to TTFT.

Long generations are strongly influenced by TPOT.

---

# 10. Throughput

Latency describes how long an individual request waits.

Throughput describes how much work the entire serving system completes per unit time.

Common throughput metrics include:

```text
requests / second
output tokens / second
total tokens / second
```

For LLM serving, output-token throughput is often especially useful:

```text
output throughput =
    total generated output tokens
    -----------------------------
            elapsed time
```

Suppose a server generates:

```text
20,000 output tokens
```

over:

```text
10 seconds
```

Then:

```text
throughput = 20,000 / 10
           = 2,000 output tokens/s
```

This does not mean every individual user receives 2,000 tokens per second.

The 2,000 tokens may be distributed across many concurrent requests.

---

# 11. Throughput versus per-user latency

This is one of the central trade-offs in inference serving.

Suppose requests are executed individually.

```text
Request A -> GPU
Request B waits
Request C waits
Request D waits
```

The GPU may have poor utilization because many operations, particularly decode operations, have low arithmetic intensity.

Now batch them:

```text
A \
B  \
C   --> GPU batch
D  /
```

The GPU processes more requests together.

This can improve:

```text
weight reuse
arithmetic intensity
Tensor Core utilization
kernel efficiency
memory-bandwidth utilization
overall tokens/s
```

Therefore:

```text
batch size increases
       |
       v
GPU utilization increases
       |
       v
system throughput increases
```

But there is another side.

To form a larger batch, requests may have to wait.

```text
Request A arrives
       |
       | wait for more work
       |
Request B arrives
       |
Request C arrives
       |
       v
batch executes
```

This adds queueing latency.

Therefore:

```text
larger batches
    |
    +--> potentially higher throughput
    |
    +--> potentially higher queueing delay
    |
    +--> potentially worse TTFT
```

This is the throughput-latency trade-off.

---

# 12. Why batching improves throughput formally

Consider a linear layer:

```text
Y = XW
```

with:

```text
X = [B, K]
W = [K, N]
Y = [B, N]
```

Approximate FLOPs:

```text
FLOPs ≈ 2 * B * K * N
```

Approximate traffic:

```text
Bytes ≈ s * (K*N + B*K + B*N)
```

where:

```text
s = bytes per element
```

Arithmetic intensity is therefore approximately:

```text
AI(B) =
    2 * B * K * N
    -----------------------------
    s * (K*N + B*K + B*N)
```

When the weight matrix dominates traffic:

```text
K*N >> B*K + B*N
```

then:

```text
AI(B) ≈ 2 * B / s
```

For FP16/BF16:

```text
s = 2 bytes
```

therefore:

```text
AI(B) ≈ B FLOPs/byte
```

Increasing batch size allows one weight load to support computations for multiple tokens.

This moves the workload toward the right side of the Roofline model:

```text
Performance
^
|
|                         ---------------- Peak compute
|                       /
|                     /
|                   /
|                 /
|               /
|             /
|           /
|_________/________________________________> Arithmetic intensity
        ^
        ridge point
```

Before the ridge point:

```text
Performance ≈ AI * memory bandwidth
```

Increasing batching can therefore substantially increase achieved compute throughput.

Eventually the workload reaches compute limits or another bottleneck and additional batching yields diminishing returns.

---

# 13. Why throughput and latency are not opposites

It is tempting to assume:

```text
higher throughput = higher latency
```

This is not always true.

At very small batch sizes, the GPU may be underutilized.

Increasing concurrency from:

```text
batch = 1
```

to:

```text
batch = 8
```

may improve GPU efficiency so substantially that both throughput and some latency measures improve.

A conceptual curve can look like:

```text
Low load:

GPU underutilized
Throughput low
Latency low/moderate

Moderate load:

GPU efficiently utilized
Throughput high
Latency still acceptable

Extreme load:

GPU saturated
Queues grow
Throughput approaches capacity
Latency rises dramatically
```

The dangerous region is near and beyond saturation.

---

# 14. Queueing changes everything

Let:

```text
lambda = incoming request rate
mu     = maximum sustainable service rate
```

If:

```text
lambda << mu
```

the system has spare capacity.

Queueing is usually limited.

If:

```text
lambda approaches mu
```

small bursts can create significant queues.

If:

```text
lambda > mu
```

requests arrive faster than the service can sustainably complete them.

The queue grows unless the system performs actions such as:

```text
autoscaling
load shedding
admission control
backpressure
request rejection
traffic routing
```

This means GPU throughput benchmarks performed with an always-full offline batch do not tell you the latency experienced by an online user.

---

# 15. The saturation knee

Imagine increasing offered traffic.

```text
Requests/s
    |
    v

Low load
    |
    v
GPU has spare capacity
    |
    v
small queue
    |
    v
low TTFT
```

As load rises:

```text
more concurrent requests
        |
        v
better batching
        |
        v
higher throughput
```

Eventually:

```text
GPU utilization -> near maximum
```

Beyond this point, additional requests mostly create queues.

Conceptually:

```text
Latency
^
|                                  /
|                               __/
|                            __/
|                         __/
|                    ____/
|_______________ ___/________________> Offered load
                ^
          saturation knee
```

Operating permanently beyond this knee produces poor tail latency.

Production capacity planning therefore should not target only maximum benchmark throughput. It should target sustainable throughput under latency SLOs.

---

# 16. Static batching

Suppose the server waits until four requests arrive:

```text
A arrives at 0 ms
B arrives at 10 ms
C arrives at 40 ms
D arrives at 90 ms
```

If execution begins only when `D` arrives:

```text
A waited 90 ms
B waited 80 ms
C waited 50 ms
D waited 0 ms
```

Batching improved execution efficiency, but request A paid 90 ms before computation even began.

This directly increases TTFT.

---

# 17. Continuous batching

Modern LLM servers commonly use continuous or in-flight batching.

Instead of:

```text
create batch
run every request until completion
destroy batch
create next batch
```

the scheduler operates approximately like:

```text
iteration 1:
[A B C D]

iteration 2:
[A B C D]

C completes

iteration 3:
[A B E D]

A completes

iteration 4:
[F B E D]
```

New requests can enter as slots or token budgets become available.

This helps maintain high GPU utilization without requiring every request to begin and finish together.

Continuous batching therefore attempts to balance:

```text
high concurrency
       +
high throughput
       +
bounded scheduling latency
```

It does not eliminate the throughput-latency trade-off, but it provides much finer control over it.

---

# 18. Prefill and decode compete for the GPU

LLM serving has heterogeneous work.

Prefill tends to involve large matrix operations and often has relatively high arithmetic intensity.

Decode frequently operates with small token dimensions while repeatedly reading model weights and KV-cache data.

Therefore:

```text
Prefill:
compute-heavy / high AI

Decode:
often memory-bandwidth-heavy / lower AI
```

Suppose a long prompt enters while many users are decoding.

If the scheduler executes the entire long prefill immediately:

```text
long prefill
------------------------------------------------>

decode requests wait
```

Existing users may experience an inter-token latency spike.

Their streaming output may appear to pause.

---

# 19. Chunked prefill and latency

Chunked prefill divides a long prompt into smaller pieces.

Instead of:

```text
[PREFILL PREFILL PREFILL PREFILL PREFILL]
[decode waits...........................]
```

the scheduler can interleave:

```text
[PREFILL CHUNK]
[DECODE]
[DECODE]
[PREFILL CHUNK]
[DECODE]
[PREFILL CHUNK]
```

This can reduce head-of-line blocking for active decode requests.

The trade-off is scheduling complexity and potentially different prefill efficiency.

Chunk size therefore becomes an important serving parameter:

```text
large chunks
    -> better prefill efficiency in some regimes
    -> potentially worse decode interruption

small chunks
    -> better decode responsiveness
    -> more scheduling/kernel overhead
```

The optimal value depends on hardware, model, traffic distribution, latency objectives, and serving implementation.

---

# 20. TTFT versus TPOT scheduling conflict

Consider two classes of users.

User A:

```text
new request
large prompt
waiting for first token
```

User B:

```text
already generating
waiting for next token
```

Prioritizing A's prefill may improve:

```text
TTFT_A
```

while hurting:

```text
TPOT_B
ITL_B
```

Prioritizing decode may improve B's streaming experience while delaying A's first token.

The scheduler is therefore solving a multi-objective optimization problem:

```text
minimize TTFT
minimize TPOT
minimize ITL variance
maximize throughput
respect memory capacity
maintain fairness
avoid starvation
```

There is no single scheduling policy that simultaneously minimizes every metric under all workloads.

---

# 21. Batch size versus TTFT

Suppose a scheduler has a batching window.

```text
batch_window = 0 ms
```

Requests are dispatched immediately.

Potential result:

```text
low queueing
small batches
lower GPU efficiency
```

Now:

```text
batch_window = 10 ms
```

The scheduler briefly waits for additional requests.

Potential result:

```text
larger batches
better GPU efficiency
higher throughput
+ up to roughly 10 ms additional batching delay
```

Increasing the window further may continue improving batching but eventually becomes harmful for interactive TTFT.

Production inference therefore often uses dynamic scheduling rather than a large fixed batching delay.

---

# 22. Batch size versus TPOT

During decode, each active sequence generally contributes a small amount of new-token work per scheduling iteration.

Increasing the number of concurrently decoded sequences changes the operation from something resembling many small matrix-vector products toward more efficient batched matrix operations.

This increases weight reuse.

Conceptually:

```text
B = 1

weight load -> token A
```

versus:

```text
B = 32

                 -> token A
                 -> token B
weight load ----> token C
                 -> ...
                 -> token AF
```

This can dramatically improve aggregate tokens/s.

However, a larger decode batch also means an iteration contains more work.

If decode iteration duration grows from:

```text
10 ms
```

to:

```text
25 ms
```

then each sequence may receive its next token less frequently even though aggregate GPU throughput is higher.

For example:

```text
Batch 1:
1 token / 10 ms
aggregate = 100 tokens/s
per-user = 100 tokens/s
```

An illustrative batched case:

```text
Batch 32:
32 tokens / 25 ms
aggregate = 1280 tokens/s
per-user ≈ 40 tokens/s
```

The numbers are illustrative, not universal.

The important relationship is:

```text
aggregate throughput:
100 -> 1280 tokens/s

per-user generation rate:
100 -> 40 tokens/s
```

The server became dramatically more efficient overall while each individual request generated tokens more slowly.

That is the throughput versus per-user latency trade-off in its clearest form.

---

# 23. Concurrency is not batch size

These terms are related but should not be treated as identical.

**Concurrency** describes how many requests are active.

**Batch size** describes how many work items are processed together by a particular execution step.

For example:

```text
100 active requests
```

does not necessarily imply:

```text
batch size = 100
```

The scheduler might process:

```text
32 requests this iteration
32 requests next iteration
36 requests later
```

because of token budgets, KV-cache capacity, priorities, prompt lengths, prefill work, and scheduling policies.

---

# 24. Goodput

Maximum throughput alone is not always useful.

Suppose:

```text
Configuration A:
5000 tokens/s
P99 TTFT = 8 seconds

Configuration B:
4200 tokens/s
P99 TTFT = 900 ms
```

If your service requires:

```text
P99 TTFT < 1 second
```

configuration A is not acceptable despite its higher raw throughput.

This motivates **goodput**.

A useful conceptual definition is:

```text
Goodput =
    work successfully completed
    while satisfying required SLOs
    --------------------------------
               time
```

For an interactive LLM service, a request might count as successful only if:

```text
TTFT <= TTFT_SLO
AND
TPOT <= TPOT_SLO
```

The exact definition depends on the service.

This shifts optimization from:

```text
maximize tokens/s
```

to:

```text
maximize tokens/s that satisfy user-facing latency objectives
```

That is a much better production objective.

---

# 25. Average latency is dangerous

Suppose TTFT measurements are:

```text
100 ms
110 ms
105 ms
120 ms
115 ms
108 ms
112 ms
5000 ms
```

The mean is heavily affected by the 5-second outlier, but even a reasonable-looking mean can hide the actual shape of a large latency distribution.

Production systems therefore track percentiles.

Common metrics include:

```text
P50
P90
P95
P99
P99.9
```

If:

```text
P50 TTFT = 250 ms
P95 TTFT = 700 ms
P99 TTFT = 2500 ms
```

then half of users receive a first token within 250 ms, while the slowest 1% represented by the P99 boundary wait around 2.5 seconds or more depending on the distribution.

Tail latency often exposes:

```text
queueing
large prompts
memory pressure
batch contention
scheduler unfairness
cache misses
GC/runtime pauses
network variance
multi-tenant interference
```

---

# 26. Measure TTFT and TPOT as distributions

Do not report only:

```text
average TTFT = 300 ms
average TPOT = 20 ms
```

Prefer reporting something like:

```text
TTFT:
P50 = ...
P95 = ...
P99 = ...

TPOT:
P50 = ...
P95 = ...
P99 = ...

ITL:
P50 = ...
P95 = ...
P99 = ...

E2E:
P50 = ...
P95 = ...
P99 = ...
```

Then segment them by:

```text
prompt-length bucket
output-length bucket
model
hardware
tenant
priority class
cache-hit/cache-miss
concurrency
batch size
```

Otherwise different workload classes become mixed into misleading aggregates.

---

# 27. A worked serving example

Suppose an LLM service receives a request with:

```text
Prompt = 2000 tokens
Output = 100 tokens
```

Measured timings:

```text
Queueing          = 50 ms
Prefill           = 180 ms
First decode      = 20 ms
Average TPOT      = 25 ms
```

Approximate TTFT:

```text
TTFT = 50 + 180 + 20
     = 250 ms
```

Approximate remaining generation:

```text
99 * 25 ms
= 2475 ms
```

Approximate E2E:

```text
E2E = 250 + 2475
    = 2725 ms
```

or:

```text
2.725 seconds
```

Now increase concurrency.

Suppose batching improves GPU efficiency, but queueing and decode-step duration rise:

```text
Queueing          = 180 ms
Prefill           = 150 ms
First decode      = 25 ms
TPOT              = 35 ms
```

TTFT becomes:

```text
TTFT = 180 + 150 + 25
     = 355 ms
```

E2E becomes:

```text
355 + 99 * 35
= 355 + 3465
= 3820 ms
```

The GPU might now serve many more users simultaneously and achieve substantially higher aggregate throughput, even though this individual user's latency worsened.

This is exactly why throughput measurements must be interpreted together with user-facing latency distributions.

---

# 28. A simple latency decomposition

A useful conceptual decomposition is:

```text
Request latency
    |
    +-- Queueing
    |
    +-- Scheduling
    |
    +-- Tokenization / preprocessing
    |
    +-- Prefill
    |
    +-- Decode
    |
    +-- Sampling
    |
    +-- Serialization / streaming
    |
    +-- Network
```

For optimization, determine which component dominates before changing kernels or scheduler parameters.

For example:

```text
Problem:
P99 TTFT is high

Possible causes:
    queueing
    long prompts
    prefill contention
    cold prefix cache
    admission control
    overloaded replicas
```

versus:

```text
Problem:
TPOT is high

Possible causes:
    decode batch too large
    memory bandwidth saturation
    KV-cache traffic
    poor kernel efficiency
    prefill interference
    communication overhead
```

The correct optimization depends on the actual bottleneck.

---

# 29. Latency and the Roofline model

From the Roofline model:

```text
Achievable performance =
min(
    Peak compute,
    Arithmetic intensity * Peak memory bandwidth
)
```

Decode frequently has low arithmetic intensity.

Increasing decode batching increases weight reuse and can increase arithmetic intensity.

Therefore:

```text
larger decode batch
       |
       v
higher arithmetic intensity
       |
       v
higher hardware utilization
       |
       v
higher aggregate throughput
```

But user-facing latency additionally depends on:

```text
queueing
scheduler policy
iteration duration
KV-cache pressure
prefill interference
```

The Roofline model explains hardware efficiency.

It does not, by itself, model queueing or scheduling latency.

This distinction is essential.

---

# 30. KV cache and latency

During autoregressive decoding, attention reuses previously generated keys and values from the KV cache.

As context length grows:

```text
KV cache read volume per decode step
            increases
```

For long contexts, this can increase memory traffic.

Therefore TPOT may vary with context length.

A benchmark that reports one TPOT value without describing context length can be misleading.

Production measurements should consider buckets such as:

```text
0-1K context
1K-4K
4K-16K
16K-64K
64K+
```

with boundaries appropriate for the model and workload.

---

# 31. Prefix caching and TTFT

Suppose many requests share:

```text
system prompt
tool definitions
common document prefix
few-shot examples
```

Without caching:

```text
shared prefix -> recomputed
```

With a suitable prefix/KV caching mechanism:

```text
shared prefix -> reused
```

This can reduce repeated prefill work and therefore improve TTFT for cache-hit requests.

However, measurements should distinguish:

```text
cold TTFT
warm TTFT
prefix-cache hit rate
```

Otherwise benchmark results may not represent real traffic.

---

# 32. Quantization and latency

Weight quantization reduces the number of bytes needed to represent model parameters.

For memory-bound decode, reducing weight traffic can improve performance substantially.

Conceptually:

```text
FP16 weights:
2 bytes/parameter

INT8:
~1 byte/parameter

INT4:
~0.5 byte/parameter
```

Ignoring metadata and implementation overhead, lower precision can reduce required memory bandwidth.

Therefore:

```text
less weight traffic
      |
      v
lower memory-transfer time
      |
      v
potentially lower TPOT
      +
higher decode throughput
```

But real performance also depends on:

```text
dequantization overhead
kernel support
Tensor Core support
quantization granularity
hardware architecture
batch size
KV-cache precision
```

Quantization should therefore be benchmarked end-to-end rather than inferred solely from model size.

---

# 33. Speculative decoding and TPOT

Speculative decoding attempts to generate multiple candidate tokens using a smaller or cheaper draft model and then verifies them using the target model.

If multiple proposed tokens are accepted, the target model can make progress on several output tokens with fewer expensive sequential target-model steps.

The objective is effectively:

```text
more accepted output tokens
per expensive target-model operation
```

This can improve generation throughput and user-visible decode speed.

However, benefit depends on acceptance rate, draft-model cost, verification efficiency, batching behavior, and workload characteristics.

---

# 34. Disaggregated prefill and decode

Prefill and decode have different hardware characteristics.

```text
Prefill:
often compute-oriented

Decode:
often memory-bandwidth-oriented
```

A disaggregated architecture can route them to separate worker pools:

```text
Request
   |
   v
+------------------+
| Prefill workers  |
+------------------+
   |
   | KV state transfer
   v
+------------------+
| Decode workers   |
+------------------+
   |
   v
Streaming response
```

Potential benefits include independently scaling resources for:

```text
TTFT demand
decode throughput
hardware specialization
```

But the architecture introduces:

```text
KV-transfer latency
network bandwidth requirements
coordination complexity
failure handling
additional observability requirements
```

A design is useful only if the benefit outweighs those costs.

---

# 35. Production SLOs

A production service should define explicit service-level objectives.

An illustrative structure is:

```text
TTFT:
P50 <= target_A
P95 <= target_B
P99 <= target_C

TPOT:
P50 <= target_D
P95 <= target_E
P99 <= target_F

Availability:
>= target_G

Goodput:
>= target_H requests/s satisfying latency SLOs
```

The actual targets must be derived from product requirements rather than copied from another service.

A coding assistant, voice assistant, offline batch summarizer, and document-generation system can tolerate very different latency profiles.

---

# 36. Interactive versus offline workloads

For interactive chat:

```text
Priority:
TTFT
TPOT
tail latency
smooth streaming
```

For offline generation:

```text
Priority:
throughput
GPU utilization
cost/token
```

An offline workload can tolerate:

```text
large batches
long queueing
large prefill chunks
```

if these maximize total throughput.

An interactive service often cannot.

Therefore there is no globally optimal batch size.

The correct configuration depends on the workload objective.

---

# 37. Observability architecture

A production inference stack should expose metrics at multiple layers.

Application layer:

```text
request rate
request failures
input tokens
output tokens
E2E latency
```

Serving layer:

```text
TTFT
TPOT
ITL
queueing latency
prefill latency
decode latency
batch size
active sequences
waiting sequences
KV-cache utilization
prefix-cache hit rate
```

GPU layer:

```text
GPU utilization
HBM utilization
memory bandwidth
Tensor Core utilization
kernel duration
GPU memory usage
```

Infrastructure layer:

```text
replica count
CPU utilization
network latency
network throughput
node health
autoscaling state
```

These layers should be correlated using request identifiers and distributed traces where practical.

---

# 38. Example request trace

A trace might look like:

```text
request_id = abc123

0 ms      request received
3 ms      tokenization complete
5 ms      request queued
45 ms     scheduler admits request
50 ms     prefill starts
170 ms    prefill completes
185 ms    first token emitted
205 ms    second token emitted
225 ms    third token emitted
...
2185 ms   final token emitted
```

From this trace:

```text
Queue wait ≈ 40 ms

TTFT ≈ 185 ms

Early ITL:
205 - 185 = 20 ms
225 - 205 = 20 ms
```

This is much more actionable than:

```text
request latency = 2185 ms
```

because it identifies where the time was spent.

---

# 39. Load testing correctly

A useful inference benchmark should vary at least:

```text
input length
output length
arrival rate
concurrency
model
precision
hardware
```

Then collect:

```text
TTFT P50/P95/P99
TPOT P50/P95/P99
ITL P50/P95/P99
E2E P50/P95/P99
requests/s
output tokens/s
goodput
GPU utilization
memory utilization
KV-cache utilization
```

Testing only one batch size with an always-full GPU is a throughput microbenchmark, not a realistic online-serving benchmark.

---

# 40. Open-loop versus closed-loop load generation

This distinction is especially important for latency tests.

A closed-loop client often behaves like:

```text
send request
wait for completion
send next request
```

When the server slows down, the client automatically sends fewer requests.

This can hide overload.

An open-loop load generator attempts to preserve an externally specified arrival process:

```text
request
   request
      request
request
        request
```

even when the server becomes slower.

This better reveals:

```text
queue buildup
saturation
tail latency
overload behavior
```

when modeling externally arriving traffic.

---

# 41. Autoscaling

Suppose:

```text
incoming load increases
```

and one GPU replica approaches its latency limit.

Autoscaling can add replicas:

```text
Load balancer
     |
     +--> GPU worker 1
     +--> GPU worker 2
     +--> GPU worker 3
     +--> GPU worker 4
```

Useful scaling signals can include combinations of:

```text
queue depth
waiting tokens
TTFT
request rate
KV-cache pressure
GPU utilization
```

Scaling only on GPU utilization can be insufficient because the user-visible symptom may first appear as queueing or KV-cache pressure.

Autoscaling also has startup delay, so capacity headroom is important for latency-sensitive services.

---

# 42. Admission control

When demand exceeds safe capacity, accepting every request indefinitely can produce:

```text
huge queues
extreme TTFT
timeouts
memory pressure
cascading failures
```

Admission control can enforce limits on:

```text
active sequences
queued requests
total scheduled tokens
KV-cache allocation
prompt length
generation length
tenant quotas
```

Under severe overload, rejecting or deferring some requests can preserve service quality for admitted requests.

This is often preferable to allowing every request to time out after a long wait.

---

# 43. Security and multi-tenancy

Latency optimization must not bypass tenant isolation.

Production serving should consider:

```text
authentication
authorization
tenant quotas
encrypted transport
secret management
request-log redaction
prompt/output data governance
cache isolation
rate limiting
```

Shared prefix caches and KV caches require particular care because data from one tenant must not become accessible to another tenant through incorrect cache keys, memory reuse, logging, or debugging interfaces.

Performance optimizations should preserve the same security boundaries as the rest of the serving architecture.

---

# 44. Cost efficiency

Maximum throughput is often related to lower cost per token because expensive GPU resources are being used more efficiently.

A simplified relationship is:

```text
cost_per_output_token =
    GPU cost per second
    -------------------
    output tokens per second
```

But optimizing only this ratio can damage latency.

A more useful production objective can be framed as:

```text
minimize cost per SLO-compliant token
```

rather than:

```text
minimize cost per raw token
```

This incorporates both infrastructure efficiency and user experience.

---

# 45. A small Python latency-analysis experiment

The following script demonstrates how TTFT, TPOT, E2E latency, throughput, and percentile statistics can be computed from request measurements.

```python
"""Analyze latency metrics for an illustrative LLM inference workload."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class RequestMeasurement:
    """Represent timing measurements for one completed LLM request."""

    request_id: str
    ttft_ms: float
    completion_ms: float
    output_tokens: int

    def tpot_ms(self) -> float:
        """Return approximate time per output token after the first token."""
        if self.output_tokens <= 1:
            return 0.0

        decode_time_ms = self.completion_ms - self.ttft_ms

        if decode_time_ms < 0:
            raise ValueError(
                f"Completion time cannot be smaller than TTFT for "
                f"{self.request_id}."
            )

        return decode_time_ms / (self.output_tokens - 1)


def percentile(values: Sequence[float], q: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0.0 and 1.0.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = q * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index

    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def summarize(requests: Sequence[RequestMeasurement]) -> None:
    """Print latency percentiles for completed request measurements."""
    if not requests:
        raise ValueError("At least one request measurement is required.")

    ttfts = [request.ttft_ms for request in requests]
    tpots = [
        request.tpot_ms()
        for request in requests
        if request.output_tokens > 1
    ]
    e2e = [request.completion_ms for request in requests]

    print(f"Requests: {len(requests)}")
    print(f"Average TTFT: {mean(ttfts):.2f} ms")

    for label, q in (("P50", 0.50), ("P95", 0.95), ("P99", 0.99)):
        print(f"{label} TTFT: {percentile(ttfts, q):.2f} ms")

    if tpots:
        print(f"Average TPOT: {mean(tpots):.2f} ms/token")

        for label, q in (("P50", 0.50), ("P95", 0.95), ("P99", 0.99)):
            print(f"{label} TPOT: {percentile(tpots, q):.2f} ms/token")

    for label, q in (("P50", 0.50), ("P95", 0.95), ("P99", 0.99)):
        print(f"{label} E2E: {percentile(e2e, q):.2f} ms")


def main() -> None:
    """Run an illustrative latency-analysis example."""
    requests = [
        RequestMeasurement("r1", 180.0, 2100.0, 100),
        RequestMeasurement("r2", 210.0, 2250.0, 105),
        RequestMeasurement("r3", 195.0, 2050.0, 95),
        RequestMeasurement("r4", 260.0, 2400.0, 110),
        RequestMeasurement("r5", 900.0, 3200.0, 100),
    ]

    try:
        summarize(requests)
    except (ValueError, TypeError) as exc:
        print(f"Unable to analyze latency measurements: {exc}")


if __name__ == "__main__":
    main()
```

The point of the experiment is not the illustrative numbers. It is the measurement pattern:

```text
never optimize from one average
```

Instead, analyze distributions and correlate them with load and scheduler state.

---

# 46. A useful production dashboard

A serving dashboard can organize metrics into four areas.

```text
USER EXPERIENCE

TTFT P50 / P95 / P99
TPOT P50 / P95 / P99
ITL P50 / P95 / P99
E2E P50 / P95 / P99


TRAFFIC

requests/s
input tokens/s
output tokens/s
active requests
queued requests


SCHEDULER

decode batch size
prefill tokens/iteration
waiting tokens
running sequences
preemptions
prefix-cache hit rate


GPU

compute utilization
HBM bandwidth utilization
GPU memory utilization
KV-cache utilization
kernel latency
```

Correlating these metrics allows questions such as:

```text
Why did P99 TTFT increase?

queue depth increased
    ->
GPU reached sustainable capacity
    ->
waiting requests increased
    ->
TTFT increased
```

or:

```text
Why did TPOT increase?

long prefills arrived
    ->
decode iterations were delayed
    ->
ITL increased
```

---

# 47. Optimization decision tree

If TTFT is poor:

```text
Is queueing high?
    |
    +-- yes -> capacity, admission control, scheduling, autoscaling
    |
    +-- no
         |
         Is prefill slow?
             |
             +-- yes -> kernels, prefix caching, chunked prefill,
             |          quantization/hardware/model choices
             |
             +-- no -> inspect tokenization, first decode,
                        networking, serialization
```

If TPOT is poor:

```text
Is decode GPU execution slow?
    |
    +-- yes
    |    |
    |    +--> memory bandwidth?
    |    +--> weight traffic?
    |    +--> KV-cache traffic?
    |    +--> kernel efficiency?
    |    +--> communication?
    |
    +-- no
         |
         Is scheduling delaying decode?
             |
             +--> prefill interference
             +--> oversized batches
             +--> fairness policy
             +--> preemption
```

This prevents optimizing the wrong subsystem.

---

# 48. Common mistakes

## Mistake 1: Reporting only tokens per second

```text
"We achieve 10,000 tokens/s."
```

This tells you little about interactive user experience.

Always pair throughput with latency distributions.

## Mistake 2: Reporting only average latency

Averages can hide tail behavior.

Track P50, P95, and P99 at minimum for important production metrics.

## Mistake 3: Calling throughput "tokens per second" without qualification

Specify whether you mean:

```text
per-request generation tokens/s
aggregate output tokens/s
total input + output tokens/s
```

These are different metrics.

## Mistake 4: Treating TTFT as prefill time

Queueing can dominate TTFT.

Measure it separately.

## Mistake 5: Treating TPOT as constant

TPOT can change with:

```text
context length
batch size
concurrency
KV-cache pressure
prefill interference
hardware utilization
```

## Mistake 6: Maximizing batch size blindly

The batch that maximizes raw throughput may violate latency SLOs.

## Mistake 7: Benchmarking only at saturation

You need the entire throughput-latency curve.

---

# 49. The complete mental model

The key relationships can be summarized as:

```text
REQUEST ARRIVAL
      |
      v
   QUEUEING
      |
      | affects TTFT
      v
   PREFILL
      |
      | affects TTFT
      v
 FIRST TOKEN
      |
      +-------------------- TTFT
      |
      v
    DECODE
      |
      | affects TPOT / ITL
      v
 FINAL TOKEN
      |
      +-------------------- E2E latency
```

At the system level:

```text
more concurrency
      |
      v
better batching
      |
      v
more weight reuse
      |
      v
higher arithmetic intensity
      |
      v
better GPU utilization
      |
      v
higher aggregate throughput
```

but simultaneously:

```text
more concurrency
      |
      v
more contention
      |
      +--> queueing
      +--> larger scheduling iterations
      +--> KV-cache pressure
      +--> prefill/decode interference
      |
      v
potentially higher per-user latency
```

Therefore the serving objective should usually be:

```text
NOT:

maximize throughput

BUT:

maximize sustainable goodput
while satisfying TTFT, TPOT,
tail-latency, reliability,
and cost objectives
```

---

# 50. Connection to the inference-engineering learning sequence

These concepts fit together as follows:

```text
Arithmetic Intensity
        |
        v
How much compute occurs per byte moved?
        |
        v
Roofline Model
        |
        v
Is a kernel compute-bound or memory-bound?
        |
        v
Prefill vs Decode
        |
        +--> Prefill often compute-bound
        |
        +--> Decode often memory-bound
        |
        v
Batching
        |
        v
Increase reuse and arithmetic intensity
        |
        v
Higher GPU throughput
        |
        v
Latency Metrics
        |
        +--> TTFT
        +--> TPOT
        +--> ITL
        +--> E2E
        |
        v
Throughput-Latency Trade-off
        |
        v
Continuous Batching
        |
        v
Chunked Prefill
        |
        v
Admission Control
        |
        v
Production LLM Scheduler Design
```

This progression is important because latency optimization is not independent from hardware performance.

Arithmetic intensity explains **why** batching improves hardware efficiency.

The Roofline model explains **when** that improvement is limited by bandwidth or compute.

TTFT and TPOT explain **what the user experiences**.

Queueing and scheduling explain **why maximum hardware throughput is not the same thing as maximum production performance**.

---

# Conclusion

LLM serving performance cannot be represented by one latency or throughput number.

**TTFT** measures how quickly generation begins. It is strongly influenced by queueing, scheduling, and prefill.

**TPOT** measures the average pace of subsequent token generation and is strongly influenced by decode execution, memory bandwidth, KV-cache traffic, batching, and scheduling.

**ITL** exposes the actual gaps users experience between streamed tokens.

**End-to-end latency** measures how long the complete request takes.

**Throughput** measures how much total work the serving system performs.

Batching can increase arithmetic intensity and GPU utilization, producing dramatically higher aggregate throughput. But larger batches and higher concurrency can also increase queueing, decode-iteration duration, memory pressure, and contention. As a result, the configuration that produces the highest raw tokens per second is not necessarily the configuration that provides the best user experience.

The production objective is therefore not simply:

```text
maximize tokens per second
```

It is closer to:

```text
maximize SLO-compliant throughput
at acceptable cost
while maintaining reliability,
fairness, and security.
```

That quantity is the practical meaning of **goodput**, and it is one of the most useful ways to think about production LLM inference optimization.
