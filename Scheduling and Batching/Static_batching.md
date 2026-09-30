**Static batching groups several inference requests together, starts them as one batch, and keeps that batch’s membership fixed until its work finishes.** In autoregressive language-model serving, the central inefficiency is that requests finish at different times, but newly arriving requests cannot take over their places. The GPU either continues processing inactive rows or runs progressively smaller batches while other requests wait in the queue. The longest-running request determines when the next group can start.

To understand precisely why this wastes GPU time, we need to distinguish three things: useful computation, unnecessary computation, and available capacity that the scheduler fails to use. A GPU can show high utilization while spending part of its time on useless work, or it can remain busy processing one request while delivering much less throughput than it could with several requests together.

**Why batching exists in the first place**

A language model applies the same learned weights to every request. Suppose a particular layer transforms a vector containing 4,096 numbers into another vector containing 4,096 numbers. With one request, the operation is roughly a vector multiplied by a weight matrix. With multiple requests, the input vectors can be stacked into a matrix, and the GPU performs a matrix multiplication using the same weights for all of them. Each request remains independent; batching does not allow one user’s tokens to attend to another user’s tokens.

That arrangement is valuable because GPUs execute large amounts of parallel arithmetic efficiently. Batching also lets the runtime amortize kernel launches and reuse model weights across multiple token computations. During decoding, small batches are often limited by memory bandwidth: the GPU must move substantial weight data to perform comparatively little arithmetic. Processing several requests together increases the useful work performed for those weight transfers. NVIDIA describes the distinction between the more compute-intensive prefill phase and the frequently memory-bound decode phase in its inference optimization guidance. :chatgpt-content-reference{index="0"}

For a simplified calculation, consider a dense model with 8 billion parameters stored using two bytes per parameter. Its raw weights occupy approximately 16 GB, ignoring auxiliary data and implementation overhead. If an inference step effectively streams those weights from GPU memory at an effective bandwidth of 1 TB/s, the weight-transfer component alone is approximately 16 milliseconds. This is an illustrative bandwidth calculation, not a prediction of the model’s actual latency.

With one active request, that step advances one sequence. With eight active requests, a well-batched implementation can reuse the weights while advancing eight sequences. The step will generally take longer because additional computation and KV-cache traffic are involved, but it need not take eight times as long. This is the economic reason batching improves throughput: **one traversal of the model can produce useful progress for several requests.**

**What a static batch does from arrival to completion**

Imagine a server configured to process up to four requests together. Incoming requests first enter a queue. The server selects requests A, B, C, and D, tokenizes their prompts, prepares their input tensors and attention metadata, allocates or associates their KV-cache storage, and performs prefill. Prefill processes each prompt and produces the state needed to begin generation.

The runtime then repeatedly executes decoding iterations. In ordinary autoregressive decoding without speculative generation, each iteration advances each unfinished sequence by one token. The next token for A depends on A’s previous tokens, but A, B, C, and D can advance together because their sequences are independent.

After each iteration, the runtime checks whether a sequence has reached an end-of-sequence token, a stop condition, a generation limit, or a cancellation condition. A completed request no longer needs useful decoding computation. However, under the static scheduling policy considered here, a waiting request E cannot enter that batch. The runtime continues until every original request has finished, then schedules another group.

“Static” therefore describes the membership boundary. It does not necessarily mean every batch has exactly the same size, that inputs are always padded, or that the execution uses a statically allocated KV cache. Those are separate implementation choices.

**The long-tail problem, with exact arithmetic**

Suppose the four requests generate 10, 20, 30, and 100 tokens respectively. Assume their prompts have already been processed, each decoding iteration takes 10 milliseconds, and the implementation retains four execution rows throughout generation. These assumptions deliberately simplify the example so we can isolate the scheduling waste.

The batch runs for 100 iterations because request D needs 100 tokens. Its useful work changes over time:

| Decode iterations | Requests still generating | Useful rows out of four | Unused rows |
|---|---|---:|---:|
| 1–10 | A, B, C, D | 4 | 0 |
| 11–20 | B, C, D | 3 | 1 |
| 21–30 | C, D | 2 | 2 |
| 31–100 | D | 1 | 3 |

The batch produces 10 + 20 + 30 + 100 = **160 useful output tokens**. But a fixed four-row execution shape provides 4 × 100 = **400 token-row opportunities**. Therefore, useful token-row occupancy is 160 / 400 = **40%**, and the remaining **60%** represents finished rows across subsequent iterations.

You can verify the wasted opportunities interval by interval. Iterations 11–20 contain 10 × 1 = 10 unused rows. Iterations 21–30 contain 10 × 2 = 20. Iterations 31–100 contain 70 × 3 = 210. Together, that is 240 unused token-row opportunities.

The general formula is straightforward. If B requests generate lengths L₁ through L_B, the batch needs L_max = max(L₁, …, L_B) iterations. Useful token-row occupancy is:

**Occupancy = sum of output lengths / (batch size × longest output length).**

This measures how much of a fixed batch’s decoding capacity contributes to real output. It does **not** directly measure GPU utilization, wasted FLOPs, or recoverable wall-clock time. In particular, 60% unused token-row opportunities does not prove that removing them would make inference 60% faster. GPU costs are shared, and step latency does not scale linearly with the number of active rows.

**There are two different ways the GPU loses efficiency**

In a simple fixed-shape implementation, completed requests remain represented in the input tensors. The runtime might feed dummy tokens or retain inactive rows, then discard the corresponding outputs. Dense operations can still calculate projections, feed-forward activations, and other intermediate values for those rows. Here, some GPU arithmetic contributes to no useful response.

An attention mask does not automatically eliminate that work. A mask controls which positions may influence the result; it does not, by itself, tell every GPU kernel to omit all arithmetic associated with an inactive row. Whether work is skipped depends on the operation, kernel, and representation.

A more capable implementation can remove completed rows and compact the remaining active requests. That avoids some or much of the dummy-row computation. However, if batch membership remains closed, the freed capacity still cannot serve queued requests. In our example, the runtime eventually executes batches containing only D. The GPU remains busy, but weight-transfer and launch costs are shared across one useful token rather than four.

This is the crucial distinction: **static batching can waste executed work, or it can waste the opportunity to execute more useful work alongside the work already happening.** Compaction addresses the first problem. Refilling the active batch addresses the second.

**Why uneven prompt lengths can add another source of waste**

Before generation even begins, requests may have different prompt lengths. Suppose four prompts contain 100, 200, 300, and 1,000 tokens. A conventional padded representation expands every sequence to 1,000 positions, producing 4,000 input positions even though only 1,600 contain real prompt tokens.

The useful-position ratio is again 1,600 / 4,000 = 40%. In a dense padded implementation, token-wise operations may process the padded positions as well as the real ones. Attention masks preserve the intended behavior, but do not necessarily eliminate all associated computation. Attention work also depends on the chosen attention kernel and sequence representation, so the 60% padding fraction cannot be treated as an exact percentage of total prefill FLOPs or time.

This overhead is avoidable without changing the batch’s membership policy. Packed or variable-length execution can process real tokens with metadata describing sequence boundaries. NVIDIA Triton’s ragged batching documentation explicitly describes avoiding padding for inputs with different lengths. Consequently, prompt padding is a common companion to simple static batching, but it is not an unavoidable property of static batching itself. :chatgpt-content-reference{index="1"}

**How the waste appears as user latency**

Return to the decoding example, where each iteration takes 10 milliseconds. A finishes after approximately 100 milliseconds, B after 200 milliseconds, C after 300 milliseconds, and D after 1,000 milliseconds, excluding prefill and other overhead.

A server that streams or returns results per request can deliver A’s completed response at 100 milliseconds. Static batch membership does not inherently force A’s user to wait until D finishes. However, an implementation that returns the entire batch together would delay A’s response until approximately 1,000 milliseconds. Those are different response-delivery policies built on the same static scheduling boundary.

Now suppose E arrives 110 milliseconds after decoding begins. A has already finished, but E still cannot occupy the freed place. Under this example’s single-worker, closed-batch policy, E waits approximately another 890 milliseconds before its own prefill can begin. The GPU spends much of that interval processing fewer useful sequences even though E is ready to run.

This is a form of head-of-line blocking: a long-running request delays admission of later work. In a system with several workers, another worker might serve E, so the exact wait depends on routing and available capacity. The underlying inefficiency remains wherever each worker refuses new admissions until its current batch finishes.

**Waiting to form a batch is a separate cost**

A server may also wait for enough arrivals to fill a batch. If it insists on collecting eight requests, light traffic can leave the GPU idle while the first request waits for seven companions. A timeout can bound that delay, but it creates a trade-off: longer waits tend to produce fuller batches, while shorter waits tend to improve response latency.

This collection delay happens before execution. The finished-request problem happens during execution. A system can solve the collection delay with a timeout and still suffer from a closed batch during generation.

The terminology matters here. Request-level dynamic batching groups whatever requests arrive within a bounded scheduling window. It is especially useful for models that finish in one forward pass. Continuous batching, also called iteration-level or in-flight batching, can change the active request set between generation iterations. NVIDIA Triton distinguishes ordinary dynamic batching from scheduling iterative sequences in this way. :chatgpt-content-reference{index="2"}

**Why memory can make the problem worse**

Every active autoregressive request needs a KV cache holding the keys and values for its processed tokens. Longer contexts require more cache storage. In a simple implementation that reserves cache capacity according to maximum sequence lengths, shorter requests may receive more space than they actually use. If completed requests’ storage also remains tied to the batch until its end, memory that could support new work remains unavailable.

Neither behavior is required by static membership. A runtime can grow cache allocations incrementally and free a completed request’s cache immediately while still refusing to admit new requests. Nevertheless, memory and scheduling interact: unused reservations reduce the batch size the GPU can accommodate, while closed admission prevents released memory from being used promptly.

Paged KV-cache management addresses allocation granularity and fragmentation. Continuous batching addresses request admission and removal. They complement each other, but one does not automatically provide the other.

**What continuous batching changes**

With continuous batching, the runtime treats each decoding iteration as a scheduling opportunity. When A finishes, the scheduler releases its state, considers waiting requests, and admits eligible new work. Later, B and C can also leave while replacement requests enter. The aim is to maintain a useful active workload instead of allowing the original batch to shrink until only its longest sequence remains.

Replacement is not free. A new request must undergo prefill before ordinary decoding. A large prefill can consume significant compute and delay existing requests’ next tokens. Effective schedulers therefore enforce token and memory budgets, and may split prefill into chunks so that admission does not cause excessive decode latency. NVIDIA’s inference guidance describes in-flight batching as removing completed sequences and admitting new ones instead of waiting for the entire group to finish. :chatgpt-content-reference{index="3"}

Continuous batching also needs enough waiting work to refill capacity. If no requests are queued, unused capacity cannot be recovered through better scheduling. Its benefits are strongest when there is sustained demand, variable generation lengths, and sufficient memory to support additional sequences.

**When static batching is still reasonable**

Static batching is simple and can be efficient for uniform workloads. If every request generates exactly 100 tokens, the fixed-row occupancy formula becomes 100%. There is no early-completion tail to refill. Similar reasoning applies to many image-classification or embedding workloads: each request usually finishes after one forward pass, so there is no long autoregressive loop in which requests repeatedly leave.

Fixed execution shapes can also help optimize kernels and reduce launch overhead through mechanisms such as CUDA graph replay. However, fixed kernel shapes and static request membership are independent concepts. A continuously batched runtime can still use a collection of fixed-shape execution graphs, sometimes padding to a captured size. NVIDIA documents this compute-versus-launch-overhead trade-off for CUDA graph batch sizes. :chatgpt-content-reference{index="4"}

For offline generation, grouping prompts by length can reduce padding, and grouping workloads with similar expected output lengths can reduce the tail. Output lengths are often uncertain, though, so such grouping improves the probability of a balanced batch rather than guaranteeing one.

**How to measure the actual production impact**

GPU utilization alone cannot establish whether batching is efficient. A kernel processing dummy rows may keep the device busy. Conversely, a small useful decode batch may be limited by memory bandwidth while achieving poor aggregate token throughput. The practical measurements are useful output tokens per second, completed requests per second, queueing delay, time to first token, time between streamed tokens, active sequence count, padding fraction, KV-cache usage, and latency distributions.

To investigate static batching specifically, record active sequence count throughout each batch. If batches start with many requests and spend long periods with only one or two active requests while the queue remains nonempty, the scheduler is leaving throughput available. Combine that trace with kernel timings and memory metrics to determine whether the implementation is executing inactive rows or merely running small batches.

Production controls should support that scheduling goal. Admission limits must account for token lengths and KV-cache demand, load balancing should consider active work and queued tokens, and autoscaling should consider queue delay and latency targets alongside GPU metrics. Authentication, authorization, encrypted transport, and request-isolated state remain essential when unrelated users share execution; telemetry should identify scheduling events without unnecessarily recording prompt contents.

**Precisely, static batching wastes GPU time because it ties admission to the completion of an entire group, even though useful capacity becomes available request by request.** Fixed-shape execution may spend arithmetic on completed rows; compacted execution may repeatedly traverse the model for too few active sequences; padded inputs may add unnecessary prefill work; and queued requests may wait despite available capacity. The central issue is the closed scheduling boundary. Batching creates efficiency through shared execution, but static membership prevents the server from maintaining that efficiency as requests finish.
