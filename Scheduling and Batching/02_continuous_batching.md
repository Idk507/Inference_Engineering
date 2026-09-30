**Continuous batching rebuilds the GPU’s workload at each scheduling iteration.** Requests that still need computation remain eligible, completed requests leave, and waiting requests enter when token and memory budgets allow. The server no longer waits for an entire original batch to finish before starting new work.

The essential change from static batching is the scheduling boundary. Static batching makes an admission decision for a whole generation job. Continuous batching makes admission decisions throughout generation. NVIDIA describes in-flight batching as dynamically batching and scheduling requests at each LLM step. :chatgpt-content-reference{index="0"}

To understand its scheduler, separate three concepts: **a request’s lifecycle state, its inference phase, and whether it is selected for the next GPU execution.** Those concepts are related, but they are not interchangeable.

A request can be in the RUNNING state while still processing its prompt. A RUNNING request can also be temporarily omitted from an iteration because the scheduler has exhausted its budget. Conversely, a WAITING request may already have generated output if it was previously running and then preempted.

**The request state machine**

A simplified scheduler maintains a waiting queue, a running collection, and terminal request records. Some engines represent preempted or blocked requests through additional states; others put them back into a waiting structure. The following is a conceptual model rather than a promise that every serving engine uses these exact names.

| State | Meaning | Typical next transition |
|---|---|---|
| WAITING | Accepted, but awaiting admission or readmission | RUNNING, CANCELLED, FAILED |
| RUNNING | Admitted and eligible for model execution | Remain RUNNING, PREEMPTED, FINISHED, CANCELLED, FAILED |
| PREEMPTED | Paused so resources can serve other work | WAITING or RUNNING |
| FINISHED | Reached a valid stopping condition | Terminal |
| CANCELLED | Client or server stopped the request | Terminal |
| FAILED | Cannot continue because of an error | Terminal |

When a request arrives, the server validates it, tokenizes its prompt, checks limits, and creates its request record. Assuming it is accepted, the scheduler places it into WAITING. Receiving an HTTP request does not mean the GPU has started processing it.

Admission changes the request from WAITING to RUNNING after the scheduler establishes that the required execution resources are available. The request usually begins with prefill, potentially split across several iterations. After enough prompt computation has completed, it proceeds to autoregressive decoding.

During decoding, the request normally remains RUNNING across many iterations. It moves to FINISHED when it reaches its configured stopping condition. If memory pressure requires it to pause, it can instead transition through PREEMPTED and return for later admission.

The meaning of RUNNING is therefore **“admitted to the engine’s active workload,” not “occupying a GPU core continuously.”** The GPU executes kernels for selected work; a request state is a scheduler bookkeeping concept.

**What the scheduler stores for each request**

The scheduler needs considerably more than a prompt string. A request record typically tracks its identity, prompt token IDs, generated token IDs, sampling configuration, arrival time, priority, output limit, stopping conditions, and cancellation status. It also tracks execution progress and the KV-cache resources associated with that progress.

One useful accounting model distinguishes tokens whose representations still need to be computed from tokens whose KV state is already available. For a new uncached prompt containing 1,000 tokens, substantial prefill work is outstanding. For a request already decoding, the outstanding work is usually the newly sampled token that must be fed through the model to predict its successor.

That last detail matters. **Sampling a token and computing that token’s KV representation are different events.** The final prompt position can produce the first output token. A subsequent decode pass consumes that output token, updates the cache, and produces the next one. It is therefore inaccurate to assume that every stored generated token already has a corresponding computed KV-cache entry.

Prefix caching, speculative decoding, and asynchronous execution make this accounting more involved, but the underlying requirement stays the same: the scheduler must know what computation remains and what state is already reusable.

**The three main admission budgets**

A scheduler cannot simply replace every finished request with the next waiting request. It must satisfy several constraints simultaneously.

The sequence budget limits how many requests may be active or selected, depending on the engine’s configuration. The token budget limits how many input token positions can be processed in one iteration. The KV-cache budget limits how much attention state can be allocated or retained. vLLM explicitly exposes a maximum number of batched tokens per iteration and supports dividing prefill work according to the remaining budget. :chatgpt-content-reference{index="1"}

These budgets measure different things. A decode request usually contributes one input token to an ordinary iteration. A prefill request may contribute hundreds. Both are requests, but they place very different demands on that iteration.

For example, suppose the sequence limit is eight and the iteration token budget is 512. Six ongoing decode requests consume six token positions. The remaining token budget is 506. The scheduler could admit another request and process a 506-token prompt chunk, provided sequence capacity and KV-cache space permit it.

That does not mean the iteration produces 512 output tokens. Most of those positions belong to prompt processing. **The scheduling token budget counts model input work, not user-visible output.**

Token count is also an imperfect proxy for cost. Attention over a long existing context can be more expensive than attention over a short one, even when both requests contribute one new input token. Practical schedulers use token budgets because they are tractable, but performance tuning must also consider context lengths and measured execution times.

**The per-iteration loop**

Conceptually, each iteration starts by incorporating events the scheduler now knows about: newly arrived requests, completed execution results, cancellations, timeouts, and failures. Results from the previous execution determine which requests still need work.

The scheduler then retires requests that have finished or stopped. It removes them from the active collection and releases their resource ownership. This creates room for subsequent work.

Next, it resets the iteration’s execution budgets and selects work for existing RUNNING requests. A decode-oriented policy usually prioritizes ongoing decoding so users already receiving output do not experience large gaps between tokens. Partially processed prompts may also receive additional chunks.

For every selected request, the scheduler determines the number of token positions to process and asks the cache manager for any necessary allocation. If allocation fails, it may reduce selected work, defer admission, or preempt another request, according to its policy.

After accounting for existing work, the scheduler considers WAITING requests. It selects eligible candidates, checks their resource requirements, and admits as many as the remaining budgets support. A newly admitted request can receive a partial prefill rather than its whole prompt.

The scheduler then builds an execution plan containing selected request IDs, token ranges, positions, cache mappings, and the metadata required by the model runner. The GPU executes that plan. The runtime collects results, updates progress, samples where appropriate, streams available output, evaluates stopping conditions, and repeats.

This is a logical ordering. Real engines may overlap CPU scheduling, GPU execution, and output handling, or divide execution into microbatches. The important invariant is that resource ownership and execution progress remain consistent despite that overlap.

**A concrete walkthrough**

Assume the engine can keep three requests active and process at most ten input token positions per iteration. A and B are already decoding. C is waiting with an eight-token uncached prompt. Assume sufficient KV-cache space, no speculative decoding, and a policy that schedules ongoing decode first.

| Iteration | Scheduled work | Input-token budget used | Result |
|---|---|---:|---|
| 1 | A decode: 1; B decode: 1; C prefill: 8 | 10 | C enters RUNNING and finishes prefill |
| 2 | A decode: 1; B decode: 1; C decode: 1 | 3 | A reaches its stopping condition |
| 3 | B decode: 1; C decode: 1; D prefill: 8 | 10 | D enters RUNNING with two prompt tokens remaining |
| 4 | B decode: 1; C decode: 1; D prefill: 2 | 4 | D finishes prefill |
| 5 | B decode: 1; C decode: 1; D decode: 1 | 3 | All three continue generating |

D has a ten-token prompt and arrives before iteration 3. Once A finishes, its active-request place becomes available. D can enter while B and C continue generating.

D receives only eight prefill tokens in iteration 3 because B and C consume two of the ten available token positions. D remains RUNNING after that iteration, but is still in its prefill phase. Its final two prompt tokens are processed in iteration 4.

In an ordinary causal language model, completing prefill can also supply the logits used to sample the first output token. The table separates prefill and subsequent decode work; it does not imply that first-token sampling requires an additional full decode pass.

Notice that iteration 4 uses only four token positions. With all three request places occupied, no additional request can enter under this example’s sequence limit. **Continuous batching makes capacity reusable; it does not guarantee that every budget is exhausted on every iteration.**

**Why chunked prefill matters**

A replacement request is not instantly ready to decode. Its prompt must first be processed, unless sufficient state is reusable from a cache.

If a scheduler inserts a 20,000-token prefill into an iteration containing ongoing decodes, that prefill can substantially increase the interval before existing users receive their next tokens. Continuous admission without controlling prefill size can therefore improve aggregate throughput while harming interactive latency.

Chunked prefill divides prompt processing into bounded portions. The engine interleaves these portions with ongoing decoding across iterations. This allows a new request to make progress without forcing all its prompt work into one scheduling step.

The trade-off is between time to first token for new requests and time between tokens for existing ones. Larger prefill chunks can accelerate admission progress but lengthen mixed iterations. Smaller chunks can preserve decode responsiveness while making a new request’s prompt take more iterations to complete. TensorRT-LLM documents scheduling around request and token limits and discusses chunked context processing as part of in-flight batching. :chatgpt-content-reference{index="2"}

**Removal and preemption are different operations**

“Eviction” can be ambiguous. Removing a completed request from the active workload is normal retirement: the request no longer needs inference. Preempting an unfinished request is a resource-management decision: the request still needs inference, but the scheduler pauses it. Evicting an unused prefix-cache entry is a third operation, involving cached state rather than an active user request.

For a finished request, the engine finalizes output, marks the terminal state, and releases its ownership of KV-cache blocks. Those blocks are not necessarily destroyed immediately. Some may remain as reusable prefix-cache entries, with reference counting determining whether they are still in use.

For an unfinished request, a recomputation-based preemption policy can release its KV cache while preserving its prompt and generated token history. When it resumes, the engine rebuilds the necessary state by processing that history. This saves GPU memory but incurs extra computation and delays the request.

An offloading policy can instead move state to another memory tier and restore it later. That trades recomputation for transfer costs and additional implementation complexity. Not every engine or configuration supports the same options.

TensorRT-LLM distinguishes conservative admission intended to avoid pausing started requests from policies that pursue higher utilization and may pause requests under pressure. These are capacity-policy choices within continuous batching. :chatgpt-content-reference{index="3"}

**Why running requests can suddenly need more memory**

A request’s cache grows as its processed context grows. Suppose KV-cache blocks hold 16 token positions. A request with 31 cached positions can append one position within its existing two blocks. Processing the next position requires a third block.

Consequently, a request that was safely admitted earlier may later need an allocation the cache manager cannot satisfy. The scheduler must account for growth, rather than checking memory only at first admission.

A conservative policy reserves or budgets enough capacity for anticipated future growth, reducing preemption risk but potentially admitting fewer requests. An aggressive policy allocates closer to immediate need, increasing concurrency but accepting a greater risk of future pauses.

Repeated preemption can become counterproductive. The engine may spend substantial time rebuilding caches rather than producing new output. Goodput—useful completed work meeting latency targets—can fall even while the GPU remains busy.

**Fairness and queue ordering**

A FIFO waiting queue is straightforward, but its oldest request may not fit the remaining resources. If the scheduler refuses to consider later candidates, a large prompt can block smaller requests that would fit.

Skipping that request can improve utilization, but repeated skipping risks starvation. Priority queues, waiting-time aging, bounded bypass rules, and limits on concurrent large prefills can balance progress and efficiency.

The scheduler must also balance admission against ongoing generation. Always protecting decode can leave new requests waiting too long under sustained load. Always favoring new prefills can repeatedly interrupt existing streams. Continuous batching provides the opportunity to make these decisions; it does not determine the fairness policy automatically.

**Execution correctness and production behavior**

A batch is a temporary execution plan, so a request must have a stable identity independent of its row position. Request A might occupy row zero in one iteration and row two in another. Outputs, cache mappings, sampling state, and streaming destinations must be associated with the request ID rather than an assumed permanent slot.

Cancellation requires similar care. If a user disconnects while GPU execution is already in progress, the engine generally cannot reclaim buffers still referenced by that execution immediately. It records cancellation, suppresses further output as appropriate, and releases resources once the relevant execution has safely completed. Failure handling must likewise avoid freeing another request’s state or leaving orphaned allocations.

In production, track waiting time, active requests, scheduled decode and prefill tokens, KV-cache occupancy, preemption count, recomputation work, scheduler CPU time, time to first token, and inter-token latency. These measurements show whether admission improves useful throughput or merely creates memory pressure and latency spikes.

Bounded queues and admission control prevent overload from becoming unbounded waiting. Load balancing should consider active context and queued work rather than request count alone, while autoscaling should respond to latency and queue pressure. Authentication, authorization, encrypted transport, isolated request state, and carefully scoped cache reuse remain necessary when different users share an engine.

**The scheduler’s central responsibility is to turn available resources into the next safe, useful execution plan.** WAITING holds work that needs admission. RUNNING holds admitted work that needs further computation. Each iteration updates progress, retires completed requests, allocates resources for continuing work, admits eligible waiting requests, and executes the resulting plan. Preemption is the fallback when continued execution cannot fit; chunked prefill makes new admission compatible with responsive ongoing generation.
