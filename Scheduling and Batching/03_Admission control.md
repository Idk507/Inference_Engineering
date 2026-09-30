## Admission control: protecting the KV-cache pool

Admission control decides whether a request may move from `WAITING` to `RUNNING`. Its core job is to ensure that accepting work now does not exhaust the GPU KV-cache pool a few decode iterations later.

This is necessary because a request does not consume a fixed amount of KV memory. Its KV cache grows as more tokens are processed. Every prompt token adds cached keys and values during prefill; every generated token extends that same cache during decoding. A request that fits when it arrives may become impossible to continue if the scheduler admits too many other requests meanwhile.

A safe admission controller treats the KV cache as a finite pool of fixed-size blocks.

```text
GPU memory
├── Model weights
├── CUDA/runtime workspace
├── Activations and communication buffers
└── KV-cache pool
    ├── blocks owned by active requests
    ├── reusable prefix-cache blocks
    └── free blocks ← admission control protects this region
```

If the KV pool contains `N_total` blocks, has `N_used` currently allocated blocks, and a candidate requires `N_candidate` additional blocks, naive admission would use:

```text
admit if N_used + N_candidate <= N_total
```

That permits the pool to become completely full. It looks efficient, but it is unsafe for continuous generation: active requests still need blocks as they produce more tokens, rounding effects occur at block boundaries, cache reuse may change allocation availability, and the runtime needs room to handle asynchronous scheduling and newly materialized pages.

A watermark reserves free capacity:

```text
N_watermark = ceil(watermark_fraction × N_total)

admit if N_used + N_candidate <= N_total - N_watermark
```

For a pool of 10,000 blocks and a 10% watermark:

```text
N_watermark = ceil(0.10 × 10,000) = 1,000 blocks
admission ceiling = 10,000 - 1,000 = 9,000 blocks
```

Even if the candidate fits in the physical 10,000-block pool, it is not admitted once doing so would push allocation beyond 9,000 blocks. The remaining 1,000 blocks are a safety reserve.

vLLM documents this watermark specifically as a fraction of total KV-cache blocks held free when admitting waiting or preempted requests, to reduce cache eviction and repeated preemption under memory pressure. :chatgpt-content-reference{index="0"}

## Why “it fits now” is not enough

Assume a pool of 1,000 KV blocks, each holding 16 token positions. The pool holds roughly 16,000 cached token positions, subject to the model’s layout and allocator metadata.

Suppose ten active requests use 850 blocks. A new request has a 512-token prompt and permits 256 generated tokens. Its lifetime upper bound is:

```text
maximum sequence length = 512 prompt tokens + 256 output tokens
                        = 768 tokens

blocks required = ceil(768 / 16)
                = 48 blocks
```

A naive scheduler admits it because:

```text
850 + 48 = 898 <= 1,000
```

But the currently running requests may collectively generate hundreds more tokens. If they later require another 150 blocks, then:

```text
850 existing blocks
+ 48 newly admitted request reservation or usage
+ 150 future growth
= 1,048 blocks
```

The GPU has only 1,000 blocks. At that point, the server must pause, swap, offload, or recompute some requests’ KV state. The GPU may remain busy, but useful throughput drops because it is doing recovery work rather than progressing user requests.

This is **KV-pool over-commitment**: the scheduler has promised more future cache growth than the pool can support.

## The two admission models

A conservative, no-eviction scheduler reserves enough cache capacity for each admitted request’s entire allowed lifetime. For request `r`:

```text
tokens_to_reserve(r) =
    uncached_prompt_tokens(r)
  + maximum_remaining_output_tokens(r)

blocks_to_reserve(r) =
    ceil(tokens_to_reserve(r) / tokens_per_block)
```

Then it admits only if the sum of all reservations remains below the safe capacity:

```text
sum(reserved_blocks for active requests)
+ candidate_reserved_blocks
<= total_blocks - watermark_blocks
```

This makes an important guarantee: once the server begins a request, it can continue it through its declared maximum length without being preempted solely because the KV cache fills. TensorRT-LLM calls this `GUARANTEED_NO_EVICT`; it is its default conservative capacity policy. :chatgpt-content-reference{index="1"}

The downside is underutilization. Most requests stop early through EOS tokens, stop strings, cancellation, or a lower-than-maximum actual output length. Reserving all possible future blocks means memory is held for output tokens that may never exist.

An aggressive scheduler instead reserves only the immediate or near-term cache requirement:

```text
immediate_blocks(r) =
    blocks needed for its current prefill chunk
    or next decode increment
```

It admits more requests and raises short-term utilization. However, if multiple requests continue generating, the pool can later fill. The scheduler must preempt some running requests, move them to host memory if supported, or discard their cache and recompute it on resume.

TensorRT-LLM describes this as the `MAX_UTILIZATION` policy: it greedily packs work at each forward iteration, at the risk of pausing requests when the KV limit is reached. :chatgpt-content-reference{index="2"}

The practical distinction is:

| Policy | Admission test | Memory efficiency | Latency predictability | Preemption risk |
|---|---|---:|---:|---:|
| Guaranteed no evict | Full worst-case request lifetime fits | Lower | Higher | Minimal |
| Aggressive utilization | Near-term work fits | Higher initially | Lower under load | Higher |
| Watermark-based hybrid | Controlled reservation plus free headroom | Balanced | Usually stable | Reduced |

## Watermarks are not merely unused memory

A watermark is intentional slack. Its purpose is to absorb uncertainty and allocator granularity.

Suppose each block stores 16 tokens. A request at 31 cached tokens uses two blocks. When it grows to 33 tokens, it needs three blocks. That request adds one token of logical sequence length but requires one entire new physical block.

If hundreds of active requests sit near a block boundary, a single decode iteration can create a burst of block allocations. Without reserve capacity, many requests can become unable to advance at the same time.

Watermarks also protect against differences between scheduler bookkeeping and allocator reality. A scheduler may estimate cache demand in tokens, whereas the allocator manages pages or blocks. Prefix-cache sharing, partially filled tail blocks, multi-layer cache pools, sliding-window attention, speculative decoding rollback, and asynchronous execution all make the exact availability more complex than a single scalar token count.

The safe rule is that the cache manager is authoritative. The scheduler proposes admission; the allocator confirms whether the concrete block table can be assigned. If the allocator rejects an allocation, the scheduler must defer or preempt safely rather than treating its estimate as proof.

## Full-input reservation prevents chunked-prefill over-admission

Chunked prefill creates a subtle failure mode. Consider a 32,000-token prompt processed in chunks of 1,024 tokens. If the scheduler checks only whether the first 1,024-token chunk fits, it can admit many long requests simultaneously:

```text
Request A: first chunk fits
Request B: first chunk fits
Request C: first chunk fits
...
```

Each request makes partial progress, but the combined remaining prompts may be far larger than the pool can ever hold. As they continue, allocations fail, requests are preempted, then restored or recomputed, and the engine repeatedly reprocesses work. This is KV-cache thrashing.

A full-input reservation check asks a different question:

```text
Can this request’s entire input sequence fit in the cache,
while preserving the configured watermark?
```

If the answer is no, the request remains in `WAITING` even though its first chunk could technically run. Current vLLM configuration documentation explicitly describes this option as preventing over-admission and KV-cache thrashing with chunked prefill. :chatgpt-content-reference{index="3"}

For interactive serving, full prompt reservation is usually the safer default. For very long-context workloads where a complete lifetime reservation is too restrictive, use bounded output limits, explicit fairness controls, preemption metrics, and a carefully tuned watermark rather than disabling protection blindly.

## Admission gate inside the per-iteration loop

The continuous-batching loop should make the admission decision after retiring completed requests and before constructing the GPU execution plan.

```python
def can_admit(candidate, kv_pool, config) -> bool:
    """Return whether a request can enter RUNNING safely."""
    watermark_blocks = math.ceil(
        config.kv_watermark_fraction * kv_pool.total_blocks
    )
    safe_capacity = kv_pool.total_blocks - watermark_blocks

    candidate_blocks = estimate_lifetime_blocks(
        prompt_tokens=candidate.uncached_prompt_tokens,
        max_new_tokens=candidate.remaining_output_limit,
        tokens_per_block=kv_pool.tokens_per_block,
    )

    projected_reserved = (
        kv_pool.reserved_blocks_for_running_requests()
        + candidate_blocks
    )

    return (
        projected_reserved <= safe_capacity
        and active_request_count() < config.max_running_requests
        and candidate.prompt_tokens <= config.max_model_len
    )
```

A production implementation must not use only this pure estimate. After this policy test passes, it should request actual blocks from the KV-cache manager. That allocator must atomically validate and commit the allocation, so two concurrent scheduler paths cannot both admit requests against the same free blocks.

The operational sequence is:

```text
1. Process completion and cancellation events.
2. Release terminal requests’ active KV ownership.
3. Determine active decode and prefill work.
4. Calculate free and reserved KV capacity.
5. Apply the watermark admission gate to waiting requests.
6. Atomically allocate concrete blocks for admitted work.
7. Build the packed GPU execution plan.
8. Execute one model iteration.
9. Update block tables and request progress.
10. Repeat.
```

The distinction between `allocated`, `reserved`, and `free` is important. `allocated` is physical KV memory already assigned. `reserved` is capacity promised to active requests for future growth. `free` is physical memory not currently assigned. A no-eviction design should ensure:

```text
allocated_blocks <= reserved_blocks <= safe_capacity
```

An aggressive scheduler can temporarily allow future logical demand to exceed safe capacity, but then it must implement reliable preemption and recovery.

## High and low watermarks

A single admission watermark is sufficient for a basic scheduler. Production systems often benefit from hysteresis through two thresholds.

Let:

```text
high watermark = 90% KV usage
low watermark  = 80% KV usage
```

When utilization rises above 90%, the scheduler stops admitting new memory-growing work. It continues current work and waits for completions. It resumes normal admission only after usage falls below 80%.

Without hysteresis, a scheduler can oscillate:

```text
admit request → usage crosses threshold
reject request → one block frees
admit again → usage crosses threshold
reject again
```

That oscillation creates unstable queue delay and unnecessary scheduler activity. The gap between high and low thresholds makes admission behavior stable.

In terms of free blocks rather than used blocks:

```text
stop admissions when free_blocks < reserve_high
resume admissions when free_blocks > reserve_low
```

Here, `reserve_high` is larger than `reserve_low`.

## How to choose the watermark

There is no universal percentage. The correct value depends on traffic distribution, model architecture, cache dtype, block size, prefix-cache hit rate, max sequence length, and latency objectives.

For stable interactive chat traffic, start with a conservative reserve based on observed simultaneous growth. Measure the 95th or 99th percentile of blocks allocated per decode iteration across all active requests, then hold several iterations of that growth as reserve. Conceptually:

```text
watermark_blocks =
    safety_multiplier
    × p99(active block growth per iteration)
    × recovery_window_iterations
```

For example, if the P99 aggregate growth is 40 blocks per iteration and the system needs roughly five iterations of safety margin, then a starting reserve might be:

```text
watermark_blocks = 1.5 × 40 × 5 = 300 blocks
```

The multiplier protects against correlated bursts, estimator error, and block-boundary effects. This is better than selecting “10%” blindly because it ties the reserve to the workload’s actual behavior.

A small watermark improves immediate admission and may improve short benchmark throughput. A large watermark reduces concurrent requests and can leave compute capacity unused. The right objective is not maximum apparent utilization; it is high **goodput**: completed requests that meet time-to-first-token and inter-token latency objectives.

## Production metrics and alerts

Monitor both physical pool behavior and scheduling outcomes. The most useful signals are KV-cache block utilization, free-block count, watermark rejections, admission wait time, active versus waiting request count, reserve-to-actual allocation ratio, preemption count, recomputed token count, host/offload transfer volume, time to first token, time per output token, and completed output tokens per second.

A healthy conservative scheduler shows watermark rejections during bursts, but very low preemption and recomputation. An unhealthy aggressive scheduler often shows high GPU utilization alongside rising preemption, high P95 inter-token latency, oscillating cache utilization, and growing queue delay.

If watermark rejection is continuously high while the GPU is underutilized, investigate a too-small KV pool, excessive worst-case reservation, oversized `max_new_tokens`, long prompts, fragmented or multi-pool cache allocation, or a sequence-count limit that is disconnected from real token capacity. TensorRT-LLM preallocates its paged KV pool from configured capacity and schedules in-flight work according to available cache space and the chosen policy. :chatgpt-content-reference{index="4"}
