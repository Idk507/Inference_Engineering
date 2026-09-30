## Chunked prefill

Chunked prefill splits one long prompt’s initial model computation across several scheduler iterations, instead of forcing the GPU to process the entire prompt in one large prefill operation.

This is important only because an LLM has two very different serving phases. **Prefill** reads the prompt and builds its KV cache. **Decode** generates one new token at a time using that cache. A long prompt can contain thousands of tokens, whereas an ongoing decode request usually contributes only one new token per iteration.

Without chunking, one newly arrived long prompt can monopolize an inference iteration and delay every user who is already streaming output.

### First, what prefill actually does

Take this prompt:

```text
"Summarize this 12,000-token legal document..."
```

The model must tokenize it, process all 12,000 tokens through every transformer layer, and write each token’s keys and values into the KV cache. Once this initial work is complete, the final prompt position produces logits from which the server samples the first generated token.

Conceptually:

```text
Prompt tokens
    ↓
Prefill forward pass
    ↓
KV cache for every processed prompt token
    ↓
Logits at final prompt position
    ↓
First output token
```

Prefill has parallelism across prompt positions, so it is much more compute-heavy than one decode token. But “parallel” does not mean free: attention, projections, feed-forward layers, and KV-cache writes still have to process all prompt positions.

A 12,000-token prompt might take a much longer GPU iteration than a normal decode iteration. If the same GPU is serving existing chat users, those users may see a long pause before their next streamed token.

### The problem with unchunked prefill

Assume requests A, B, and C are already generating. Each needs one decode token in the next iteration. Then request D arrives with an 8,000-token prompt.

With an unchunked prefill policy, the scheduler may produce a plan like:

| Iteration | GPU work | Consequence |
|---|---|---|
| 1 | Decode A, B, C | Users receive next tokens |
| 2 | Prefill D: 8,000 tokens | A, B, C wait |
| 3 | Decode A, B, C, D | Streaming resumes |

The issue is not that A, B, and C are removed from the active request set. They are still logically `RUNNING`. The problem is that the shared GPU is occupied by D’s giant prefill execution, so A, B, and C cannot obtain their next decode iteration until it completes.

This worsens **inter-token latency**, often abbreviated ITL or TPOT. It can make a response appear stalled despite the request being healthy.

At low traffic, unchunked prefill can be reasonable. The GPU can use a large efficient prefill batch, and the new request obtains its first token quickly. Under concurrent interactive traffic, it causes latency spikes because one long prompt blocks many short decode operations.

### The central idea

The server places a token budget on every GPU iteration:

```text
max_num_batched_tokens = maximum token positions processed in one iteration
```

Suppose that budget is 1,024 token positions. A, B, and C each need one decode token. They consume three positions, leaving:

```text
remaining prefill budget = 1,024 - 3 = 1,021 token positions
```

Rather than processing all 8,000 prompt tokens for D, the scheduler processes only 1,021 prompt tokens for D in this iteration.

```text
Iteration plan:
A decode:       1 token
B decode:       1 token
C decode:       1 token
D prefill:  1,021 tokens
-----------------------
Total:      1,024 tokens
```

D’s prompt is now partly processed. Its request remains in `RUNNING`, but it is still in the prefill phase. Its progress might look like:

```text
D prompt progress = 1,021 / 8,000 tokens
D remaining prefill = 6,979 tokens
```

At the next scheduling point, A, B, and C again receive their one-token decode work first. The scheduler spends the remaining budget on D’s next prompt chunk.

```text
Iteration 2:
A decode:       1 token
B decode:       1 token
C decode:       1 token
D prefill:  1,021 tokens
```

This continues until D’s last prompt chunk is processed. That final chunk produces D’s first output-token distribution. D can then join A, B, and C as a normal decoding request.

### The interleaving pattern

The key pattern is:

```text
Decode work first
    ↓
Use remaining token budget for prompt chunks
    ↓
Repeat at the next iteration
```

For the earlier example:

| Iteration | A | B | C | D | What changes |
|---|---:|---:|---:|---:|---|
| 1 | Decode 1 | Decode 1 | Decode 1 | Prefill 1,021 | D starts |
| 2 | Decode 1 | Decode 1 | Decode 1 | Prefill 1,021 | D progresses |
| 3 | Decode 1 | Decode 1 | Decode 1 | Prefill 1,021 | D progresses |
| 4 | Decode 1 | Decode 1 | Decode 1 | Prefill 1,021 | D progresses |
| 5 | Decode 1 | Decode 1 | Decode 1 | Prefill 1,021 | D progresses |
| 6 | Decode 1 | Decode 1 | Decode 1 | Prefill 1,021 | D progresses |
| 7 | Decode 1 | Decode 1 | Decode 1 | Prefill 1,021 | D progresses |
| 8 | Decode 1 | Decode 1 | Decode 1 | Prefill 853 | D prefill completes |
| 9 | Decode 1 | Decode 1 | Decode 1 | Decode 1 | D generates normally |

The exact values vary with the engine and model runner, but the control-flow principle is the same.

vLLM’s chunked-prefill scheduling gives decode work priority and uses remaining `max_num_batched_tokens` capacity for pending prefills. Its documentation notes that smaller token-budget values improve inter-token latency because fewer prefill tokens can delay decode work. :chatgpt-content-reference{index="0"}

### Why packed execution is essential

A naive batch representation creates a dense tensor shaped roughly like:

```text
[batch_size, sequence_length]
```

Suppose A, B, and C each contribute one decode token while D contributes 1,021 prompt tokens. Padding every request to 1,021 positions would create:

```text
A: 1 real token + 1,020 padding positions
B: 1 real token + 1,020 padding positions
C: 1 real token + 1,020 padding positions
D: 1,021 real prompt tokens
```

That would waste most of the work for A, B, and C.

Modern in-flight engines use **packed variable-length inputs**. The GPU receives a flat token buffer plus metadata identifying which ranges belong to which request.

```text
Packed tokens:
[D chunk tokens ......................][A][B][C]

Sequence metadata:
D → positions 0 through 1,020
A → position 1,021
B → position 1,022
C → position 1,023
```

Attention metadata, sequence offsets, position IDs, and KV block tables ensure that D attends only to D’s prior tokens and each decode token attends only to its own request history. The shared GPU batch does not mix user conversations.

TensorRT-LLM documents that in-flight batching requires packed, non-padded inputs for efficiency, and can process context-phase and generation-phase sequences together. :chatgpt-content-reference{index="1"}

### How the KV cache grows across chunks

Chunking does not change the model’s final interpretation of the prompt. It only changes when the prompt tokens are processed.

Suppose D has a 100-token prompt split into four chunks of 32, 32, 32, and 4 tokens.

```text
Before iteration 1:
KV cache for D = empty

After chunk 1:
KV cache for D = tokens 1–32

After chunk 2:
KV cache for D = tokens 1–64

After chunk 3:
KV cache for D = tokens 1–96

After chunk 4:
KV cache for D = tokens 1–100
First output token can be sampled
```

When processing chunk 2, the model uses the KV cache from tokens 1–32 and computes new keys and values for tokens 33–64. It therefore preserves causal attention exactly as if the 100-token prompt were processed in a single prefill pass.

The request must retain its partial KV state between iterations. It is not restarting prefill from token one each time. Restarting would turn an 8,000-token prompt into repeated redundant work and make chunking unusable.

### Why chunks often align to KV-cache blocks

Paged KV caches allocate memory in fixed-size blocks. If one block stores 16 token positions, a prefill chunk of 1,024 tokens occupies exactly 64 blocks.

```text
1,024 / 16 = 64 KV blocks
```

Aligned chunks simplify allocation, block-table creation, prefix-cache accounting, and kernel operation. A final chunk may be smaller because it contains the remainder. TensorRT-LLM notes that, except for the last context chunk, chunk sizes must be integer multiples of the KV-cache block size in its paged-context implementation. :chatgpt-content-reference{index="2"}

This alignment is an implementation requirement in some engines, not a property of transformer mathematics. The model can conceptually process arbitrary token counts; the serving system chooses block-aligned chunks for memory-management efficiency.

### Prefill priority versus decode priority

There is no universally correct policy. The scheduler must choose which users it protects.

A **prefill-first** policy gives new requests large chunks first. It usually improves time to first token for new arrivals, because their prompts finish quickly. But it can harm users already receiving streaming output, because their decode steps wait behind prompt computation.

A **decode-first** policy schedules every active decode first, then uses remaining capacity for prefill chunks. It protects inter-token latency and produces smoother streams. But new requests may wait through several iterations before their prompt completes, increasing their time to first token.

A useful way to view the trade-off is:

| Scheduling choice | New request TTFT | Existing request ITL | Typical use |
|---|---:|---:|---|
| Large prefill-first | Better | Worse | Offline or low-concurrency workloads |
| Decode-first chunked prefill | Slightly worse | Better | Interactive chat and agents |
| Strict decode-only during load | Often worse | Best | Very tight streaming latency SLOs |

An engine may change this policy based on queue pressure, request priority, remaining prompt length, or service class.

### Fairness between long prompts

Chunking solves the long-prompt-versus-decode problem, but it introduces a second decision: how should the remaining prefill budget be shared among multiple long prompts?

Suppose 1,000 token positions remain after decoding, and three waiting prompts have significant work left.

A first-come-first-served chunking policy might give all 1,000 positions to the oldest request:

```text
Iteration:
Request D prefill: 1,000 tokens
Request E prefill: 0
Request F prefill: 0
```

This minimizes D’s time to first token, but E and F wait.

An equal-progress policy could distribute budget:

```text
Iteration:
Request D prefill: 334 tokens
Request E prefill: 333 tokens
Request F prefill: 333 tokens
```

This increases fairness and lets multiple prompts advance, but delays D’s first token.

TensorRT-LLM exposes first-come-first-served and equal-progress context-chunking policies. The former continues the earliest request’s context chunks first; the latter advances requests more evenly. :chatgpt-content-reference{index="3"}

### Chunking and admission control must work together

Chunked prefill means a request may begin using the KV cache long before its whole prompt is complete. A dangerous scheduler would admit many huge prompts merely because their first chunks fit.

For example, suppose 20 requests each have a 32,000-token prompt. A scheduler with a 1,024-token chunk limit could start all 20 if it only checks each first chunk. But their partial caches grow across later iterations. Eventually, the KV pool fills, requests are preempted, and the engine starts discarding or rebuilding cache state.

Therefore, chunking needs admission control that asks at least:

```text
Does this request’s full prompt fit in the KV pool?
Can we retain a configured KV watermark after admission?
Can its declared maximum output also be supported?
```

The exact reservation policy depends on whether the system promises no eviction. But chunking must never become an excuse for unbounded partial admission.

### Why chunking can improve overall throughput

At first glance, splitting one large matrix operation into many smaller operations seems less efficient. Sometimes it is. But server throughput is not only about one request’s isolated prefill speed.

Without chunking, a huge prefill creates an oversized iteration that delays all decodes. With chunking, the engine mixes compute-heavy prompt positions with memory-bound decode positions. This can produce steadier iteration durations, better user latency, and better useful work under concurrent traffic.

It also permits a smaller `max_num_batched_tokens`. Without chunking, an engine may require the per-iteration budget to be at least as large as the longest allowed prompt. If the maximum prompt is 32,000 tokens, that can force a huge execution capacity. With chunking, the engine can support that prompt using many smaller iterations. NVIDIA notes that this permits smaller token-budget settings and prevents large prompts from becoming unschedulable when other requests are in flight. :chatgpt-content-reference{index="4"}

### Choosing a chunk budget

The main tuning parameter is usually the maximum token positions processed in a single iteration.

A larger budget means the scheduler can process larger prompt chunks. New prompts reach first-token readiness faster, and GPU arithmetic intensity may improve. However, mixed iterations become longer, increasing the delay before ongoing decodes receive their next token.

A smaller budget limits prefill interference and improves streaming responsiveness. But long prompts need more iterations, which can worsen their time to first token and may reduce aggregate throughput through more scheduling and kernel-launch overhead.

A reasonable production process is to benchmark with the actual prompt-length distribution and concurrency, then choose against latency objectives:

```text
Primary interactive metric:
P95 and P99 inter-token latency

New-request metric:
P95 and P99 time to first token

Capacity metric:
completed output tokens per second

Safety metrics:
KV utilization, watermark rejections, preemptions, recomputed tokens
```

Do not tune only for total tokens per second. A large prefill budget can produce attractive aggregate throughput while making active users experience long generation pauses.

### The complete scheduler logic

A practical decode-first chunked-prefill iteration has this shape:

```python
def schedule_iteration() -> ExecutionPlan:
    """Build one mixed decode and chunked-prefill GPU plan."""
    retire_finished_or_cancelled_requests()
    token_budget = max_num_batched_tokens

    # 1. Preserve responsiveness for users already streaming.
    for request in active_decode_requests():
        if token_budget == 0:
            break
        plan.add_decode(request, token_count=1)
        token_budget -= 1

    # 2. Continue partial prefills already admitted earlier.
    for request in active_prefill_requests():
        if token_budget == 0:
            break
        chunk = min(
            request.remaining_prompt_tokens,
            configured_chunk_limit,
            token_budget,
        )
        plan.add_prefill(request, token_count=chunk)
        token_budget -= chunk

    # 3. Admit safe waiting requests and give them an initial chunk.
    while token_budget > 0 and waiting_requests_exist():
        candidate = waiting_queue.peek()

        if not kv_admission_control.allows(candidate):
            break

        request = admit_to_running(candidate)
        chunk = min(
            request.remaining_prompt_tokens,
            configured_chunk_limit,
            token_budget,
        )
        plan.add_prefill(request, token_count=chunk)
        token_budget -= chunk

    return pack_variable_length_inputs_and_kv_metadata(plan)
```

The execution plan contains packed input token IDs, request-to-token offsets, positions, causal-attention metadata, KV block tables, sampling parameters, and output destinations. The GPU executes the forward pass. The runtime then updates each request’s processed prompt length, appends new KV state, samples tokens where logits are available, identifies terminal requests, releases completed resources, and begins the next scheduling iteration.

**Chunked prefill is therefore not simply “break a prompt into pieces.”** It is the scheduling discipline that lets a serving engine process a long prompt incrementally while continuing to advance short, latency-sensitive decode requests. It works because partial KV state persists across iterations, packed execution avoids padding waste, decode work receives deliberate priority, and admission control prevents too many incomplete prompts from exhausting the KV-cache pool.
