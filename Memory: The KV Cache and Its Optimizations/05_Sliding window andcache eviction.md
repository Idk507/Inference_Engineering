# Sliding Window Attention and KV-Cache Eviction

After understanding **KV cache, MHA/MQA/GQA, PagedAttention, and prefix caching**, the next problem is almost unavoidable: **what happens when the conversation becomes longer than the amount of KV cache we can afford?**

Suppose a model has a 128K context window. That does not mean serving 128K tokens is free. The model still needs to maintain K and V information for the tokens it is allowed to attend to. With a large number of concurrent users, those KV tensors can consume tens or hundreds of gigabytes of GPU memory.

So the central question becomes:

> **Can we keep only the KV information that is actually useful, instead of keeping every previous token forever?**

That is where **sliding-window attention** and **KV-cache eviction** come in.

---

# 1. Start from the normal KV cache

Consider a conversation:

```text
User: My name is Dhanush.
Assistant: Nice to meet you.
User: I am learning inference engineering.
Assistant: Great. Let's study KV caching.
User: Explain PagedAttention.
Assistant: ...
User: Now explain sliding-window attention.
```

During generation, the transformer normally maintains something conceptually like:

```text
Token 0  → K0, V0
Token 1  → K1, V1
Token 2  → K2, V2
Token 3  → K3, V3
...
Token N  → KN, VN
```

When generating token N+1, the new query can attend to the previous keys and values.

Conceptually:

```text
Q(N+1)
   |
   +---- K0, V0
   +---- K1, V1
   +---- K2, V2
   +---- ...
   +---- KN, VN
```

The problem is that the cache grows with every token.

If we have:

```text
1,000 tokens
10,000 tokens
100,000 tokens
1,000,000 tokens
```

the amount of KV memory grows correspondingly.

The fundamental KV-cache formula we discussed earlier is:

```text
KV bytes per token
=
2 × number of layers
  × number of KV heads
  × head dimension
  × bytes per element
```

For our earlier Llama-3-8B-style example:

```text
layers       = 32
KV heads     = 8
head dim     = 128
BF16         = 2 bytes
```

Therefore:

```text
2 × 32 × 8 × 128 × 2
= 131,072 bytes
```

So:

```text
1 token   = 128 KiB
8K tokens = 1 GiB
32K       = 4 GiB
64K       = 8 GiB
128K      = 16 GiB
```

That is for **one sequence**.

If a server has many users, the memory pressure becomes enormous.

This gives us the motivation for bounded KV memory.

---

# 2. The basic idea of a sliding window

Instead of allowing the model to attend to the entire history, we define a maximum attention window.

For example:

```text
window size = 4096 tokens
```

Suppose the conversation has reached token 10,000.

Instead of retaining:

```text
0 ... 9999
```

we retain approximately:

```text
5904 ... 9999
```

Then when token 10,000 arrives, the window moves forward.

Conceptually:

```text
Before:

[0................................9999]
                                  ^
                                current


After:

[5904...........................9999]
```

Then later:

```text
[5905...........................10000]
```

Then:

```text
[5906...........................10001]
```

The window continuously slides forward.

That is why it is called **sliding-window attention**.

---

# 3. Why does this bound memory?

This is the important part.

Without a sliding window:

```text
KV memory ∝ sequence length
```

So:

```text
1K → 1K worth of KV
10K → 10K worth
100K → 100K worth
```

With a window of size W:

```text
KV memory ≈ W × KV-bytes-per-token
```

Once the sequence becomes larger than W, the cache stops growing proportionally with the conversation.

For our example:

```text
KV/token = 128 KiB
window = 4096 tokens
```

Therefore:

```text
4096 × 128 KiB
= 524,288 KiB
= 512 MiB
```

So one layer? No — this **512 MiB is the entire 32-layer KV cache** because our 128 KiB/token number already included all layers.

That distinction is important.

A 4K sliding window therefore gives approximately:

```text
512 MiB per active sequence
```

instead of:

```text
8K  → 1 GiB
32K → 4 GiB
128K → 16 GiB
```

The memory becomes approximately constant with respect to conversation length.

---

# 4. But there is a subtle problem

If we throw away old tokens, the model can no longer attend to them.

Suppose the conversation is:

```text
Token 0:
My database password is ...

Token 1:
...

Token 5000:
What was the password I told you earlier?
```

If token 0 has already been evicted, the model cannot directly retrieve it from the KV cache.

This is the fundamental tradeoff:

```text
More history
    ↓
better long-range access
    ↓
more KV memory
```

versus:

```text
Less history
    ↓
bounded KV memory
    ↓
less long-range attention
```

Sliding-window attention deliberately chooses bounded memory at the cost of unrestricted historical attention.

---

# 5. Sliding window is an attention rule, not merely a cache trick

This distinction is extremely important.

People sometimes say:

> "Sliding window means delete old KV cache entries."

That is incomplete.

A true sliding-window attention mechanism changes **which tokens the model is allowed to attend to**.

Suppose:

```text
window = 4
```

For token 9, the model might attend to:

```text
6 7 8 9
```

but not:

```text
0 1 2 3 4 5
```

So the attention pattern itself becomes local.

Conceptually, instead of:

```text
Q9
 |
 +-- K0
 +-- K1
 +-- K2
 +-- ...
 +-- K9
```

we have:

```text
Q9
 |
 +-- K6
 +-- K7
 +-- K8
 +-- K9
```

The model's architecture therefore establishes a local receptive field.

---

# 6. Normal causal attention versus sliding-window attention

Normal causal attention looks approximately like this:

```text
Token       Can attend to

0           0
1           0 1
2           0 1 2
3           0 1 2 3
4           0 1 2 3 4
5           0 1 2 3 4 5
6           0 1 2 3 4 5 6
```

The amount of accessible history continually increases.

With a window of 4:

```text
Token       Can attend to

0           0
1           0 1
2           0 1 2
3           0 1 2 3
4           1 2 3 4
5           2 3 4 5
6           3 4 5 6
7           4 5 6 7
```

Once the window reaches its maximum size, the oldest token disappears from the attention region.

This is what actually bounds the attention state.

---

# 7. Sliding window versus cache eviction

These concepts are related but not identical.

**Sliding-window attention** says:

> The model is architecturally allowed to attend only to the most recent W tokens.

**KV-cache eviction** says:

> We physically remove KV entries that are no longer needed or useful.

If the attention mechanism guarantees that tokens older than W will never be accessed, then their KV entries can safely be evicted.

This gives us:

```text
Attention policy
       ↓
determines which KV entries are needed
       ↓
cache manager
       ↓
evicts everything outside the allowed region
```

So the attention pattern provides the correctness rule for eviction.

---

# 8. The simplest eviction algorithm

Imagine:

```text
window_size = 4096
```

and a request has:

```text
current_length = 5000
```

Then:

```text
tokens_to_keep = last 4096 tokens
tokens_to_remove = first 904 tokens
```

Conceptually:

```python
start = max(0, current_length - window_size)
end = current_length

active_tokens = cache[start:end]
```

When another token arrives:

```text
current_length = 5001

start = 5001 - 4096
      = 905
```

Now the active region becomes:

```text
905 ... 5000
```

The window moved forward by one token.

---

# 9. But production systems don't usually shift tensors one token at a time

This is where **PagedAttention** becomes extremely relevant.

Earlier we saw that PagedAttention divides KV memory into blocks.

Suppose:

```text
block size = 16 tokens
window = 4096 tokens
```

Then:

```text
4096 / 16 = 256 blocks
```

Instead of thinking:

```text
delete token 1
delete token 2
delete token 3
...
```

we can think:

```text
block 0
block 1
block 2
...
block 255
```

As the window advances sufficiently, entire blocks can be released.

For example:

```text
Before:

[0-15] [16-31] [32-47] ... [4080-4095]
```

After enough new tokens:

```text
        [16-31] [32-47] ... [4096-4111]
```

Block `[0-15]` can be returned to the allocator.

This is much more efficient than physically moving every KV tensor.

---

# 10. The allocator view

Imagine a GPU KV block pool:

```text
GPU KV memory

[Block 0]
[Block 1]
[Block 2]
[Block 3]
[Block 4]
[Block 5]
...
[Block 999]
```

Request A owns:

```text
0 1 2 3
```

Request B owns:

```text
8 9 10
```

When A's sliding window moves and block 0 is no longer needed:

```text
Block 0 → FREE
```

The allocator can immediately give block 0 to another request.

This is one of the major advantages of combining:

```text
PagedAttention
+
sliding-window eviction
```

The memory manager doesn't need to move the remaining blocks around.

---

# 11. Ring-buffer implementation

Another common conceptual implementation is a **ring buffer**.

Imagine the cache has 8 slots:

```text
[0][1][2][3][4][5][6][7]
```

Initially:

```text
A B C D E F G H
```

When the next token arrives, instead of allocating more memory, we overwrite the oldest position:

```text
I B C D E F G H
```

Then:

```text
I J C D E F G H
```

Then:

```text
I J K D E F G H
```

The physical storage remains fixed.

Conceptually:

```text
write_position = current_position % window_size
```

So if:

```text
window_size = 4096
```

the physical slot for token N can be determined by:

```text
slot = N % 4096
```

This is extremely memory efficient.

However, production GPU implementations often combine block-based allocation and more sophisticated attention kernels rather than literally maintaining a simple Python-style circular array.

---

# 12. Sliding window with GQA

Now connect this with our previous discussion.

We already learned:

```text
MHA → many KV heads
GQA → fewer KV heads
MQA → one KV head
```

Sliding windows attack memory in a different dimension.

GQA reduces:

```text
KV bytes per token
```

Sliding windows reduce:

```text
number of tokens retained
```

So they multiply together.

Suppose:

```text
32 layers
8 KV heads
128 head dimension
BF16
```

We calculated:

```text
128 KiB/token
```

Now use:

```text
window = 4096
```

Memory:

```text
4096 × 128 KiB
= 512 MiB
```

If the same architecture used MHA with 32 KV heads, the KV cache would be four times larger:

```text
512 KiB/token
```

Therefore:

```text
4096 × 512 KiB
= 2 GiB
```

So:

```text
GQA + sliding window
```

is substantially more memory efficient than:

```text
MHA + full context
```

This illustrates something important about inference optimization: **different techniques attack different terms in the memory equation.**

---

# 13. What happens to a long conversation?

Suppose you have a chatbot with:

```text
window = 4096
```

The conversation reaches:

```text
100,000 tokens
```

The model does not necessarily keep:

```text
100,000 tokens × KV bytes
```

Instead, it maintains approximately:

```text
4096 tokens × KV bytes
```

while the logical conversation may continue to contain 100K tokens.

So there are really two different things:

```text
Conversation history
        ≠
active transformer attention state
```

The application can store the complete conversation externally:

```text
Database
Object storage
Conversation store
RAG index
```

while the transformer only maintains the currently active attention window.

This distinction becomes extremely important for production LLM systems.

---

# 14. How do we remember information outside the window?

This is where a simple sliding window alone is insufficient.

Suppose:

```text
Conversation:

User:
My project is called VIDRAG.

... 20,000 tokens later ...

User:
What is the name of my project?
```

If the original statement has disappeared from the attention window, the model cannot directly attend to that token.

A production system can therefore combine sliding-window attention with **external memory**.

Conceptually:

```text
Long conversation
       |
       +------------------+
       |                  |
recent tokens        historical data
       |                  |
       ↓                  ↓
KV cache              storage/RAG
       |                  |
       +--------+---------+
                |
                ↓
             context
                |
                ↓
              model
```

The recent conversation remains in the KV cache.

Older information can be retrieved only when needed.

This transforms an unlimited conversation into a bounded active working set.

---

# 15. Sliding window plus summarization

Another approach is summarization.

Suppose the original history is:

```text
100,000 tokens
```

The system could periodically compress older conversation into something like:

```text
Conversation summary:
The user is building VIDRAG, a multimodal video RAG system using
CLIP embeddings, LanceDB and GPT-based generation.
```

Then the model sees:

```text
SYSTEM PROMPT

LONG-TERM SUMMARY

RECENT 4096 TOKENS

CURRENT USER MESSAGE
```

The KV cache only needs to hold the active context.

But there is an important caveat: summarization is **lossy**.

The original 20,000 tokens cannot generally be reconstructed exactly from a 500-token summary.

So this becomes a semantic memory system rather than exact attention preservation.

---

# 16. Sliding window does not mean "the model forgets everything"

There is an important distinction between:

```text
not being directly accessible through attention
```

and:

```text
information no longer existing anywhere
```

When KV entries are evicted:

```text
K_old
V_old
```

are removed from the active GPU cache.

That does not necessarily mean the application deleted the conversation.

The application may still have:

```text
raw conversation
database record
summary
embedding
document store
```

So a useful architecture is:

```text
                    ┌──────────────────┐
                    │ Long-term memory │
                    │ DB / RAG / Store │
                    └────────┬─────────┘
                             │
                             │ retrieve
                             ↓
User → recent context → KV cache → Transformer
                             ↑
                             │
                       sliding window
```

---

# 17. Hard eviction versus intelligent eviction

A simple sliding window uses a very straightforward rule:

```text
keep the newest W tokens
evict everything older
```

But this is not necessarily optimal.

Imagine:

```text
Token 100:
The user's name is Alice.

Token 101-9000:
lots of conversation

Token 9001:
What is my name?
```

The token containing "Alice" may be old but extremely important.

A pure recency policy would have evicted it.

This motivates more sophisticated **KV-cache eviction policies**.

Instead of:

```text
oldest → evict
```

we might consider:

```text
importance
attention frequency
recency
token type
layer
position
semantic relevance
```

The cache manager then becomes more like a memory-management policy.

---

# 18. Recency versus importance

A simple eviction policy is:

```text
LRU
```

meaning:

```text
Least Recently Used
```

The intuition is:

> If something hasn't been accessed recently, it may be less useful.

For transformer KV caches, however, the situation is more complicated because attention isn't a normal CPU cache access pattern.

The model's future queries determine which historical tokens are useful.

A token can be:

```text
old
but highly important
```

or:

```text
recent
but completely irrelevant
```

Therefore, sophisticated KV eviction research often considers attention-based importance rather than pure chronological order.

---

# 19. The attention-score intuition

Suppose the current query produces attention scores approximately like:

```text
Token A → 0.01
Token B → 0.02
Token C → 0.60
Token D → 0.01
Token E → 0.36
```

Tokens C and E are contributing much more to the current attention computation.

This suggests a possible heuristic:

```text
high attention importance → retain
low attention importance  → candidate for eviction
```

But there is a major complication.

The importance of a token can change as the conversation progresses.

A token that looks unimportant now may become important for a future query.

So eviction policies must balance:

```text
past importance
+
recency
+
future uncertainty
```

This is why intelligent KV eviction is much harder than ordinary cache eviction.

---

# 20. Why some tokens are disproportionately important

Long-context transformer research has observed phenomena such as **attention sinks**, where certain early tokens can receive substantial attention even when they do not carry obvious semantic information.

This creates an interesting problem.

Suppose we simply keep:

```text
last 4096 tokens
```

but the model's behavior benefits from retaining a few special early tokens.

Then pure sliding-window eviction can hurt model quality.

This has motivated architectures and techniques that preserve a small number of important tokens while sliding the rest.

Conceptually:

```text
[important initial tokens]
+
[recent sliding window]
```

instead of:

```text
[recent sliding window only]
```

You can think of this as:

```text
STATIC MEMORY
      +
RECENT MEMORY
```

rather than a single contiguous window.

---

# 21. Streaming-style attention

A particularly useful mental model is:

```text
               permanent / sink tokens
                         |
                         ↓
[ S S S ][................recent window................]
          ↑
       old tokens
       continuously evicted
```

Suppose:

```text
sink tokens = 4
window = 4092
```

Then the active cache might contain:

```text
4 persistent tokens
+
4092 most recent tokens
```

giving approximately:

```text
4096 active tokens
```

The old middle portion is discarded as the sequence progresses.

This allows the model to operate on arbitrarily long streams with bounded memory, subject to the quality characteristics of the architecture and eviction strategy.

---

# 22. Why this matters for inference servers

Now connect this to serving.

Imagine a GPU with:

```text
80 GB VRAM
```

and you are serving many requests.

Suppose each request without bounded caching consumes:

```text
8 GB KV
```

Then even before considering weights, activations, CUDA workspace, and other overheads, only a small number of requests can fit.

With a bounded window consuming:

```text
512 MB per request
```

you can support a much larger active set.

This directly affects:

```text
concurrency
throughput
GPU utilization
queueing
latency
cost per request
```

So sliding-window KV management is not merely a theoretical optimization.

It changes the economics of serving long-context workloads.

---

# 23. Interaction with PagedAttention

The concepts now fit together very naturally.

We have:

```text
GQA
```

which reduces the amount of KV data generated per token.

Then:

```text
KV cache
```

stores that data so it doesn't need to be recomputed.

Then:

```text
PagedAttention
```

organizes that cache into fixed-size physical blocks.

Then:

```text
Sliding window
```

defines how much recent history needs to remain active.

Then:

```text
KV eviction
```

releases blocks that are no longer required.

The architecture looks like:

```text
                 Transformer
                      |
                      ↓
                  KV Cache
                      |
                      ↓
               PagedAttention
                      |
              ┌───────┴───────┐
              ↓               ↓
        active blocks     old blocks
              |               |
              ↓               ↓
          keep/reuse         evict
              |
              ↓
       sliding attention
```

And if prefix caching is also enabled:

```text
             Prefix Cache
                  |
                  ↓
         reusable KV blocks
                  |
                  ↓
           PagedAttention
                  |
         ┌────────┴────────┐
         ↓                 ↓
   active window      evictable blocks
         |
         ↓
     Transformer
```

---

# 24. The memory equation with a sliding window

Without a window:

```text
KV memory
=
sequence_length
×
layers
×
KV heads
×
head dimension
×
2
×
bytes per element
```

With a maximum window W:

```text
KV memory
≈
W
×
layers
×
KV heads
×
head dimension
×
2
×
bytes per element
```

The critical difference is:

```text
sequence_length
```

becomes:

```text
window_size
```

once the sequence exceeds the window.

For our example:

```text
layers = 32
KV heads = 8
head dimension = 128
BF16 = 2 bytes
window = 4096
```

we have:

```text
2 × 32 × 8 × 128 × 2 × 4096
```

which gives:

```text
536,870,912 bytes
```

or:

```text
512 MiB
```

Whether the actual runtime allocation is exactly this depends on block size, padding, metadata, implementation overhead, and other runtime requirements. The number is the raw KV payload.

---

# 25. Fragmentation comes back into the picture

Remember our PagedAttention discussion.

Suppose:

```text
block size = 16 tokens
window = 4096 tokens
```

Then:

```text
4096 / 16 = 256 blocks
```

This is convenient because the window is exactly divisible by the block size.

But suppose:

```text
window = 4000
block size = 16
```

Then:

```text
4000 / 16 = 250 blocks exactly
```

Still fine.

Now suppose:

```text
window = 4095
```

Then:

```text
ceil(4095 / 16)
= 256 blocks
```

Allocated token capacity:

```text
256 × 16
= 4096
```

So:

```text
4096 - 4095
= 1 token
```

of internal unused capacity.

The same block-fragmentation principles we discussed earlier still apply.

---

# 26. The really important production distinction

There are actually three different memory-management questions:

```text
1. What information should the model be allowed to attend to?

2. Which KV entries are currently required?

3. Where should those KV entries physically live?
```

They correspond roughly to:

```text
Attention pattern
        ↓
Sliding window / eviction policy
        ↓
PagedAttention / block allocator
```

And this is why these topics should not be treated as isolated tricks.

They are layers of one inference-memory system.

---

# 27. A concrete 128K conversation

Let's make the entire thing concrete.

Suppose:

```text
conversation = 128K tokens
window = 4096
model KV cost = 128 KiB/token
```

Without sliding-window retention:

```text
128K × 128 KiB
≈ 16 GiB
```

With a 4096-token active window:

```text
4096 × 128 KiB
= 512 MiB
```

So the raw KV payload changes from approximately:

```text
16 GiB
```

to:

```text
512 MiB
```

for that active sequence.

That is a:

```text
32× reduction
```

in active KV payload.

But there is a crucial price:

> The transformer no longer has direct attention access to the entire 128K history.

This is the central tradeoff.

---

# 28. Sliding window versus full-context attention

The decision can therefore be viewed like this:

| Property                |       Full KV cache |                      Sliding window |
| ----------------------- | ------------------: | ----------------------------------: |
| KV memory               | grows with sequence |                             bounded |
| Long-range attention    |           available |                          restricted |
| Very long conversations |           expensive |                        much cheaper |
| Memory predictability   |                poor |                              strong |
| Historical access       |              direct | requires external memory if evicted |
| Implementation          |     straightforward |      requires eviction/window logic |
| Concurrency             |               lower |                              higher |

Neither mechanism is universally appropriate. The correct choice depends on whether the application genuinely needs exact attention over the entire history.

---

# 29. The deeper architecture: working memory versus long-term memory

This is probably the most useful way to think about the entire topic.

A modern LLM application can be designed similarly to a computer system.

The GPU KV cache behaves like:

```text
working memory
```

It contains information immediately available to the transformer.

The database, vector store, conversation store, or summaries behave like:

```text
long-term memory
```

Then retrieval brings relevant historical information back into the model's active context.

So:

```text
                    Long-term memory
                 ┌────────────────────┐
                 │ Conversation DB    │
                 │ Vector DB          │
                 │ Summaries          │
                 │ Documents          │
                 └─────────┬──────────┘
                           │
                       retrieval
                           │
                           ↓
User → Recent Context → KV Cache → Transformer
                           ↑
                           │
                     Sliding Window
                           │
                           ↓
                     KV Eviction
```

This is conceptually very close to how a real memory hierarchy works.

---

# 30. The inference-engineering hierarchy so far

You can now see the sequence we have been building:

```text
Transformer
    ↓
Attention
    ↓
Q / K / V
    ↓
KV Cache
    ↓
MHA / MQA / GQA
    ↓
KV memory reduction
    ↓
PagedAttention
    ↓
Block-based KV allocation
    ↓
Prefix Caching
    ↓
Reuse KV across requests
    ↓
Sliding Window
    ↓
Bound active KV memory
    ↓
KV Eviction
    ↓
Release blocks that are no longer needed
```

And the next level is:

```text
Continuous Batching
```

because once you have many requests whose KV caches are growing, shrinking, sharing, and being evicted, you need a scheduler that decides **which requests receive GPU computation at each decoding iteration**.

That is where inference serving becomes much more interesting: the problem stops being merely "run a transformer" and becomes **GPU memory management + scheduling + attention computation + request lifecycle management**.
