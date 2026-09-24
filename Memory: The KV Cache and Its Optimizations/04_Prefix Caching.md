# Prefix Caching: From First Principles to Chained Hashing, Block Matching, and When It Actually Helps

Prefix caching is one of those LLM inference optimizations that sounds simple:

> “If two requests have the same beginning, why compute the beginning twice?”

That is exactly the idea.

Suppose thousands of users interact with an enterprise chatbot. Every request begins with the same 8,000-token system prompt containing company policies, tool instructions, safety rules, and documentation. Only the final user question changes.

Without prefix caching, the inference server performs the expensive **prefill computation** for those same 8,000 tokens again and again.

With prefix caching, the server computes the Key/Value state for that shared prefix once, stores it, and reuses it for subsequent requests.

The important part is that prefix caching is not merely “cache the prompt string.” A production implementation needs to determine whether two token sequences have exactly the same prefix, identify the corresponding KV blocks, manage those blocks safely, and evict them when GPU memory is needed.

This is where the **chained-hash matching mechanism** becomes important.

---

# 1. Start with the problem

Imagine this system prompt:

```text id="2ayq7v"
You are an enterprise financial assistant.

Follow these company policies...

Use these tools...

Here is the company's financial policy...

Here is the product documentation...

Here are examples of how to answer...
```

Suppose this consumes:

```text id="1h9vce"
8,000 tokens
```

Now three users ask different questions:

```text id="x7p3ab"
Request A:
[8,000-token shared prefix]
What is our refund policy?


Request B:
[8,000-token shared prefix]
Can I cancel my subscription?


Request C:
[8,000-token shared prefix]
How long does settlement take?
```

The first 8,000 tokens are identical.

But a normal inference pipeline might process:

```text id="c2xq0v"
A → prefill 8,000 tokens
B → prefill 8,000 tokens
C → prefill 8,000 tokens
```

That's:

```text id="7kg7g3"
24,000 tokens of prefix computation
```

even though only:

```text id="9c4v5q"
8,000 tokens
```

are unique.

Prefix caching tries to turn this into:

```text id="j5z2k1"
Compute shared 8K prefix once

             ↓

       Cached KV blocks

        ↙     ↓     ↘

       A      B      C
       ↓      ↓      ↓
   question question question
```

This can dramatically reduce **prefill computation**.

---

# 2. First understand prefill

Prefix caching makes sense only if you understand the difference between **prefill** and **decode**.

Suppose the request is:

```text id="fj2m0n"
System prompt
+
User question
```

The model first processes the input tokens.

This is the **prefill phase**.

Conceptually:

```text id="yrm4ht"
Input tokens
     ↓
Transformer
     ↓
K/V generated for every layer
     ↓
KV cache
```

Then the model starts generating output tokens.

That's the **decode phase**.

So:

```text id="m6i8s0"
Prefill:
process existing input

Decode:
generate new tokens
```

Prefix caching primarily accelerates the repeated **prefill** portion.

---

# 3. Why KV cache makes prefix caching possible

Remember what we learned about the KV cache.

For every processed token, the transformer generates:

```text id="xg8g1q"
K
V
```

and stores them.

Suppose the shared prefix is:

```text id="1s3w7e"
Token 0
Token 1
Token 2
...
Token 7999
```

After processing it, we have:

```text id="d4x8v6"
K/V for token 0
K/V for token 1
...
K/V for token 7999
```

If another request begins with exactly the same token sequence, we don't necessarily need to recompute those K/V representations.

We can reuse the existing KV state.

That's prefix caching.

---

# 4. But how do we know two prefixes are identical?

This is where the interesting part begins.

Imagine a cache containing thousands or millions of KV blocks.

When a new request arrives, we need to answer:

> “Does the beginning of this token sequence already exist in the cache?”

We could compare the entire token sequence against every cached sequence.

That would obviously be inefficient.

Instead, production systems can associate a hash with each KV block.

The basic idea is:

```text id="4x6j80"
Tokens
   ↓
Hash
   ↓
Cache lookup
```

But there is a subtle issue.

A block's identity depends not only on the tokens inside the block but also on what came before it.

That leads to **chained hashing**.

---

# 5. Why a simple block hash isn't always enough

Suppose our block size is:

```text id="7z3m1p"
4 tokens
```

and we have:

```text id="6q6t2e"
Block 0:
A B C D

Block 1:
E F G H
```

If we independently hash each block:

```text id="j0c3py"
hash(A B C D)

hash(E F G H)
```

then Block 1 doesn't explicitly encode that it followed:

```text id="5blp70"
A B C D
```

Now imagine another sequence:

```text id="qkvw1p"
X Y Z W
E F G H
```

The second block is still:

```text id="5ev0cx"
E F G H
```

So an independent block hash would say:

> “The second block is identical.”

But the KV representation associated with `E F G H` depends on its position and preceding context.

Therefore, we need the identity of a block to depend on the prefix before it.

This is where chained hashing comes in.

---

# 6. Chained hashing

Conceptually, define:

```text id="x1o8e8"
H0 = hash(initial_seed, block_0_tokens)

H1 = hash(H0, block_1_tokens)

H2 = hash(H1, block_2_tokens)

H3 = hash(H2, block_3_tokens)
```

So:

```text id="g5m7ri"
Block 0
   ↓
H0
   ↓
Block 1
   ↓
H1
   ↓
Block 2
   ↓
H2
```

Each block's hash incorporates the hash of the previous block.

Therefore, the hash of Block 2 indirectly depends on:

```text id="rj4y29"
Block 0
+
Block 1
+
Block 2
```

This gives us a compact representation of the entire prefix history.

---

# 7. A simple example

Suppose the token sequence is:

```text id="j8g0m9"
A B C D | E F G H | I J K L
```

with:

```text id="9x8r2c"
4 tokens/block
```

We calculate:

```text id="j8x4bh"
H0 = hash(seed, A B C D)

H1 = hash(H0, E F G H)

H2 = hash(H1, I J K L)
```

Now another request arrives:

```text id="0c4dy2"
A B C D | E F G H | I J X Y
```

The first block gives:

```text id="rb0yqy"
H0'
```

Since:

```text id="h1sk3c"
A B C D
```

is identical, ideally:

```text id="f54w7b"
H0' = H0
```

Then the second block:

```text id="c4sl9x"
E F G H
```

also matches.

Therefore:

```text id="v4lv2r"
H1' = H1
```

But the third block differs:

```text id="yjw4ck"
I J X Y
```

instead of:

```text id="3r5vcm"
I J K L
```

Therefore:

```text id="b9c2fg"
H2' ≠ H2
```

The cache can reuse:

```text id="fj6q7x"
Block 0
Block 1
```

but not Block 2.

This is exactly the behavior we want.

---

# 8. Why the chain is useful

The chained hash gives us a natural prefix boundary.

Suppose:

```text id="z9h2ai"
Request A:

Block 0
Block 1
Block 2
Block 3
```

and:

```text id="ak0n5f"
Request B:

Block 0
Block 1
Block 2
Block 9
```

The cache can discover:

```text id="1s8o9u"
Block 0 → match
Block 1 → match
Block 2 → match
Block 3 → mismatch
```

and stop.

Therefore:

```text id="a0z1gi"
Longest cached prefix = Blocks 0–2
```

This is much more efficient than comparing the entire prompt character by character.

---

# 9. What exactly gets cached?

This is an important distinction.

We are not necessarily caching:

```text id="8w8qv2"
"the text string"
```

as the primary computational artifact.

We want to cache the already-computed:

```text id="9x8a9k"
KV blocks
```

For example:

```text id="0f5q0x"
Hash H0 → physical KV block 100
Hash H1 → physical KV block 101
Hash H2 → physical KV block 102
```

So the cache might conceptually look like:

```text id="e4f9pb"
Prefix hash
      ↓
Physical KV block
```

The hash provides identity.

The KV block provides the reusable computation.

---

# 10. Combining prefix caching with PagedAttention

Now the previous topic becomes extremely useful.

PagedAttention gives us:

```text id="c7m8r9"
physical KV blocks
+
logical-to-physical block tables
```

Prefix caching can use those same physical blocks.

Suppose:

```text id="n8c2v7"
Shared prefix
=
3 blocks
```

The prefix cache might contain:

```text id="8v4m2x"
H0 → physical block 50
H1 → physical block 72
H2 → physical block 18
```

A new request arrives.

The server computes the same chained hashes:

```text id="1v4x7n"
H0
H1
H2
```

and discovers:

```text id="7y2m6b"
H0 → block 50
H1 → block 72
H2 → block 18
```

So instead of running the transformer over those tokens again, the request can reference those existing KV blocks.

This is where the combination becomes powerful:

```text id="c4a5n8"
Prefix caching
       +
PagedAttention
       ↓
Reusable physical KV blocks
```

---

# 11. The reference-counting problem

There is an important memory-management issue.

Suppose:

```text id="m6v1q2"
Request A
```

is currently using:

```text id="b9c4k7"
Block 100
```

Now Request B arrives with the same prefix.

Both requests want:

```text id="c7d2s4"
Block 100
```

We cannot free Block 100 when Request A finishes because Request B still needs it.

Therefore the block can maintain a reference count.

Conceptually:

```text id="i5n3v8"
Block 100

refcount = 2
```

because:

```text id="u6z0q1"
A → Block 100
B → Block 100
```

When A finishes:

```text id="l2x9c4"
refcount = 1
```

When B finishes:

```text id="p8w3n7"
refcount = 0
```

Now the block can potentially be evicted or returned to the free pool.

This allows multiple requests to share the same cached prefix safely.

---

# 12. Prefix caching is basically computation memoization

There is a useful computer-science analogy.

Suppose you have:

```text id="a2p7x8"
f(x)
```

and computing `f(x)` is expensive.

If you calculate:

```text id="6v4y2k"
f(x) = result
```

you can store the result.

Next time someone asks for:

```text id="0k9m3n"
f(x)
```

you simply return the cached result.

That's **memoization**.

Prefix caching is essentially memoization for part of the transformer's inference state.

Instead of:

```text id="q3f8z1"
input prefix
    ↓
expensive transformer computation
    ↓
KV state
```

every time, we do:

```text id="r8c2m7"
input prefix
    ↓
cache lookup
    ↓
existing KV state
```

when possible.

---

# 13. The key requirement: exact token equality

This is one of the most important practical details.

Prefix caching generally requires the prefix to match at the token level.

These strings may look nearly identical:

```text id="8q2s5w"
"Explain RAG."

"Explain RAG. "
```

But the trailing space can change tokenization.

Similarly:

```text id="3p6n9v"
"Hello, world"

"Hello,  world"
```

may tokenize differently.

Therefore prefix caching cannot safely rely on simple visual string similarity.

The matching process is fundamentally about the actual token sequence and the resulting prefix identity.

---

# 14. Tokenization matters

Suppose:

```text id="r1f7c8"
Prompt A:
"Hello world"
```

tokenizes to:

```text id="m3x7q1"
[15496, 995]
```

while:

```text id="k5v2s9"
Prompt B:
"Hello  world"
```

might tokenize differently.

Even though a human sees:

```text id="b7q3p2"
Hello world
```

and:

```text id="v6x1r9"
Hello  world
```

as nearly identical, the model doesn't necessarily see them as the same token sequence.

Therefore:

```text id="9c2y6w"
Different tokens
        ↓
Different prefix
        ↓
Different KV state
        ↓
Cannot safely reuse the cached KV state
```

This is why canonical prompt construction is important when you want prefix caching to work reliably.

---

# 15. System prompts are an ideal use case

Consider a production AI assistant.

Every request begins:

```text id="m3a7y1"
You are an enterprise assistant.

Company policy:
...

Security policy:
...

Available tools:
...

Response format:
...

Business glossary:
...
```

Suppose that's:

```text id="z9w3v2"
12,000 tokens
```

Then every user request adds:

```text id="n2x6p8"
"What is the status of invoice 123?"
```

or:

```text id="p4c8k1"
"Explain our refund policy."
```

The shared prefix might be:

```text id="g7v3s5"
12,000 tokens
```

If 1,000 requests arrive using the same prefix, prefix caching can eliminate repeated prefill computation for that shared portion after the prefix has been cached.

The exact speedup depends on hardware, prompt structure, cache hit rate, and the relative cost of prefill versus decode.

But the computational opportunity is enormous.

---

# 16. Few-shot prompting is another strong use case

Suppose you use a classifier prompt like:

```text id="e1f4g7"
System instructions

Example 1
Input: ...
Output: ...

Example 2
Input: ...
Output: ...

Example 3
Input: ...
Output: ...

Example 4
Input: ...
Output: ...

Now classify:
<user input>
```

The examples might consume:

```text id="5w7z2k"
5,000 tokens
```

while the final user input is only:

```text id="0m3p8q"
100 tokens
```

If thousands of requests use exactly the same few-shot template, prefix caching is highly beneficial.

The architecture becomes:

```text id="b8z3c1"
Shared few-shot examples
        ↓
Cached KV blocks
        ↓
New user input
        ↓
Only new suffix requires prefill
```

This can significantly reduce repeated computation.

---

# 17. Tool-use agents are another interesting case

Agent systems often use large fixed instructions.

For example:

```text id="g5r7c2"
System prompt
+
Tool definitions
+
Tool schemas
+
Safety instructions
+
Agent policy
+
Examples
+
Conversation
```

The tool definitions might be thousands of tokens.

If every request uses the same tools, the prefix is highly reusable.

For example:

```text id="p2n6v4"
Request A:
[system + tools + examples] + user task A

Request B:
[system + tools + examples] + user task B

Request C:
[system + tools + examples] + user task C
```

The prefix:

```text id="z7x1c3"
system + tools + examples
```

can potentially be cached.

This is particularly relevant for production agent infrastructure.

---

# 18. When prefix caching does NOT help much

Now the important part:

> Prefix caching is not universally useful.

Suppose every request starts with a completely different document:

```text id="n3y7f1"
Request A → unique 20K document
Request B → completely different 20K document
Request C → completely different 20K document
```

There may be almost no shared prefix.

Then:

```text id="x8v2m4"
cache hit rate ≈ low
```

and the prefix cache provides little benefit.

You may consume substantial memory storing KV blocks that are rarely reused.

So prefix caching works best when:

```text id="g6q1z9"
same prefix
+
many requests
+
expensive prefix
```

---

# 19. Prefix caching is not semantic caching

This distinction is extremely important.

Suppose two prompts mean the same thing:

```text id="j3k7p9"
"Explain the refund policy."

"Can you tell me about our refund rules?"
```

These are semantically similar.

But they are different token sequences.

Prefix caching generally cannot say:

> "These prompts mean roughly the same thing, so let's reuse the same KV cache."

That would be unsafe because the KV representation depends on the actual token sequence.

So:

```text id="q6w2e8"
Semantic similarity
        ≠
Token identity
```

Prefix caching is deterministic and exact.

Semantic caching is a different optimization.

---

# 20. Prefix caching versus response caching

These are also completely different.

Suppose someone asks:

```text id="v3r8x1"
"What is 2 + 2?"
```

A response cache might store:

```text id="s7m2p4"
"What is 2 + 2?"
→ "4"
```

and return the answer directly.

Prefix caching does not cache the final answer.

It caches intermediate transformer state:

```text id="n5x8c2"
Prompt prefix
      ↓
KV state
```

Then the model still processes the new suffix and generates a fresh answer.

So:

```text id="y2p6k8"
Response caching
→ avoids model generation entirely

Prefix caching
→ avoids recomputing a shared prefix
```

This distinction matters enormously when designing inference systems.

---

# 21. Prefix caching versus embedding caching

Embedding caching stores something like:

```text id="k7m1x4"
text
 ↓
embedding vector
```

Prefix caching stores:

```text id="q2v8p5"
token prefix
 ↓
layer-wise K/V state
```

These serve completely different purposes.

Embedding caches are commonly used in:

```text id="m4z6r8"
RAG
semantic search
deduplication
retrieval
```

Prefix caches are used in:

```text id="b9x3k2"
LLM inference
repeated prompt prefixes
agent systems
few-shot templates
shared system prompts
```

---

# 22. Why the hash must include the right information

A production prefix cache cannot simply hash:

```text id="w2r7y9"
block tokens
```

and assume that is universally sufficient.

The identity of a KV block can depend on the model execution context.

Conceptually, the cache key may need to account for things such as:

```text id="v8q1c5"
previous prefix identity
+
current block tokens
+
relevant model/context configuration
```

The exact implementation depends on the serving engine.

The key principle is:

> A cache hit must imply that the cached KV state is valid for the new request.

If two requests produce different KV states but accidentally map to the same cache entry, the result can be incorrect.

Therefore hash design and cache-key correctness are critical.

---

# 23. Why chained hashing naturally finds the longest prefix

Suppose the cached sequence is:

```text id="z5m3k1"
A B C D | E F G H | I J K L | M N O P
```

and a new request is:

```text id="h8q2v7"
A B C D | E F G H | I J K L | X Y Z W
```

The chain becomes:

```text id="r4t6p8"
H0 = hash(seed, A B C D)

H1 = hash(H0, E F G H)

H2 = hash(H1, I J K L)

H3 = hash(H2, X Y Z W)
```

The cached chain has:

```text id="j2f5x9"
H0
H1
H2
H3_cached
```

The new request produces:

```text id="b7c1m6"
H0
H1
H2
H3_new
```

Therefore:

```text id="f8r2k5"
H0 matches
H1 matches
H2 matches
H3 differs
```

So the longest reusable prefix is exactly:

```text id="m5z8q1"
3 blocks
```

This gives the inference engine a very clean stopping point.

---

# 24. Why this works especially well with fixed-size blocks

Suppose:

```text id="b8x4n2"
block size = 16 tokens
```

Then:

```text id="1y5q7z"
Tokens 0–15   → Block 0
Tokens 16–31  → Block 1
Tokens 32–47  → Block 2
Tokens 48–63  → Block 3
```

The cache can match complete blocks.

If a request has:

```text id="q3m6v9"
50 tokens
```

the first:

```text id="g5n1p7"
48 tokens
```

can correspond to:

```text id="r6c2x8"
3 complete blocks
```

while the final two tokens form a partial block.

Therefore, prefix caching often operates most naturally on complete blocks.

This is another reason PagedAttention and prefix caching fit together so well.

---

# 25. A complete prefix-cache lookup

Let's walk through what could happen when a new request arrives.

Suppose:

```text id="m8q3x5"
block size = 16 tokens
```

The request has:

```text id="r7n2c4"
64 tokens
```

Therefore:

```text id="g1x8p6"
4 blocks
```

The serving system computes:

```text id="h3m7q2"
H0
H1
H2
H3
```

It looks up:

```text id="v5c8z1"
H0 → ?
H1 → ?
H2 → ?
H3 → ?
```

Suppose the cache contains:

```text id="b2x6k9"
H0 → Block 101
H1 → Block 203
H2 → Block 417
H3 → MISS
```

Then:

```text id="f6p1y4"
Blocks 0–2 can be reused.
Block 3 must be computed.
```

The request's block table becomes something like:

```text id="a4n8z6"
Logical block 0 → Physical block 101
Logical block 1 → Physical block 203
Logical block 2 → Physical block 417
Logical block 3 → New physical block
```

Only the unmatched suffix needs new KV computation.

---

# 26. What happens after the new block is computed?

Suppose the model computes Block 3.

The system now has:

```text id="q7x2m5"
H3
```

and can insert:

```text id="w8c1n4"
H3 → physical block 512
```

into the prefix cache.

Now another request arrives with exactly the same 64-token prefix.

The lookup becomes:

```text id="e6p3r7"
H0 → hit
H1 → hit
H2 → hit
H3 → hit
```

The entire 64-token prefix can potentially be reused.

This means the value of a prefix cache increases as the same prefixes recur.

---

# 27. Cache hit rate is the critical metric

Prefix caching isn't useful simply because you have a cache.

You need **cache hits**.

A useful conceptual metric is:

```text id="r4v8z2"
prefix_cache_hit_rate
=
requests with reusable prefix
/
total requests
```

But an even more useful metric for inference cost is the fraction of tokens reused:

```text id="h7m3q9"
reused_prefix_tokens
/
total input tokens
```

For example, suppose:

```text id="s5x2k8"
100 requests
10,000 input tokens each
```

Total input tokens:

```text id="j8c4p1"
1,000,000
```

If every request shares the first:

```text id="n6v2z9"
8,000 tokens
```

then potentially:

```text id="q1m7x3"
800,000 tokens
```

belong to reusable prefixes.

That's an enormous amount of repeated prefill computation.

---

# 28. A simple compute-saving estimate

Suppose:

```text id="y8c2m5"
100 requests
```

and each request has:

```text id="r4x7n1"
8,000-token shared prefix
+
500-token unique suffix
```

Without prefix caching:

```text id="d6p2q8"
100 × 8,500
=
850,000 input tokens processed
```

With prefix caching:

```text id="z3m8k1"
8,000 shared tokens
+
100 × 500 unique tokens
```

which is:

```text id="n7c4v6"
8,000 + 50,000
=
58,000 tokens
```

of unique prefill computation, conceptually.

So the repeated prefix work drops from:

```text id="f5q2x9"
800,000 tokens
```

to approximately:

```text id="b8m4z1"
8,000 tokens
```

assuming all requests truly share the prefix and the cache remains available.

That is the basic economic argument for prefix caching.

The actual wall-clock improvement will not equal the token reduction one-for-one because transformer throughput, scheduling, memory bandwidth, cache lookup overhead, and batching all matter.

---

# 29. Shared system prompt example

Consider an enterprise chatbot with:

```text id="6x3m8q"
System prompt = 6,000 tokens
```

and:

```text id="2p7v4n"
User question = 100 tokens
```

Suppose 10,000 users send requests.

Without caching:

```text id="9c5m1x"
10,000 × 6,000
=
60 million
```

system-prompt tokens must be prefetched.

With an effective prefix cache:

```text id="f2q8m7"
6,000 tokens
```

of shared prefix computation may be performed once and then reused, subject to cache residency and the serving architecture.

The potential reduction in repeated work is enormous.

This is why long, static system prompts are one of the strongest use cases.

---

# 30. Few-shot template example

Suppose you build a sentiment classifier using a prompt:

```text id="e4q8m2"
You are a sentiment classifier.

Example 1:
Text: ...
Label: ...

Example 2:
Text: ...
Label: ...

Example 3:
Text: ...
Label: ...

Example 4:
Text: ...
Label: ...

Classify the following:
```

Assume the template is:

```text id="g2n7v5"
4,000 tokens
```

and each incoming example is:

```text id="m5x1c9"
100 tokens
```

If you process:

```text id="h8q3z6"
100,000 requests
```

the same 4,000-token prefix appears repeatedly.

This is exactly the sort of workload where prefix caching can provide significant savings.

---

# 31. Where prefix caching may not help

Suppose you have a RAG application where every request looks like:

```text id="p3n7x2"
System prompt
+
retrieved document A
+
retrieved document B
+
retrieved document C
+
user query
```

If the retrieved documents change every time, the prefix might be:

```text id="q6m1v8"
System prompt
```

followed immediately by:

```text id="x4c9z5"
different retrieval context
```

Then the reusable prefix might only be a few hundred tokens.

In that case, the cache may not provide much value relative to its memory cost.

This is why prefix caching and RAG interact in interesting ways.

---

# 32. RAG can still benefit from prefix caching

However, consider a different RAG architecture:

```text id="v5q2m8"
System instructions
+
fixed tool instructions
+
fixed output schema
+
fixed few-shot examples
+
retrieved documents
+
user question
```

Everything before the retrieved documents could potentially be cached.

For example:

```text id="c7m3x9"
6,000-token static prefix
+
2,000-token dynamic retrieval
+
100-token user query
```

Then prefix caching can still save the 6,000-token prefill.

So the right question isn't:

> "Is this a RAG application?"

The right question is:

> "How much of the beginning of each request is exactly reusable?"

---

# 33. Agentic systems can be even more favorable

Agent frameworks often construct very large prompts containing:

```text id="q4x8m2"
System policy
+
tool descriptions
+
function schemas
+
agent instructions
+
examples
+
workflow state
+
history
```

If the first portion is stable across requests, prefix caching can be highly valuable.

But there is a subtle issue.

If the agent dynamically inserts information into the middle of the prompt, everything after that insertion becomes a different prefix.

For example:

```text id="n7p3c5"
Static prefix
+
dynamic state
+
static instructions
```

The second static portion is no longer a prefix.

Therefore, prompt layout matters.

---

# 34. Prompt architecture can affect cacheability

Consider:

```text id="q5m8v2"
System instructions
+
static tools
+
static examples
+
dynamic user information
+
dynamic retrieved documents
```

This is cache-friendly.

But:

```text id="z3x7c1"
System instructions
+
dynamic user information
+
static tools
+
dynamic retrieval
```

only the first section is reusable.

The static tools after the dynamic information cannot be reused as part of the same prefix.

Therefore, when designing production prompts, you can deliberately structure them so that stable information comes first.

Conceptually:

```text id="h4n8m2"
STATIC
STATIC
STATIC
STATIC
---------
DYNAMIC
DYNAMIC
DYNAMIC
```

rather than:

```text id="y7q2c6"
STATIC
DYNAMIC
STATIC
DYNAMIC
```

This is an important architectural optimization.

---

# 35. Cache locality and eviction

The prefix cache is finite.

Suppose your GPU has room for:

```text id="w6x2m8"
100,000 KV blocks
```

but your workload produces:

```text id="n3c7q1"
1 million unique prefixes
```

You cannot keep everything.

The cache therefore needs an eviction policy.

Common caching strategies can consider things like:

```text id="m8q4z6"
recency
frequency
reference count
memory cost
prefix popularity
```

A frequently reused prefix is more valuable than a prefix that was used once several hours ago.

This turns prefix caching into a classic cache-management problem.

---

# 36. LRU-style eviction

A simple strategy is:

```text id="p2x7m4"
Least Recently Used
```

The cache tracks when each cached prefix block was last accessed.

When memory is needed:

```text id="v5q1c8"
remove the least recently used blocks
```

For example:

```text id="h7m3x2"
Block A → used 1 second ago
Block B → used 5 seconds ago
Block C → used 20 minutes ago
```

Block C becomes a candidate for eviction.

Real inference systems may use more sophisticated policies, but the fundamental trade-off is the same:

```text id="j4q8z1"
GPU memory is finite.
```

---

# 37. Why cache eviction must respect shared blocks

Remember reference counting.

Suppose:

```text id="k2m7x4"
Block 100
```

is actively used by:

```text id="r8c3n6"
Request A
Request B
```

You cannot evict it just because the prefix-cache metadata says it is old.

The active requests still need it.

So there is an important distinction between:

```text id="p5q9x2"
cached and reusable
```

and:

```text id="n3m7c8"
actively referenced
```

An actively referenced block cannot simply be thrown away.

This is why a production prefix-cache manager must integrate with the active KV-cache manager.

---

# 38. Prefix cache lifecycle

A useful lifecycle looks like:

```text id="j8q3m6"
Request arrives
       ↓
Tokenize
       ↓
Divide prefix into blocks
       ↓
Compute chained hashes
       ↓
Lookup cache
       ↓
Find longest matching prefix
       ↓
Reuse matching KV blocks
       ↓
Compute unmatched suffix
       ↓
Insert newly computed blocks
       ↓
Decode
       ↓
Release references
       ↓
Keep blocks cached if useful
       ↓
Evict when memory pressure occurs
```

This is the full lifecycle.

---

# 39. What exactly does the hash chain protect against?

The chained hash gives us a way to distinguish:

```text id="p4m7x1"
A B C D | E F G H
```

from:

```text id="n6q2z8"
X Y Z W | E F G H
```

even though the second block is identical.

Because:

```text id="h3c8m5"
H1 = hash(H0, E F G H)
```

and:

```text id="y7q1v9"
H0
```

is different between the two sequences.

Therefore:

```text id="m5x2c7"
H1 ≠ H1'
```

This makes the block identity prefix-sensitive.

That is exactly what we want.

---

# 40. A more formal view of chained hashing

Let the token blocks be:

```text id="t2m7q4"
B0, B1, B2, ..., Bn
```

and let:

```text id="c6x1z8"
H(-1) = seed
```

Then:

```text id="j4p9m2"
H0 = Hash(H(-1), B0)

H1 = Hash(H0, B1)

H2 = Hash(H1, B2)

...

Hn = Hash(Hn-1, Bn)
```

Therefore:

```text id="x8q3m5"
Hn
```

implicitly commits to the entire sequence:

```text id="b2m6v9"
B0 || B1 || ... || Bn
```

where `||` means concatenation.

This gives the prefix cache a compact identity for the sequence of blocks.

---

# 41. Why not hash the entire prompt once?

You might ask:

> "Why not calculate one hash for the entire prompt?"

Because we want to reuse **partial prefixes**.

Suppose:

```text id="q7m2c8"
Request A:
A | B | C | D
```

and:

```text id="x4n9p1"
Request B:
A | B | C | E
```

If we only hash the entire prompt, then:

```text id="v3m8q5"
hash(A B C D)
```

doesn't equal:

```text id="j6c1z7"
hash(A B C E)
```

and we'd conclude:

> "No match."

But that's wrong.

The first three blocks are identical.

Chained block hashes allow us to identify:

```text id="k8p3m5"
A → match
B → match
C → match
D → mismatch
```

and reuse the longest prefix.

That's the critical reason for the block-level chain.

---

# 42. The relationship between prefix length and benefit

The benefit is roughly proportional to:

```text id="m2x7q4"
reused prefix tokens
```

but also depends on how expensive those tokens are to process.

Suppose:

```text id="y4n8c1"
Prefix = 100 tokens
Suffix = 10,000 tokens
```

Caching the prefix doesn't help much.

But:

```text id="z7m3p8"
Prefix = 10,000 tokens
Suffix = 100 tokens
```

is an excellent candidate.

Therefore:

```text id="h5q2x9"
long shared prefix
+
many repeated requests
=
high value
```

---

# 43. Prefix caching and long context

Prefix caching becomes increasingly valuable as the prefix gets longer.

Suppose a system prompt is:

```text id="p7m3x9"
500 tokens
```

The savings are modest.

If it becomes:

```text id="v2q8c4"
10,000 tokens
```

the opportunity becomes much larger.

If it becomes:

```text id="m6x1z7"
50,000 tokens
```

the potential prefill savings can be enormous.

This is one reason long-context agent systems can benefit strongly from prefix caching when their instructions and documents are reused.

---

# 44. But longer cached prefixes consume more memory

There is a trade-off.

Suppose:

```text id="g4n8m2"
prefix = 50,000 tokens
```

and our example architecture requires:

```text id="k1q7x3"
128 KiB/token
```

Then the cached KV state is approximately:

```text id="r6m2c9"
50,000 × 128 KiB
```

which is roughly:

```text id="z8p4v1"
6.25 GiB
```

of KV data.

So keeping that prefix cached consumes substantial GPU memory.

The prefix cache therefore needs to answer:

> Is this prefix reused frequently enough to justify several gigabytes of GPU memory?

This is a classic cache economics problem.

---

# 45. Cache value versus cache cost

A useful conceptual equation is:

```text id="x3m7q1"
Cache value

≈

reused computation saved
```

while:

```text id="n8c2v5"
Cache cost

≈

KV memory consumed
+
lookup overhead
+
management overhead
```

A prefix is worth caching when:

```text id="p6q1z8"
saved compute
>
memory and management cost
```

This is why a prefix used once is not particularly valuable.

A prefix used thousands of times can be extremely valuable.

---

# 46. The three strongest use cases

The first is **shared system prompts**.

For example:

```text id="q3m8x1"
Large enterprise assistant instructions
+
policies
+
tool descriptions
```

shared by thousands of requests.

The second is **few-shot templates**.

For example:

```text id="v7p2c4"
classification examples
+
format instructions
```

reused for every inference.

The third is **agent/tool templates**.

For example:

```text id="n5x8m2"
system instructions
+
tool schemas
+
agent policies
+
workflow instructions
```

shared across many sessions.

These are all cases where the beginning of the prompt remains stable while the suffix changes.

---

# 47. The cases where you should be cautious

Prefix caching is less useful when requests have:

```text id="r4m7x2"
highly unique prompts
```

or:

```text id="c8q1n5"
very short shared prefixes
```

or:

```text id="z6p3v9"
low request repetition
```

or:

```text id="m2x8q4"
the cache is constantly evicting entries before reuse
```

or:

```text id="k7n3c1"
prompt construction changes small details frequently
```

For example:

```text id="y4p8m2"
Current timestamp:
2026-09-24 20:15:31
```

embedded near the beginning of every prompt can destroy prefix reuse if it changes every request.

Moving highly dynamic content later in the prompt can improve cacheability.

---

# 48. Prompt engineering becomes cache engineering

This is an interesting consequence.

Traditionally, prompt engineering asks:

> "How should I structure the instructions to get better model behavior?"

With prefix caching, you also ask:

> "How should I structure the prompt so stable content appears in the reusable prefix?"

For example:

```text id="m5x9q2"
STATIC SYSTEM INSTRUCTIONS
STATIC TOOL DEFINITIONS
STATIC EXAMPLES
STATIC OUTPUT FORMAT
----------------------
DYNAMIC USER CONTEXT
DYNAMIC RETRIEVAL
DYNAMIC QUERY
```

This can make the prompt both:

```text id="x7c3m8"
behaviorally consistent
```

and:

```text id="q1n6v4"
cache-friendly
```

That is a useful production design principle.

---

# 49. Prefix caching doesn't necessarily mean zero prefill

Suppose:

```text id="b8q2m6"
10,000-token prompt
```

and:

```text id="h3x7c1"
8,000-token prefix
```

is cached.

The model still needs to process:

```text id="n6m4p8"
2,000-token suffix
```

and integrate that suffix with the cached context.

So the savings are:

```text id="j2c7x5"
avoid computing 8,000 tokens
```

not:

```text id="q4m8n1"
avoid the entire model computation
```

The request still requires inference.

This is why prefix caching should be thought of as **partial computation reuse**.

---

# 50. Prefix caching and decode latency

Prefix caching primarily improves:

```text id="c5m8q2"
prefill time
```

which directly affects:

```text id="x7n3p9"
Time To First Token
```

If your request has:

```text id="m4q8z1"
20,000-token prefix
```

and:

```text id="100-token output"
```

then prefill may dominate the time before the first token arrives.

Prefix caching can therefore significantly improve perceived latency.

However, once decoding starts, the system still needs to process the output tokens.

So prefix caching doesn't automatically increase decode speed by the same factor.

---

# 51. Prefix caching and throughput

Prefix caching can also improve throughput because the GPU spends less compute on repeated prefix processing.

Suppose your workload is:

```text id="h8m2x5"
same 10K-token prefix
+
many different short questions
```

Without caching, GPU compute is dominated by repeatedly processing the same 10K tokens.

With caching:

```text id="p3q7c1"
compute prefix once
+
process short suffixes
+
decode outputs
```

The GPU can spend more of its capacity on useful unique work.

This can substantially increase effective throughput for suitable workloads.

---

# 52. The cache-hit workflow

The simplest mental model is:

```text id="x6m2q8"
New request
     ↓
Tokenize
     ↓
Split into KV blocks
     ↓
Calculate chained hashes
     ↓
Lookup first block
     ↓
Hit?
  /      \
Yes       No
 |         |
 ↓         ↓
next      compute
block     prefix
 |
 ↓
continue until first miss
     ↓
reuse longest cached prefix
     ↓
compute remaining suffix
     ↓
decode
```

This is essentially a trie-like traversal implemented through block hashes.

---

# 53. Why the cache lookup is usually prefix-oriented

Suppose:

```text id="j8c3m6"
A B C D E
```

is cached.

And the new request is:

```text id="q4x7n2"
A B C D X
```

We can reuse:

```text id="p6m1z8"
A B C D
```

but not:

```text id="r3v9c5"
E
```

because the sequence diverged.

The cached state is therefore naturally useful from the beginning of the sequence forward.

This is why it is called:

```text id="z2m6q8"
prefix caching
```

rather than arbitrary sequence caching.

---

# 54. The relationship to a trie

There is another useful conceptual model.

Imagine a tree:

```text id="t5q8m1"
                 START
                   |
                   A
                   |
                   B
                  / \
                 C   X
                 |
                 D
                / \
               E   Y
```

Different prompts share paths from the root.

A prefix cache effectively wants to exploit these shared paths.

For example:

```text id="k2m7x4"
A → B → C → D
```

is shared by multiple requests.

The chained hash approach gives a compact identity to each point along that path.

So you can think of prefix caching as:

```text id="v8q3n6"
shared prompt prefixes
        ↓
shared computation
```

represented efficiently through block-level identities.

---

# 55. A complete production architecture

A production inference server might conceptually look like:

```text id="g5x2m8"
                    Requests
                       |
                       ↓
                  Tokenizer
                       |
                       ↓
                Prefix manager
                       |
             ┌─────────┴─────────┐
             ↓                   ↓
       Hash computation      Cache lookup
             |                   |
             └─────────┬─────────┘
                       ↓
              Longest prefix hit
                       |
                       ↓
               Reuse KV blocks
                       |
                       ↓
                Block allocator
                       |
                       ↓
              Compute new suffix
                       |
                       ↓
              Insert new blocks
                       |
                       ↓
                Decode tokens
                       |
                       ↓
              Reference release
                       |
                       ↓
                Cache eviction
```

The key components are therefore:

```text id="r4m8x2"
Tokenizer
Hashing
Prefix lookup
KV block manager
Reference counting
Allocator
Eviction policy
Attention kernel
Scheduler
```

---

# 56. Security and correctness considerations

Prefix caching is an inference optimization, but it also creates important production concerns.

A cache entry must not accidentally cross incompatible model configurations.

For example, cached KV state for one model should not be treated as valid for another model simply because the token prefix is identical.

Similarly, if your system prompt contains tenant-specific information, you need to ensure that shared cache entries cannot be incorrectly reused across tenants.

Conceptually, the cache identity may need to include a namespace or model-specific context:

```text id="n7m3x5"
tenant
+
model/version
+
relevant configuration
+
prefix identity
```

This prevents accidental cross-context reuse.

In multi-tenant systems, cache isolation and access-control semantics are therefore part of correctness, not merely security decoration.

---

# 57. The most important metrics to monitor

If you deploy prefix caching, don't simply measure:

```text id="v3x7m2"
cache enabled = true
```

Measure whether it is actually working.

Useful production telemetry includes:

```text id="m8q2c5"
prefix cache hit rate

prefix tokens reused

prefix tokens computed

cache memory utilization

cache eviction rate

average cached-prefix length

cache lookup latency

time-to-first-token

prefill throughput

decode throughput
```

One particularly useful metric is:

```text id="q6m1x8"
reused prefix tokens / total input tokens
```

because a 90% request hit rate isn't necessarily impressive if each hit only reuses 50 tokens.

Conversely, even a lower request-level hit rate can be extremely valuable if each hit reuses 20,000 tokens.

---

# 58. A practical decision rule

When evaluating whether to implement prefix caching, ask four questions.

First:

```text id="h7m3x1"
Do requests share an exact token prefix?
```

Second:

```text id="c4q8n2"
How long is that prefix?
```

Third:

```text id="m9x2v6"
How frequently is it reused before eviction?
```

Fourth:

```text id="p5q1z8"
How much GPU memory does keeping its KV state consume?
```

If you have:

```text id="y3m7c2"
long prefix
+
high reuse
+
reasonable cache residency
```

prefix caching is likely to provide meaningful value.

---

# 59. The connection to our previous topics

Now we can connect all four concepts we've covered.

First:

```text id="j4x8m2"
KV Cache
```

stores K/V state so previous tokens don't need to be recomputed.

Second:

```text id="v6q2c9"
GQA
```

reduces how much K/V state exists.

Third:

```text id="n8m3x5"
PagedAttention
```

divides the KV state into blocks and manages those blocks efficiently.

Fourth:

```text id="q1z7m4"
Prefix caching
```

allows identical prefix KV blocks to be reused across different requests.

So the progression is:

```text id="a7m2c8"
KV cache
   ↓
Don't recompute previous tokens


GQA
   ↓
Make the cache smaller


PagedAttention
   ↓
Manage the cache efficiently


Prefix caching
   ↓
Reuse cache across requests
```

These four ideas together form a significant part of modern LLM inference engineering.

---

# 60. The deepest takeaway

Prefix caching is essentially **memoization of transformer computation for shared token prefixes**.

The key insight is:

```text id="c5m9x2"
Same tokens
+
same relevant model context
=
same KV state
```

Therefore, if many requests share:

```text id="n3q7m1"
system prompts
tool definitions
few-shot examples
agent instructions
static documents
```

you can compute the corresponding KV state once and reuse it.

The **chained hash** provides a compact identity for each complete KV block:

```text id="r8m2x5"
H0 = hash(seed, Block0)

H1 = hash(H0, Block1)

H2 = hash(H1, Block2)

H3 = hash(H2, Block3)
```

This lets the system discover the **longest matching cached prefix** without comparing huge prompt strings or scanning every cached sequence.

The **PagedAttention block manager** then provides the physical storage:

```text id="v4q8m1"
Hash
 ↓
Physical KV block
```

and multiple requests can reference the same physical blocks through reference counting.

The most important production insight is that **prefix caching is valuable when the beginning of requests is both long and repetitive**.

A 50-token common prefix used twice is not particularly exciting.

A 10,000-token system prompt reused across 100,000 requests is a completely different problem.

That is why the strongest use cases are:

```text id="z7m3c5"
Large shared system prompts
        +
Few-shot templates
        +
Agent/tool definitions
        +
Repeated enterprise workflows
```

while highly dynamic prompts with little exact prefix reuse may see little benefit.

And there is one architectural rule worth remembering:

```text id="p4x8m2"
Put stable content first.
Put dynamic content later.
```

For example:

```text id="q7m1c9"
STATIC SYSTEM PROMPT
STATIC POLICIES
STATIC TOOL DEFINITIONS
STATIC FEW-SHOT EXAMPLES
STATIC OUTPUT FORMAT
────────────────────────
DYNAMIC USER CONTEXT
DYNAMIC RETRIEVED DOCUMENTS
DYNAMIC QUERY
```

This structure maximizes the portion of the prompt that can potentially become a reusable KV prefix.

The overall inference optimization stack now looks like:

```text id="x8m3q6"
                    LLM Inference
                         |
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
         GQA          KV Cache       Prefix Cache
          ↓              ↓              ↓
    fewer KV heads   reuse K/V     reuse across
          ↓              ↓           requests
          └──────────────┼──────────────┘
                         ↓
                  PagedAttention
                         ↓
                 KV blocks + allocator
                         ↓
                Continuous batching
                         ↓
                  High throughput
```

That is the conceptual bridge from **transformer architecture** to **real LLM inference systems**: the model determines the KV state, GQA controls how large that state is, PagedAttention determines how that state is physically managed, and prefix caching determines whether the same state can be reused across different requests.
