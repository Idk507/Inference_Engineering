# MHA vs MQA vs GQA: The Head-Sharing Math Behind Modern LLM Attention

The easiest way to understand **MHA, MQA, and GQA** is to forget the names for a moment and ask one simple question:

> **How many different Key and Value heads do we actually store for all the Query heads?**

That question matters enormously during inference because the Key and Value tensors are exactly what go into the **KV cache**. More KV heads mean more memory. Fewer KV heads mean less memory and less memory bandwidth during decoding.

The evolution from MHA → MQA → GQA is therefore largely the story of finding a practical balance between **attention quality, KV-cache memory, memory bandwidth, and inference throughput**.

---

# 1. Start with ordinary multi-head attention

Let's begin with the original idea: **Multi-Head Attention**, or MHA.

Suppose our transformer has:

```text
8 attention heads
```

For every token, the model creates:

```text
8 Query heads
8 Key heads
8 Value heads
```

So conceptually:

```text
Query heads:

Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8

Key heads:

K1 K2 K3 K4 K5 K6 K7 K8

Value heads:

V1 V2 V3 V4 V5 V6 V7 V8
```

Each Query head has its own corresponding Key and Value head:

```text
Q1 → K1 → V1
Q2 → K2 → V2
Q3 → K3 → V3
...
Q8 → K8 → V8
```

The heads independently perform attention and their outputs are eventually combined.

The important thing for our discussion is:

```text
8 Q heads
8 K heads
8 V heads
```

Therefore, the model needs to cache all 8 Key heads and all 8 Value heads.

---

# 2. Why do we need multiple heads?

Before looking at memory, we should understand why multiple heads exist.

Imagine the sentence:

```text
"The dog that chased the cat was tired."
```

Different attention heads can learn different relationships.

One head might focus on:

```text
dog ↔ chased
```

Another might focus on:

```text
cat ↔ chased
```

Another might focus on:

```text
dog ↔ tired
```

Another might focus on positional or syntactic relationships.

So instead of having one enormous attention mechanism, the transformer divides attention into multiple smaller heads.

Conceptually:

```text
                    Transformer

              ┌───────┼───────┐
              ↓       ↓       ↓
            Head 1   Head 2   Head 3
              ↓       ↓       ↓
          relationship relationship
              ↓       ↓       ↓
              └───────┼───────┘
                      ↓
                 Combined output
```

This gives the model multiple representational "views" of the sequence.

---

# 3. The problem appears during inference

During training, MHA is computationally manageable because the model processes many tokens in parallel.

Inference is different.

Suppose the model has already processed:

```text
10,000 tokens
```

For every layer, we have stored:

```text
K1 ... K10000
V1 ... V10000
```

for every attention head.

If there are 32 heads, that means:

```text
32 Key heads
32 Value heads
```

for every token.

That's a lot of memory.

And during decoding, the GPU repeatedly needs to read those cached K/V tensors.

So the problem isn't only:

> "How much VRAM does the cache occupy?"

It is also:

> "How much memory bandwidth do I need to read the cache for every generated token?"

This second problem is extremely important.

---

# 4. The fundamental MHA memory equation

For one token, the KV cache contains:

```text
K + V
```

For each layer.

If we have:

```text
L = number of layers
H = number of KV heads
D = head dimension
B = bytes per element
```

then the cache size per token is:

```text
2 × L × H × D × B
```

The first `2` is:

```text
Key + Value
```

For MHA:

```text
H = number of query heads
```

because every query head has its own K/V head.

---

# 5. Example MHA model

Let's construct a simple model:

```text
Layers       = 32
Query heads  = 32
KV heads     = 32
Head dim     = 128
Precision    = BF16
Bytes        = 2
```

The per-token KV cache is:

```text
2 × 32 × 32 × 128 × 2
```

Calculate it step by step:

```text
2 × 32 = 64

64 × 32 = 2,048

2,048 × 128 = 262,144

262,144 × 2 = 524,288 bytes
```

So:

```text
524,288 bytes/token
```

which is:

```text
512 KiB/token
```

Now look at a 128K context:

```text
512 KiB × 131,072
```

That gives:

```text
64 GiB
```

of raw KV cache.

That's huge.

This is one reason why reducing the number of KV heads became so attractive.

---

# 6. Enter Multi-Query Attention

Researchers asked a very natural question:

> Do we really need a separate Key and Value head for every Query head?

The answer turned out to be:

> Not necessarily.

This led to **Multi-Query Attention**, or MQA.

Instead of:

```text
32 Q heads
32 K heads
32 V heads
```

we could use:

```text
32 Q heads
1 K head
1 V head
```

The Query heads remain independent.

But they all share the same Key and Value representation.

Conceptually:

```text
Q1 ──┐
Q2 ──┤
Q3 ──┤
Q4 ──┤
Q5 ──┤
...  ├──→ K
Q32 ─┘

Q1 ──┐
Q2 ──┤
Q3 ──┤
...  ├──→ V
Q32 ─┘
```

This is a dramatic reduction in KV-cache size.

---

# 7. MQA memory calculation

Use the same model:

```text
Layers       = 32
Query heads  = 32
KV heads     = 1
Head dim     = 128
BF16         = 2 bytes
```

Now:

```text
2 × 32 × 1 × 128 × 2
```

Calculate:

```text
2 × 32 = 64

64 × 1 = 64

64 × 128 = 8,192

8,192 × 2 = 16,384 bytes
```

Therefore:

```text
16,384 bytes/token
```

or:

```text
16 KiB/token
```

At 128K tokens:

```text
16 KiB × 131,072
```

equals:

```text
2 GiB
```

Compare that with MHA:

```text
MHA = 64 GiB
MQA = 2 GiB
```

That's a:

```text
32× reduction
```

in KV-cache memory.

Why exactly 32×?

Because:

```text
32 KV heads / 1 KV head = 32
```

This is the fundamental mathematical advantage of MQA.

---

# 8. But MQA isn't free

At this point, MQA might look obviously superior.

Why would anyone use MHA anymore?

Because there is a trade-off.

Remember that multiple attention heads allow the model to learn different relationships.

With MHA:

```text
Q1 → K1,V1
Q2 → K2,V2
Q3 → K3,V3
...
```

Every query head has its own Key/Value representation.

With MQA:

```text
Q1 ─┐
Q2 ─┤
Q3 ─┤
... ├──→ shared K,V
Q32 ┘
```

All query heads have to share the same K/V representation.

The Queries remain different, so the heads are not identical, but the information they retrieve from is much more constrained.

In practice, MQA can provide very substantial inference-memory and bandwidth savings, but aggressive sharing can introduce quality degradation depending on the model and training recipe.

This created the need for a middle ground.

---

# 9. That middle ground is GQA

GQA means:

# Grouped-Query Attention

Instead of:

```text
32 Q heads
32 KV heads
```

or:

```text
32 Q heads
1 KV head
```

we might use:

```text
32 Q heads
8 KV heads
```

Now each KV head is shared by a group of Query heads.

For example:

```text
Q1  Q2  Q3  Q4  → KV1

Q5  Q6  Q7  Q8  → KV2

Q9  Q10 Q11 Q12 → KV3

Q13 Q14 Q15 Q16 → KV4

Q17 Q18 Q19 Q20 → KV5

Q21 Q22 Q23 Q24 → KV6

Q25 Q26 Q27 Q28 → KV7

Q29 Q30 Q31 Q32 → KV8
```

Therefore:

```text
32 Query heads
8 KV heads
```

Each KV head serves:

```text
32 / 8 = 4
```

Query heads.

That's the key mathematical relationship in GQA.

---

# 10. The general GQA head-sharing formula

Let:

```text
Q = number of query heads

K = number of KV heads
```

Then the number of Query heads sharing each KV head is:

```text
Q / K
```

For GQA to form equal-sized groups, `Q` should generally be divisible by `K`.

For example:

```text
Q = 32
K = 8
```

gives:

```text
32 / 8 = 4
```

So:

```text
4 Query heads
        ↓
1 KV head
```

Another example:

```text
Q = 64
K = 8
```

gives:

```text
64 / 8 = 8
```

So:

```text
8 Query heads
        ↓
1 KV head
```

---

# 11. Now calculate GQA memory

Use our example:

```text
Layers       = 32
Query heads  = 32
KV heads     = 8
Head dim     = 128
BF16         = 2 bytes
```

The cache formula is:

```text
2 × 32 × 8 × 128 × 2
```

Calculate:

```text
2 × 32 = 64

64 × 8 = 512

512 × 128 = 65,536

65,536 × 2 = 131,072 bytes
```

Therefore:

```text
131,072 bytes/token
```

or:

```text
128 KiB/token
```

At 128K tokens:

```text
128 KiB × 131,072
```

equals:

```text
16 GiB
```

Now we have:

```text
MHA → 64 GiB
GQA → 16 GiB
MQA → 2 GiB
```

for the same hypothetical 128K context.

This makes the design trade-off immediately visible.

---

# 12. The key ratio

There is a very useful shortcut.

If the Query head count is:

```text
Q
```

and the KV head count is:

```text
K
```

then compared with MHA, the KV-cache memory ratio is approximately:

```text
K / Q
```

because all other terms remain the same.

So with:

```text
Q = 32
K = 8
```

we get:

```text
8 / 32
=
1 / 4
```

Therefore GQA uses approximately:

```text
25%
```

of the MHA KV-cache memory.

Equivalently:

```text
4× less KV memory
```

than MHA.

---

# 13. The same idea applies to memory bandwidth

This is even more important than just VRAM capacity.

During decoding, the model needs to read the historical K/V tensors to perform attention.

Imagine:

```text
128K tokens
```

and every generated token requires accessing a huge KV cache.

The GPU has to move those tensors from memory into the computation pipeline.

Therefore:

```text
More KV heads
       ↓
More K/V data
       ↓
More memory traffic
       ↓
More memory-bandwidth pressure
```

GQA reduces the amount of K/V data that needs to be stored and read.

So GQA can improve not only:

```text
memory capacity
```

but also:

```text
memory bandwidth efficiency
```

This is one of the reasons the architecture is particularly attractive for autoregressive decoding.

---

# 14. Why GQA is such a practical compromise

We can now visualize the three approaches.

### MHA

```text
32 Q
32 K
32 V
```

Maximum KV-head diversity, but maximum KV memory.

### MQA

```text
32 Q
1 K
1 V
```

Extremely small KV cache, but maximum sharing of K/V information.

### GQA

```text
32 Q
8 K
8 V
```

Moderate KV memory with substantially less sharing than MQA.

So GQA effectively says:

> Keep many independent Query heads, but make a smaller number of Key/Value heads that can be shared among groups of Query heads.

That is a very useful compromise.

---

# 15. Why not simply reduce the number of Query heads?

This is an important conceptual distinction.

Suppose we have:

```text
32 Query heads
8 KV heads
```

You might wonder:

> Why not simply make it an 8-head model?

Because Query heads and KV heads have different roles.

The Query side controls the different ways in which the current representation asks questions about the context.

The Key/Value side controls the information available for retrieval.

GQA keeps a large number of independent queries while reducing the amount of cached information.

That's why it can retain much of the modeling flexibility associated with many attention heads while substantially reducing inference memory.

---

# 16. The matrix perspective

Let's make the mathematics slightly more concrete.

Suppose the hidden dimension is:

```text
4096
```

and we have:

```text
32 Query heads
```

Then:

```text
4096 / 32
=
128
```

So each Query head has dimension:

```text
128
```

Therefore the Query projection produces approximately:

```text
32 × 128
=
4096
```

values per token.

With MHA, Key and Value projections also produce:

```text
32 × 128
=
4096
```

values each.

So:

```text
Q = 4096 values
K = 4096 values
V = 4096 values
```

per token before considering layers and precision.

---

# 17. Now GQA changes the K/V projection size

Suppose:

```text
Query heads = 32
KV heads = 8
head dimension = 128
```

Then:

```text
K dimension
=
8 × 128
=
1024
```

and:

```text
V dimension
=
8 × 128
=
1024
```

So the model produces:

```text
Q = 4096 values
K = 1024 values
V = 1024 values
```

per token.

This is an important architectural difference.

The Query representation remains large:

```text
4096
```

while the Key and Value representations are reduced:

```text
1024
```

This is exactly where the KV-cache savings come from.

---

# 18. The projection-matrix perspective

Suppose the input hidden dimension is:

```text
4096
```

For MHA, a simplified representation of the projections would be:

```text
Wq → 4096 × 4096
Wk → 4096 × 4096
Wv → 4096 × 4096
```

For GQA with 8 KV heads:

```text
Wq → 4096 × 4096

Wk → 4096 × 1024

Wv → 4096 × 1024
```

So GQA also reduces the parameters and computation associated with the K/V projections.

The major inference benefit, however, is especially visible in the KV cache and memory bandwidth.

---

# 19. A subtle but important point about head dimension

When comparing MHA, MQA and GQA, don't accidentally change the head dimension.

Suppose:

```text
hidden dimension = 4096
query heads = 32
```

Then:

```text
head dimension = 4096 / 32 = 128
```

With GQA:

```text
KV heads = 8
```

The KV projection dimension becomes:

```text
8 × 128
=
1024
```

We do **not** normally say:

```text
4096 / 8 = 512
```

and then use 512 as the KV head dimension merely because there are fewer KV heads.

That would change the architecture in a different way.

In standard GQA, the Query and KV heads use the same head dimension.

So:

```text
Q head dimension = 128
K head dimension = 128
V head dimension = 128
```

while the number of heads differs.

This distinction is critical when calculating KV memory correctly.

---

# 20. Head sharing during actual attention

Let's take:

```text
32 Q heads
8 KV heads
```

and focus on the first group:

```text
Q1
Q2
Q3
Q4
```

These four Query heads share one K/V head.

Conceptually:

```text
Q1 ─┐
Q2 ─┤
Q3 ─┤──→ K1, V1
Q4 ─┘
```

The next group:

```text
Q5
Q6
Q7
Q8
```

shares:

```text
K2, V2
```

and so on.

During attention, each Query still produces its own attention scores.

For example:

```text
Q1 × K1
Q2 × K1
Q3 × K1
Q4 × K1
```

So the Query heads remain different.

They simply operate against shared K/V information.

That is why GQA is not equivalent to simply reducing the number of attention heads.

---

# 21. A simple analogy

Imagine 32 researchers.

With MHA, every researcher has their own:

```text
database
```

So:

```text
32 researchers
32 databases
```

With MQA:

```text
32 researchers
1 database
```

Everyone asks questions against the same database.

With GQA:

```text
32 researchers
8 databases
```

Every four researchers share one database.

That is the essence of GQA.

The researchers are the Query heads.

The databases are the Key/Value heads.

The database contents are the cached attention information.

---

# 22. Why GQA became popular in modern open models

GQA became particularly attractive as open models moved toward longer contexts and production inference.

The reason is not simply:

> "GQA is newer."

The underlying hardware economics make it attractive.

Modern LLM inference is frequently constrained by memory capacity and memory bandwidth, especially during autoregressive decoding.

As context length grows:

```text
context ↑
   ↓
KV cache ↑
   ↓
memory traffic ↑
```

As batch size grows:

```text
batch ↑
   ↓
number of active KV caches ↑
   ↓
VRAM pressure ↑
```

GQA attacks both problems by reducing the number of stored K/V heads.

This is particularly useful for serving many simultaneous requests.

---

# 23. Why open models didn't simply standardize on MQA

MQA provides an even greater memory reduction.

But model architecture is not just about minimizing inference memory.

There is a quality/efficiency trade-off.

A model designer has to consider:

```text
Attention expressiveness
+
training behavior
+
model quality
+
inference latency
+
KV memory
+
memory bandwidth
+
hardware efficiency
```

MQA pushes harder toward efficiency.

MHA pushes harder toward independent attention representations.

GQA sits between the two.

That makes it an attractive architecture for many modern decoder-only models.

---

# 24. The Llama example

The idea becomes especially concrete with Llama-family architectures.

For an 8B-class Llama 3-style configuration, a commonly cited attention setup is approximately:

```text
32 transformer layers

32 query heads

8 KV heads

head dimension = 128
```

So:

```text
32 / 8
=
4
```

Query heads share each KV head.

The resulting KV-cache calculation in BF16 is:

```text
2 × 32 × 8 × 128 × 2
```

which equals:

```text
131,072 bytes/token
```

or:

```text
128 KiB/token
```

At 128K tokens:

```text
128 KiB × 131,072
=
16 GiB
```

That is dramatically smaller than the approximately 64 GiB that the same layer/head configuration would require with 32 KV heads.

This is the practical power of GQA.

---

# 25. The broader pattern in modern model design

As models became larger and context windows became longer, the industry increasingly had to treat inference memory as a first-class architectural constraint.

A model isn't just:

```text
parameters
```

anymore.

For inference, you need to think about:

```text
weights
+
KV cache
+
activations
+
batch size
+
context length
+
memory bandwidth
+
communication
```

GQA directly addresses one of the largest variable components:

```text
KV cache
```

And importantly, the savings scale with sequence length.

At 1K context, the difference may be manageable.

At 128K context, it becomes enormous.

---

# 26. MHA vs MQA vs GQA mathematically

Let's put the three architectures into one framework.

Suppose:

```text
Q = 32 Query heads
D = 128 head dimension
L = 32 layers
B = 2 bytes
```

For MHA:

```text
KV heads = 32
```

Therefore:

```text
KV/token
=
2 × 32 × 32 × 128 × 2

=
524,288 bytes

=
512 KiB
```

For GQA:

```text
KV heads = 8
```

Therefore:

```text
KV/token
=
2 × 32 × 8 × 128 × 2

=
131,072 bytes

=
128 KiB
```

For MQA:

```text
KV heads = 1
```

Therefore:

```text
KV/token
=
2 × 32 × 1 × 128 × 2

=
16,384 bytes

=
16 KiB
```

So:

```text
Architecture    Q heads    KV heads    KV/token

MHA             32         32          512 KiB

GQA             32         8           128 KiB

MQA             32         1           16 KiB
```

The query count remains constant.

Only the number of KV heads changes.

---

# 27. The most useful equation for comparing them

For the same model dimensions:

```text
KV memory ∝ number of KV heads
```

Therefore:

```text
MHA : GQA : MQA

32 : 8 : 1
```

in our example.

Or normalized against MHA:

```text
MHA = 1.00

GQA = 0.25

MQA = 0.03125
```

So GQA uses one quarter of the MHA KV memory, while MQA uses one thirty-second.

---

# 28. But GQA doesn't eliminate the KV-cache problem

This is important.

GQA makes the problem smaller.

It does not make it disappear.

Even:

```text
128 KiB/token
```

becomes:

```text
1 GiB
```

at roughly 8K tokens.

And:

```text
16 GiB
```

at 128K tokens.

Then multiply that by concurrent requests.

For example:

```text
16 GiB/request
×
8 concurrent long-context requests
```

could theoretically require:

```text
128 GiB
```

of KV-cache payload alone.

That's why GQA must often be combined with:

```text
Paged KV cache
Continuous batching
Prefix caching
KV quantization
Tensor parallelism
KV offloading
Memory-aware scheduling
```

The architecture reduces the fundamental cost; the serving system manages what remains.

---

# 29. GQA and continuous batching

Imagine a production server handling:

```text
Request A → 2K tokens
Request B → 8K tokens
Request C → 32K tokens
Request D → 4K tokens
```

Their KV caches have different sizes.

A good inference engine doesn't want to waste GPU memory by allocating the maximum context size to every request.

Instead, it manages KV memory dynamically.

GQA helps because every request's KV footprint is already smaller.

Then the serving engine can use that memory more efficiently across concurrent users.

So the combination is powerful:

```text
GQA
  ↓
Smaller KV state

Paged KV management
  ↓
Less memory fragmentation

Continuous batching
  ↓
Better GPU utilization
```

These architectural and systems-level optimizations complement each other.

---

# 30. One subtle issue: GQA doesn't necessarily mean every model uses exactly 8 KV heads

GQA is a design pattern, not a fixed configuration.

You might encounter:

```text
64 Q heads
8 KV heads
```

or:

```text
32 Q heads
8 KV heads
```

or:

```text
32 Q heads
4 KV heads
```

or other ratios.

The important relationship is:

```text
group size = Q heads / KV heads
```

For example:

```text
64 / 8 = 8
```

means:

```text
8 Query heads per KV head
```

while:

```text
32 / 8 = 4
```

means:

```text
4 Query heads per KV head
```

So when you inspect a new model architecture, don't ask:

> "Does it use GQA?"

and stop there.

Ask:

> "How many Query heads and how many KV heads does it actually have?"

That gives you the real memory calculation.

---

# 31. How to identify GQA from a model config

When working with Hugging Face model configurations, you'll often see fields related to:

```text
num_attention_heads
num_key_value_heads
hidden_size
num_hidden_layers
```

Suppose you find:

```text
num_attention_heads = 32

num_key_value_heads = 8
```

Then:

```text
32 ≠ 8
```

which indicates grouped/shared KV heads.

The sharing ratio is:

```text
32 / 8 = 4
```

So every KV head serves four Query heads.

If instead you see:

```text
32
32
```

you're dealing with the standard MHA-style head count.

If you see:

```text
32
1
```

you're dealing with MQA-style sharing.

This is one of the easiest ways to understand the architecture of an unfamiliar model.

---

# 32. The evolution in one picture

The entire progression can be summarized like this:

```text
MHA

Q1 → K1,V1
Q2 → K2,V2
Q3 → K3,V3
Q4 → K4,V4
...
Q32 → K32,V32


        ↓ reduce KV memory


GQA

Q1 ─┐
Q2 ─┤
Q3 ─┤ → K1,V1
Q4 ─┘

Q5 ─┐
Q6 ─┤
Q7 ─┤ → K2,V2
Q8 ─┘

...


        ↓ push sharing further


MQA

Q1 ─┐
Q2 ─┤
Q3 ─┤
... ├──→ K,V
Q32 ─┘
```

So:

```text
MHA
Many Q
Many K/V

GQA
Many Q
Some K/V

MQA
Many Q
One K/V
```

That's the entire architectural idea.

---

# 33. Why GQA is often the practical sweet spot

The reason GQA became so common is that it addresses a real systems constraint without forcing the model all the way to maximum sharing.

MHA gives every Query head independent K/V information, but its KV cache can become expensive.

MQA minimizes KV memory, but it imposes maximal sharing on K/V representations.

GQA allows model designers to choose an intermediate point.

For example:

```text
32 Q heads
8 KV heads
```

means:

```text
4 Q heads share one KV head
```

So the model receives a substantial:

```text
4× KV-memory reduction
```

relative to MHA while retaining multiple independent KV groups.

This is particularly valuable when serving long-context workloads where the KV cache and memory bandwidth dominate inference behavior.

---

# 34. The deeper lesson

MHA, MQA, and GQA are not merely three different names for attention.

They represent three different answers to a fundamental systems question:

> **How much independent Key/Value information do we really need to support many Query heads?**

MHA says:

```text
Give everyone their own K/V.
```

MQA says:

```text
Let everyone share K/V.
```

GQA says:

```text
Let groups of Query heads share K/V.
```

And that middle ground is extremely attractive because modern LLM inference is increasingly constrained not only by compute, but by **memory capacity and memory bandwidth**.

---

# 35. The mental model you should keep

If you remember only one diagram, remember this:

```text
                    QUERY HEADS
                       |
          ┌────────────┼────────────┐
          ↓            ↓            ↓

MHA:      Q1           Q2           Q3 ... Q32
          |            |            |
          K1           K2           K3 ... K32
          |            |            |
          V1           V2           V3 ... V32


GQA:      Q1 Q2 Q3 Q4   Q5 Q6 Q7 Q8   ...
          \  |  |  /     \  |  |  /
             K1,V1          K2,V2
               


MQA:      Q1 Q2 Q3 Q4 Q5 ... Q32
          \  |  |  |       /
                 K,V
```

And the memory relationship is:

```text
MHA
KV heads = Q heads
KV memory = maximum


GQA
KV heads < Q heads
KV memory = reduced


MQA
KV heads = 1
KV memory = minimum
```

For the concrete 32-Q-head example:

```text
MHA → 32 KV heads → 512 KiB/token

GQA → 8 KV heads  → 128 KiB/token

MQA → 1 KV head   → 16 KiB/token
```

At 128K context:

```text
MHA → 64 GiB

GQA → 16 GiB

MQA → 2 GiB
```

That single calculation explains a huge portion of why **GQA became such an important architecture for modern open-weight LLMs**: it substantially reduces the variable inference cost of the KV cache while preserving more independent attention structure than MQA.

The next concept that naturally follows from this is **PagedAttention**, because once you understand that GQA reduces *how much* KV memory you need, PagedAttention explains how inference engines efficiently manage the KV memory you still have.


KV-cache memory at 128K tokens

Illustrative comparison using 32 layers, 32 query heads, head dimension 128, and BF16 KV cache.

architecture	memory
MHA	64
GQA (8 KV)	16
MQA	2
