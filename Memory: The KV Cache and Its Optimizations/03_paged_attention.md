# PagedAttention: From First Principles to the Block Data Structure, Allocator, and Fragmentation Math

PagedAttention becomes much easier to understand once you connect it directly to the KV cache.

The KV cache is one of the largest pieces of mutable memory in LLM inference. Every active request continuously grows its cache as new tokens are generated. If we allocate that cache as one large contiguous region for every request, GPU memory becomes difficult to manage efficiently. Requests have different lengths, they finish at different times, and their KV caches grow dynamically.

**PagedAttention solves this memory-management problem by separating the logical KV cache from its physical GPU memory layout.**

Instead of saying:

```text
Request A owns one giant contiguous KV-cache region.
```

we say:

```text
Request A owns a sequence of fixed-size KV blocks.

Logical block 0 → physical block 17
Logical block 1 → physical block 4
Logical block 2 → physical block 29
Logical block 3 → physical block 8
```

The attention operation follows the logical sequence, while the underlying GPU memory can be scattered across physical blocks.

This idea is closely related to virtual memory in operating systems, but applied to the KV cache.

---

# 1. The problem PagedAttention is solving

Suppose we have a model whose KV cache requires:

```text
128 KiB per token
```

and a request currently contains:

```text
10,000 tokens
```

The raw KV-cache requirement is approximately:

```text
10,000 × 128 KiB
```

which is approximately:

```text
1.25 GiB
```

Now imagine 100 users.

Some requests might be:

```text
User A → 500 tokens
User B → 2,000 tokens
User C → 15,000 tokens
User D → 800 tokens
User E → 40,000 tokens
...
```

The problem is that these requests don't grow uniformly.

A request that starts with 500 tokens might later become:

```text
500
1000
1500
2000
...
```

Another request might terminate after 700 tokens.

Another might grow to 30,000.

If you allocate large contiguous memory regions up front, you either waste memory or repeatedly move memory around.

Neither is desirable on a GPU.

---

# 2. The traditional contiguous KV-cache approach

Let's imagine GPU memory as:

```text
GPU VRAM

+------------------------------------------------------+
| Request A KV cache                                   |
+------------------------------------------------------+
| Request B KV cache                                   |
+------------------------------------------------------+
| Request C KV cache                                   |
+------------------------------------------------------+
| Free space                                           |
+------------------------------------------------------+
```

Initially this looks perfectly fine.

But then Request B finishes.

You get:

```text
+------------------------------------------------------+
| Request A KV cache                                   |
+------------------------------------------------------+
| FREE                                                 |
+------------------------------------------------------+
| Request C KV cache                                   |
+------------------------------------------------------+
| Free space                                           |
+------------------------------------------------------+
```

Now another large request arrives.

Suppose it needs a contiguous region larger than the available free region in the middle.

Even if the total free memory is sufficient, the largest contiguous region might not be.

This is **external fragmentation**.

---

# 3. What fragmentation actually means

Suppose GPU memory contains:

```text
Used = 6 GB
Free = 4 GB
```

You might think:

> "I have 4 GB available."

But suppose the free memory is arranged as:

```text
1 GB free
500 MB free
1.5 GB free
1 GB free
```

The total free memory is:

```text
1 + 0.5 + 1.5 + 1
=
4 GB
```

But if the next request requires:

```text
3 GB contiguous
```

you cannot satisfy it.

The problem isn't total free memory.

The problem is:

```text
largest contiguous free region < requested region
```

That's fragmentation.

PagedAttention avoids requiring each request's KV cache to be physically contiguous.

---

# 4. The central idea

PagedAttention divides the KV cache into fixed-size blocks.

For example, suppose one block stores:

```text
16 tokens
```

Then a request containing:

```text
1 token
```

uses one block.

A request containing:

```text
16 tokens
```

uses one block.

A request containing:

```text
17 tokens
```

uses two blocks.

A request containing:

```text
32 tokens
```

uses two blocks.

A request containing:

```text
33 tokens
```

uses three blocks.

The number of blocks is therefore:

```text
ceil(sequence_length / tokens_per_block)
```

The important word is **ceil**.

---

# 5. Why fixed-size blocks help

Suppose:

```text
tokens_per_block = 16
```

and GPU memory contains:

```text
Block 0
Block 1
Block 2
Block 3
Block 4
Block 5
Block 6
Block 7
...
```

Request A might receive:

```text
Block 2
Block 7
Block 11
```

Request B might receive:

```text
Block 1
Block 4
Block 9
```

Request C might receive:

```text
Block 0
Block 3
Block 10
```

Physically:

```text
GPU memory

Block 0 → Request C
Block 1 → Request B
Block 2 → Request A
Block 3 → Request C
Block 4 → Request B
Block 5 → FREE
Block 6 → FREE
Block 7 → Request A
Block 8 → FREE
Block 9 → Request B
Block 10 → Request C
Block 11 → Request A
```

The requests don't care that their physical blocks are scattered.

Each request maintains a mapping:

```text
logical block → physical block
```

That mapping is the key data structure.

---

# 6. Logical memory versus physical memory

This is the most important concept in PagedAttention.

Imagine Request A has:

```text
48 tokens
```

with:

```text
16 tokens per block
```

Therefore:

```text
48 / 16 = 3 blocks
```

Logically, the request sees:

```text
Logical block 0
Logical block 1
Logical block 2
```

But physically those might be:

```text
Physical block 17
Physical block 3
Physical block 42
```

So the mapping table is:

```text
Logical       Physical

Block 0   →   Block 17
Block 1   →   Block 3
Block 2   →   Block 42
```

The model thinks:

```text
Token 0 ... 15
Token 16 ... 31
Token 32 ... 47
```

while the GPU memory might actually store them at:

```text
17
3
42
```

The attention kernel uses the mapping table to find the correct physical data.

---

# 7. This is very similar to virtual memory

Operating systems commonly separate:

```text
Virtual address
```

from:

```text
Physical address
```

A process might think it has:

```text
Virtual page 0
Virtual page 1
Virtual page 2
```

while the physical RAM locations might be:

```text
Page 0 → physical frame 10
Page 1 → physical frame 3
Page 2 → physical frame 18
```

PagedAttention applies a similar abstraction:

```text
Logical KV block
        ↓
Block table
        ↓
Physical KV block
```

The analogy is extremely useful.

But there is an important difference.

PagedAttention isn't literally implementing a general-purpose CPU virtual-memory system. It is a specialized memory-management mechanism for attention/KV-cache storage.

---

# 8. What exactly is inside a KV block?

Now let's go deeper.

Suppose:

```text
block_size = 16 tokens
```

and the model has:

```text
32 layers
8 KV heads
128 head dimension
BF16
```

For one token, our previous calculation gave:

```text
128 KiB
```

of KV data.

Therefore one block contains:

```text
16 × 128 KiB
```

which is:

```text
2 MiB
```

of KV data.

So conceptually:

```text
Physical KV block

+----------------------------------+
| Layer 0   K/V for 16 tokens      |
| Layer 1   K/V for 16 tokens      |
| Layer 2   K/V for 16 tokens      |
| ...                              |
| Layer 31  K/V for 16 tokens      |
+----------------------------------+
```

The exact physical layout depends on the inference implementation and GPU kernel, but logically the block represents KV state for a fixed number of sequence positions.

---

# 9. Why choose a block size such as 16 or 32 tokens?

There is a trade-off.

Smaller blocks provide:

```text
less internal fragmentation
more flexible allocation
```

but create:

```text
more block-table entries
more metadata
potentially more indexing overhead
```

Larger blocks provide:

```text
fewer blocks
less metadata
potentially more efficient memory operations
```

but create:

```text
more unused space in the final block
```

So block size is a systems parameter.

For example:

```text
8 tokens
16 tokens
32 tokens
64 tokens
```

can lead to different memory-management behavior.

The optimal value depends on:

```text
GPU architecture
attention kernel
model architecture
batch size
workload
sequence-length distribution
memory allocator
```

---

# 10. The basic block-count equation

For sequence length:

```text
S
```

and block size:

```text
B
```

the number of blocks required is:

```text
ceil(S / B)
```

For example:

```text
S = 100
B = 16
```

Then:

```text
100 / 16 = 6.25
```

so:

```text
ceil(6.25) = 7
```

blocks are required.

Those seven blocks provide capacity for:

```text
7 × 16
=
112 tokens
```

but only:

```text
100 tokens
```

are currently used.

Therefore:

```text
112 - 100
=
12 token slots
```

are unused in the final block.

That's **internal fragmentation**.

---

# 11. Internal fragmentation

This is different from the external fragmentation we discussed earlier.

With fixed-size blocks, you can still waste some capacity inside the last block.

Suppose:

```text
block size = 16 tokens
sequence = 17 tokens
```

You need:

```text
2 blocks
```

Capacity:

```text
2 × 16
=
32 tokens
```

Actual:

```text
17 tokens
```

Unused:

```text
32 - 17
=
15 token slots
```

That's a lot of waste relative to the actual request.

The percentage is:

```text
15 / 32 × 100
```

which is:

```text
46.875%
```

of allocated token capacity.

However, that worst case happens when the sequence barely crosses a block boundary.

For long sequences, the unused portion is at most one block.

That makes the fragmentation behavior much more predictable.

---

# 12. Maximum internal fragmentation

For block size:

```text
B
```

the final partially filled block can have at most:

```text
B - 1
```

unused token slots.

For:

```text
B = 16
```

maximum unused capacity is:

```text
15 tokens
```

For:

```text
B = 32
```

maximum unused capacity is:

```text
31 tokens
```

Therefore smaller blocks generally reduce worst-case internal fragmentation.

---

# 13. Expected internal fragmentation

If sequence lengths are spread reasonably across block boundaries, the average unused capacity in the final block is approximately:

```text
B / 2
```

token slots.

For a 16-token block:

```text
16 / 2
=
8 tokens
```

average unused capacity.

For a 32-token block:

```text
32 / 2
=
16 tokens
```

So a rough approximation is:

```text
Expected wasted token slots ≈ block_size / 2
```

under a roughly uniform distribution of sequence endings.

This is not a universal exact law because real workloads have non-uniform sequence lengths, but it is a useful mental model.

---

# 14. Why this is still dramatically better than contiguous allocation

With contiguous allocation, you can have significant **external fragmentation** because free memory is split across arbitrary regions.

With block allocation:

```text
Block 0
Block 1
Block 2
Block 3
...
```

every block has the same size.

When a request finishes:

```text
Block 5 → FREE
```

That block can be immediately reused by another request.

It doesn't matter where it is physically located.

Therefore:

```text
Request A
Block 5

Request B
Block 19

Request C
Block 2
```

can all coexist.

The allocator doesn't need to find one giant contiguous region.

This is the central advantage.

---

# 15. The block allocator

Now let's build the allocator conceptually.

Suppose the GPU has:

```text
100 physical KV blocks
```

Initially:

```text
Free blocks:

0, 1, 2, 3, 4, ..., 99
```

The allocator maintains something equivalent to:

```text
free_block_pool
```

which contains available physical block IDs.

A request arrives:

```text
Request A
sequence length = 40
block size = 16
```

Required blocks:

```text
ceil(40 / 16)
=
3
```

The allocator takes three blocks:

```text
5
17
23
```

and creates:

```text
Request A block table

logical 0 → physical 5
logical 1 → physical 17
logical 2 → physical 23
```

The free pool removes those blocks.

---

# 16. Another request arrives

Request B:

```text
sequence length = 25
```

Required blocks:

```text
ceil(25 / 16)
=
2
```

The allocator may assign:

```text
physical 1
physical 8
```

Now:

```text
Request A

logical 0 → 5
logical 1 → 17
logical 2 → 23


Request B

logical 0 → 1
logical 1 → 8
```

Notice that A and B's physical blocks don't need to be adjacent.

---

# 17. Request A grows

Now Request A generates more tokens.

It started with:

```text
40 tokens
```

and therefore had:

```text
3 blocks
```

Suppose it grows to:

```text
50 tokens
```

Three blocks provide:

```text
48 token slots
```

so Request A now needs a fourth block.

The allocator takes another free block:

```text
physical 31
```

and updates:

```text
Request A

logical 0 → 5
logical 1 → 17
logical 2 → 23
logical 3 → 31
```

No existing KV data needs to move.

This is one of the most important advantages.

---

# 18. Why dynamic growth is so important

With contiguous allocation, growing:

```text
40 tokens → 50 tokens
```

can be problematic if the memory immediately after the allocation is already occupied.

You might need to:

```text
allocate larger region
copy existing KV cache
free old region
```

That is expensive.

Paged allocation says:

```text
Need another block?
Allocate another block.
```

Existing blocks stay where they are.

So growth becomes approximately:

```text
allocate one block
update block table
```

rather than:

```text
reallocate entire cache
copy entire cache
```

That is a major systems-level improvement.

---

# 19. Request completion

Suppose Request B finishes.

Its blocks were:

```text
1
8
```

The allocator simply returns them:

```text
1 → FREE
8 → FREE
```

Now those blocks can be assigned to another request.

There is no requirement to merge adjacent free blocks.

That is why physical fragmentation becomes much less problematic.

---

# 20. The block table

The core metadata structure can be represented as:

```text
Request ID     Logical block     Physical block

A              0                 5
A              1                 17
A              2                 23
A              3                 31

B              0                 1
B              1                 8
```

Or more compactly:

```text
A → [5, 17, 23, 31]

B → [1, 8]
```

The position in the array represents the logical block number.

For example:

```text id="6c7ywl"
A → [5, 17, 23, 31]
     ↑   ↑   ↑   ↑
     0   1   2   3
```

So:

```text
logical_block = 2
```

maps to:

```text
physical_block = 23
```

---

# 21. Mapping a token to a physical block

Suppose:

```text
block_size = 16
```

and we want token:

```text
token_index = 37
```

First calculate:

```text
logical_block = floor(37 / 16)
```

which gives:

```text
2
```

Then calculate the offset within the block:

```text
offset = 37 % 16
```

which gives:

```text
5
```

So:

```text
Token 37
    ↓
Logical block 2
    ↓
Block table[2]
    ↓
Physical block
    ↓
Offset 5
```

If:

```text
block_table[2] = 23
```

then token 37 is physically located at:

```text
physical block 23
offset 5
```

That's the fundamental address translation.

---

# 22. The attention kernel's perspective

Suppose the current Query needs to attend over 100 tokens.

With block size:

```text
16
```

we have:

```text
7 logical blocks
```

The kernel can conceptually process:

```text
logical block 0 → physical block X
logical block 1 → physical block Y
logical block 2 → physical block Z
...
```

For every logical block:

```text
1. Read physical block ID.
2. Load K/V data from that physical block.
3. Perform attention computation.
4. Continue to the next logical block.
```

So the kernel sees the sequence logically as:

```text
Token 0
Token 1
Token 2
...
Token 99
```

even though the physical storage may look like:

```text
Block 17
Block 4
Block 91
Block 2
...
```

This separation is the heart of PagedAttention.

---

# 23. A concrete memory example

Let's use the earlier model:

```text
32 layers
8 KV heads
128 head dimension
BF16
```

We calculated:

```text
128 KiB per token
```

Now suppose:

```text
block size = 16 tokens
```

Then one physical block stores:

```text
16 × 128 KiB
```

which is:

```text
2 MiB
```

Now suppose the GPU reserves:

```text
10,000 KV blocks
```

Then raw KV capacity is:

```text
10,000 × 2 MiB
```

which equals:

```text
20,000 MiB
```

or approximately:

```text
19.53 GiB
```

of raw KV-cache storage.

The allocator can distribute these blocks among active requests.

---

# 24. Request allocation example

Suppose we have:

```text
10,000 physical blocks
```

and three requests:

```text
A = 100 tokens
B = 500 tokens
C = 1,000 tokens
```

With:

```text
16 tokens/block
```

Request A requires:

```text
ceil(100 / 16)
=
7 blocks
```

Request B:

```text
ceil(500 / 16)
=
32 blocks
```

Request C:

```text
ceil(1000 / 16)
=
63 blocks
```

Total:

```text
7 + 32 + 63
=
102 blocks
```

So only 102 of the 10,000 blocks are occupied.

The physical placement could be arbitrary:

```text
A → [3, 8, 17, 44, 52, 71, 91]

B → [1, 2, 5, 9, ...]

C → [0, 4, 6, ...]
```

The allocator doesn't care that they are interleaved.

---

# 25. Fragmentation math

Now let's calculate fragmentation more formally.

Let:

```text
S = sequence length
B = block size
```

Allocated token capacity is:

```text
ceil(S / B) × B
```

Actual tokens are:

```text
S
```

Therefore wasted capacity is:

```text
ceil(S / B) × B - S
```

This is the internal fragmentation.

For example:

```text
S = 100
B = 16
```

Allocated:

```text
ceil(100 / 16) × 16

=
7 × 16

=
112
```

Waste:

```text
112 - 100
=
12 token slots
```

Percentage of allocated capacity wasted:

```text
12 / 112 × 100
```

which is approximately:

```text
10.71%
```

---

# 26. Compare different block sizes

Take:

```text
S = 100 tokens
```

For:

```text
B = 8
```

we need:

```text
ceil(100 / 8)
=
13 blocks
```

capacity:

```text
13 × 8
=
104
```

waste:

```text
4 tokens
```

For:

```text
B = 16
```

capacity:

```text
112
```

waste:

```text
12
```

For:

```text
B = 32
```

capacity:

```text
128
```

waste:

```text
28
```

Therefore:

```text
Block size     Capacity     Waste

8              104          4

16             112          12

32             128          28
```

Smaller blocks reduce internal fragmentation.

But smaller blocks also mean more blocks and therefore more metadata and potentially more indexing overhead.

That is the allocator trade-off.

---

# 27. The surprising part: external fragmentation largely disappears

Traditional contiguous allocation can suffer from:

```text
external fragmentation
```

because requests need large contiguous regions.

Paged allocation changes the requirement.

Instead of asking:

> "Do I have 2 GB contiguous?"

the allocator asks:

> "Do I have enough free blocks?"

Suppose:

```text
Free blocks:
1
4
7
12
19
25
31
```

These are physically scattered.

A new request needs:

```text
4 blocks
```

That's fine.

It doesn't matter that the blocks aren't adjacent.

This is the fundamental fragmentation advantage.

---

# 28. The allocator becomes a block-count problem

Suppose the total number of physical blocks is:

```text
N
```

and the currently allocated blocks are:

```text
A
```

Then:

```text
Free blocks = N - A
```

A request requiring:

```text
R
```

blocks can be allocated if:

```text
N - A >= R
```

There is no additional requirement that those blocks be contiguous.

This is much simpler than contiguous allocation.

---

# 29. The free-block data structure

A production allocator might conceptually maintain:

```text
Free block pool
```

containing:

```text
[3, 8, 17, 24, 31, 45, 67, ...]
```

When a request needs a block:

```text
pop()
```

When a request releases a block:

```text
push(block_id)
```

So the allocator is conceptually similar to a pool allocator.

A simplified abstraction is:

```text
allocate():
    if free_blocks is empty:
        return OUT_OF_MEMORY

    return free_blocks.pop()


free(block_id):
    free_blocks.push(block_id)
```

Real inference engines have more sophisticated mechanisms around this, but this is the fundamental idea.

---

# 30. Reference counting and shared blocks

Paged KV systems can become even more interesting when blocks are shared.

Imagine two requests have the same prefix:

```text
System prompt
+
Company documentation
```

They may share the same KV blocks for that prefix.

Conceptually:

```text
Shared prefix

Logical block 0 → Physical block 100
Logical block 1 → Physical block 101
Logical block 2 → Physical block 102
```

Request A:

```text
[100, 101, 102, 150, 151]
```

Request B:

```text
[100, 101, 102, 180, 181]
```

Now physical blocks 100, 101 and 102 are shared.

The allocator needs to know:

```text
reference_count(block)
```

For example:

```text
Block 100 → refcount 2
Block 101 → refcount 2
Block 102 → refcount 2
```

When Request A finishes:

```text
refcount:
2 → 1
```

The blocks remain alive because Request B still uses them.

When Request B finishes:

```text
1 → 0
```

Now they can be returned to the free pool.

This is closely related to prefix caching.

---

# 31. Copy-on-write

Now imagine Request A and Request B initially share:

```text
Block 100
```

Then Request A needs to modify or extend data associated with a shared block.

You cannot simply overwrite Block 100 because Request B still depends on it.

Instead, the system can allocate another physical block:

```text
Block 100 → shared
Block 205 → private copy
```

Request A then points to:

```text
205
```

while Request B continues using:

```text
100
```

This is conceptually similar to **copy-on-write** in operating systems.

Again, the virtual-memory analogy becomes useful.

---

# 32. Why the block table is so important

The block table gives the system an abstraction layer:

```text
Logical sequence
       ↓
Block table
       ↓
Physical GPU memory
```

This abstraction allows the memory manager to make physical decisions without changing the logical sequence.

For example:

```text
Before:

Logical:
[0, 1, 2, 3]

Physical:
[10, 11, 12, 13]
```

Later:

```text
Physical memory pressure
```

could theoretically lead to:

```text
Logical:
[0, 1, 2, 3]

Physical:
[50, 8, 91, 17]
```

The logical sequence hasn't changed.

Only the mapping has changed.

That is an extremely powerful property.

---

# 33. Why this matters for batching

Now imagine 100 active requests.

Without paging, the runtime might have to manage 100 independently sized contiguous KV regions.

With paging:

```text
Global KV block pool

Block 0
Block 1
Block 2
...
Block 99999
```

Every request simply owns some subset of those blocks.

This turns memory management into a common resource-pool problem.

The scheduler can therefore reason about:

```text
available blocks
allocated blocks
requested blocks
released blocks
```

rather than complicated variable-sized contiguous regions.

This makes high-throughput batching much easier.

---

# 34. Continuous batching

Suppose:

```text
Request A → generating token 500
Request B → generating token 20
Request C → generating token 1000
```

Request A may finish now.

Its blocks are released.

At the next scheduling step, another request can use those blocks.

The GPU doesn't have to wait for all requests in a traditional static batch.

This is the foundation of **continuous batching**.

Paged KV allocation and continuous batching work extremely well together because KV memory can be dynamically allocated and released at block granularity.

---

# 35. Why a normal tensor doesn't naturally support this

A standard tensor usually looks conceptually like:

```text
Tensor:

[token 0]
[token 1]
[token 2]
[token 3]
...
```

with contiguous or structured memory.

But PagedAttention introduces:

```text
Logical tensor

        ↓

Block table

        ↓

Multiple physical blocks
```

Therefore, the attention kernel needs to understand the mapping.

This is why PagedAttention isn't merely:

> "Use a different malloc."

It requires changes to the attention execution path.

The kernel must know how to retrieve the right K/V block for each logical sequence position.

---

# 36. The attention computation with pages

Suppose the current Query is:

```text
Q
```

and the sequence contains:

```text
100 tokens
```

with:

```text
16 tokens/block
```

The logical blocks are:

```text
0
1
2
3
4
5
6
```

The block table might be:

```text
[17, 4, 90, 2, 45, 31, 72]
```

The attention kernel processes:

```text
Logical block 0
      ↓
Physical block 17

Logical block 1
      ↓
Physical block 4

Logical block 2
      ↓
Physical block 90

...
```

Each physical block contains K/V for a subset of sequence positions.

The kernel calculates attention across all of them.

Mathematically, the attention result hasn't fundamentally changed.

What changes is how the K/V data is located.

---

# 37. PagedAttention does not change the attention equation

This distinction is important.

PagedAttention isn't a new attention mechanism like:

```text
MHA
MQA
GQA
```

Those change the structure of the attention representation.

PagedAttention is primarily a **memory-management and kernel-execution technique**.

The model still performs the same conceptual attention:

```text
Query
    ↓
compare with Keys
    ↓
attention weights
    ↓
weighted Values
```

PagedAttention changes:

```text
where Keys and Values live
```

and:

```text
how the kernel accesses them
```

---

# 38. PagedAttention versus GQA

These two concepts solve completely different problems.

GQA changes the architecture:

```text
32 Q heads
8 KV heads
```

Therefore:

```text
less KV data exists
```

PagedAttention changes the memory organization:

```text
KV data
↓
fixed-size blocks
↓
non-contiguous physical allocation
```

Therefore:

```text
existing KV data is managed more efficiently
```

They can be combined.

For example:

```text
GQA
↓
128 KiB/token

PagedAttention
↓
organize those KV bytes into blocks
```

So:

```text
GQA = reduce KV size

PagedAttention = manage KV size efficiently
```

This distinction is extremely important.

---

# 39. A complete allocator example

Let's build a small simulation mentally.

Suppose:

```text
Total physical blocks = 10

Block size = 4 tokens
```

Initially:

```text
Free:

0 1 2 3 4 5 6 7 8 9
```

Request A arrives:

```text
10 tokens
```

Required:

```text
ceil(10 / 4)
=
3 blocks
```

Allocate:

```text
0 1 2
```

Mapping:

```text
A → [0, 1, 2]
```

Free:

```text
3 4 5 6 7 8 9
```

Request B arrives:

```text
7 tokens
```

Required:

```text
ceil(7 / 4)
=
2 blocks
```

Allocate:

```text
3 4
```

Mapping:

```text
B → [3, 4]
```

Free:

```text
5 6 7 8 9
```

Request A grows from 10 to 13 tokens.

Current capacity:

```text
3 × 4 = 12
```

Need another block.

Allocate:

```text
5
```

Now:

```text
A → [0, 1, 2, 5]
```

No data movement required.

---

# 40. Now B finishes

Release:

```text
3
4
```

Free blocks:

```text
3 4 6 7 8 9
```

Notice that physical memory looks fragmented:

```text
0 A
1 A
2 A
3 FREE
4 FREE
5 A
6 FREE
7 FREE
8 FREE
9 FREE
```

A contiguous allocator might see:

```text
FREE
FREE
```

and then:

```text
FREE
FREE
FREE
FREE
```

But the paged allocator doesn't care.

A new request needing four blocks can simply receive:

```text
3
4
6
7
```

even though they aren't contiguous.

That's the central win.

---

# 41. Fragmentation becomes bounded

Suppose every request's final block is partially filled.

For each request, the wasted capacity is less than one block.

Therefore, if there are:

```text
N active requests
```

and block size is:

```text
B tokens
```

the total token-slot waste due to final partial blocks is bounded approximately by:

```text
N × (B - 1)
```

in the worst case.

That's much easier to reason about than arbitrary external fragmentation.

For example:

```text
1,000 requests
block size = 16
```

maximum unused token slots from final blocks:

```text
1,000 × 15
=
15,000 token slots
```

Under our example model where:

```text
1 token = 128 KiB
```

that corresponds to:

```text
15,000 × 128 KiB
```

which is approximately:

```text
1.83 GiB
```

This illustrates why block size and request count still matter.

---

# 42. Why smaller blocks aren't automatically better

You might now say:

> "Then let's use one token per block."

That would almost eliminate internal fragmentation.

But now suppose:

```text
1 million tokens
```

are active.

You could have:

```text
1 million blocks
```

The block table becomes enormous.

The GPU kernel also has to perform much more address translation and bookkeeping.

So there is a trade-off:

```text
Smaller blocks
    ↓
Less fragmentation
    ↓
More metadata
    ↓
More indexing overhead


Larger blocks
    ↓
Less metadata
    ↓
Potentially better locality
    ↓
More internal fragmentation
```

The practical block size is therefore a systems optimization.

---

# 43. Memory capacity calculation with blocks

Suppose our GPU has:

```text
24 GiB
```

available for KV cache.

Our model requires:

```text
128 KiB/token
```

and block size is:

```text
16 tokens
```

Each block requires:

```text
16 × 128 KiB
=
2 MiB
```

Now calculate the number of blocks:

```text
24 GiB / 2 MiB
```

Since:

```text
1 GiB = 1024 MiB
```

we have:

```text
24 × 1024 MiB
=
24,576 MiB
```

Then:

```text
24,576 / 2
=
12,288 blocks
```

So approximately:

```text
12,288 KV blocks
```

fit in 24 GiB of raw KV memory.

Again, real systems must reserve memory for other GPU allocations, so you cannot normally dedicate the entire physical VRAM capacity to KV blocks.

---

# 44. Maximum theoretical tokens

With:

```text
12,288 blocks
```

and:

```text
16 tokens/block
```

the theoretical token capacity is:

```text
12,288 × 16
```

which equals:

```text
196,608 tokens
```

So under these simplified assumptions:

```text
24 GiB KV memory
≈ 196K token slots
```

But this does **not** mean you can necessarily serve one 196K-token request.

You might have:

```text
multiple concurrent requests
scheduler overhead
model weights
CUDA allocations
temporary buffers
```

and other constraints.

The block pool is a resource shared among requests.

---

# 45. Why PagedAttention is especially useful for unpredictable workloads

Imagine a chatbot serving users.

One user asks:

```text
"Hi"
```

Another uploads:

```text
500-page document
```

Another has:

```text
a 50-turn conversation
```

Another sends:

```text
a 100K-token context
```

The lengths are unpredictable.

A fixed contiguous allocation strategy struggles with this variability.

Paged allocation naturally adapts:

```text
Short request
→ few blocks

Long request
→ many blocks

Request grows
→ allocate more blocks

Request finishes
→ release blocks
```

This dynamic behavior is exactly what LLM serving requires.

---

# 46. The allocator and scheduler work together

A production inference engine typically needs to answer two questions at every scheduling step.

First:

> Which requests should run?

Second:

> Does the GPU have enough KV blocks to run them?

Suppose:

```text
Free blocks = 100
```

and a new request needs:

```text
30 blocks
```

The scheduler can admit it.

After admission:

```text
Free blocks = 70
```

Now another request needs:

```text
80 blocks
```

The scheduler may have to delay it.

So KV memory becomes a scheduling resource.

This is a major conceptual shift:

> In LLM inference, GPU scheduling isn't only about compute capacity. It is also about KV-cache capacity.

---

# 47. Token generation and block allocation

During decoding, every newly generated token eventually consumes a KV slot.

Suppose:

```text
block size = 16
```

A request has:

```text
16 tokens
```

and therefore:

```text
1 block
```

Generate token 17.

Now:

```text
1 block → full
```

The allocator obtains another block.

The mapping becomes:

```text
[physical_block_7, physical_block_42]
```

Token 17 goes into:

```text
physical_block_42
offset 0
```

Generate token 18:

```text
physical_block_42
offset 1
```

Continue until:

```text
offset 15
```

Then the next token requires another block.

So block allocation happens naturally as the sequence crosses block boundaries.

---

# 48. A useful visualization

Think of the KV cache as a hotel.

A contiguous allocator says:

> "Each guest needs one large room."

That creates problems because guests have different lengths of stay and different room sizes.

PagedAttention says:

> "We have many identical rooms. A guest can occupy several rooms, and the rooms don't have to be adjacent."

So:

```text
Guest A → Room 3 → Room 12 → Room 20

Guest B → Room 1 → Room 5

Guest C → Room 2 → Room 8 → Room 14 → Room 27
```

The guest still experiences one logical stay.

The hotel manages physical rooms independently.

That is essentially what the block table does for the KV cache.

---

# 49. PagedAttention in the complete inference architecture

Now connect it with everything we've discussed.

The complete pipeline looks roughly like:

```text
                    User request
                         |
                         ↓
                    Tokenization
                         |
                         ↓
                     Prefill
                         |
                         ↓
                 Generate K / V
                         |
                         ↓
              KV block allocator
                         |
              ┌──────────┴──────────┐
              ↓                     ↓
       Logical blocks        Physical blocks
              ↓                     ↓
         Block table  ─────────→ GPU KV pool
                                      |
                                      ↓
                               Attention kernel
                                      |
                                      ↓
                                  Next token
                                      |
                                      ↓
                              Allocate new block
                              when necessary
```

The architecture is therefore not simply:

```text
Transformer → GPU
```

A production LLM server looks more like:

```text
Requests
   ↓
Scheduler
   ↓
KV-cache manager
   ↓
Block allocator
   ↓
Block tables
   ↓
Attention kernels
   ↓
GPU
```

---

# 50. The three concepts you should now keep separate

At this point, it is useful to separate three concepts.

**GQA** answers:

> How many K/V representations do we create?

For example:

```text
32 Q heads
8 KV heads
```

**KV caching** answers:

> Should we store previous K/V representations instead of recomputing them?

Answer:

```text
Yes.
```

**PagedAttention** answers:

> How should those cached K/V representations be physically stored and accessed efficiently?

Answer:

```text
Fixed-size physical blocks + logical-to-physical block mapping.
```

So:

```text
GQA
↓
reduces amount of KV data


KV cache
↓
stores KV data for reuse


PagedAttention
↓
manages that data efficiently
```

These concepts operate at different levels.

---

# 51. The deeper architectural connection

Now we can connect the previous topics:

```text
MHA
 ↓
Large number of KV heads
 ↓
Large KV cache


GQA
 ↓
Fewer KV heads
 ↓
Smaller KV cache


PagedAttention
 ↓
Divide KV cache into blocks
 ↓
Avoid large contiguous allocations
 ↓
Reduce external fragmentation
 ↓
Efficient dynamic allocation


Continuous batching
 ↓
Reuse freed blocks immediately
 ↓
Increase GPU utilization


Prefix caching
 ↓
Share identical KV blocks
 ↓
Avoid recomputing common prefixes
```

This is why modern LLM inference is increasingly a combination of **model architecture + memory management + scheduling + specialized GPU kernels**.

---

# 52. The most important fragmentation equations

There are four equations worth remembering.

The number of blocks for a request is:

```text
number_of_blocks
=
ceil(sequence_length / block_size)
```

The allocated token capacity is:

```text
allocated_capacity
=
ceil(sequence_length / block_size)
×
block_size
```

Internal fragmentation is:

```text
internal_fragmentation
=
allocated_capacity
-
sequence_length
```

The maximum internal fragmentation per request is:

```text
block_size - 1
```

The approximate average internal fragmentation under uniformly distributed sequence endings is:

```text
block_size / 2
```

These equations let you reason about the allocator without needing to know the implementation.

---

# 53. One complete numerical example

Let's put everything together.

Assume:

```text
Model:

32 layers
8 KV heads
128 head dimension
BF16
```

We already calculated:

```text
KV = 128 KiB/token
```

Now choose:

```text
block size = 16 tokens
```

Therefore:

```text
one block
=
16 × 128 KiB
=
2 MiB
```

Suppose we have four requests:

```text
A = 100 tokens
B = 500 tokens
C = 1,000 tokens
D = 10,000 tokens
```

Block requirements are:

```text
A:
ceil(100 / 16) = 7

B:
ceil(500 / 16) = 32

C:
ceil(1000 / 16) = 63

D:
ceil(10000 / 16) = 625
```

Total:

```text
7 + 32 + 63 + 625
=
727 blocks
```

Total allocated capacity:

```text
727 × 16
=
11,632 token slots
```

Actual tokens:

```text
100 + 500 + 1000 + 10000
=
11,600
```

Internal waste:

```text
11,632 - 11,600
=
32 token slots
```

Because each request wastes some capacity in its final block.

Now convert that waste to memory:

```text
32 × 128 KiB
=
4,096 KiB
=
4 MiB
```

So across these four requests, the final partial blocks waste approximately:

```text
4 MiB
```

of KV capacity.

That's a concrete example of the fragmentation behavior.

---

# 54. Why the allocator can be extremely efficient

Notice what happened when the requests grew.

Request D might go:

```text
10,000
10,001
10,002
...
10,015
10,016
```

From:

```text
10,000 → 10,015
```

no additional physical block is needed.

At:

```text
10,016
```

the allocator obtains exactly one additional block.

There is no need to move the previous:

```text
625 blocks
```

of KV data.

That's the critical efficiency improvement.

---

# 55. What PagedAttention does not solve

PagedAttention doesn't magically eliminate all inference bottlenecks.

You can still run out of:

```text
GPU memory
```

You can still have:

```text
memory-bandwidth bottlenecks
```

You can still have:

```text
attention compute bottlenecks
```

You can still have:

```text
scheduler overhead
```

You can still have:

```text
CUDA kernel launch overhead
```

And long contexts still require substantial KV memory.

PagedAttention simply makes the available KV memory much easier to use efficiently.

---

# 56. Production-level mental model

When designing an LLM serving system, think about the layers of optimization in this order.

At the model architecture level:

```text
MHA → GQA/MQA
```

reduces the amount of KV state.

At the representation level:

```text
BF16 → FP8/other KV quantization
```

can reduce bytes per KV element.

At the memory-management level:

```text
Contiguous allocation → Paged KV blocks
```

reduces external fragmentation and enables dynamic allocation.

At the scheduling level:

```text
Static batching → Continuous batching
```

improves GPU utilization.

At the caching level:

```text
Repeated prefix → Prefix KV reuse
```

avoids repeated prefill computation.

At the hardware level:

```text
One GPU → Multi-GPU parallelism
```

distributes computation and memory.

These layers attack different bottlenecks.

---

# 57. The entire concept in one diagram

```text
                         LLM REQUESTS
                              |
                              ↓
                         Token sequence
                              |
                              ↓
                       Transformer layers
                              |
                              ↓
                         K/V generation
                              |
                              ↓
                     ┌─────────────────┐
                     │   KV CACHE      │
                     │                 │
                     │ Logical tokens  │
                     └────────┬────────┘
                              |
                              ↓
                       Divide into blocks
                              |
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
          Block 0         Block 1         Block 2
              |               |               |
              ↓               ↓               ↓
        Physical 17      Physical 4       Physical 91
              \               |               /
               \              |              /
                └────── Block Table ─────────┘
                              |
                              ↓
                       Attention kernel
                              |
                              ↓
                          Next token
                              |
                              ↓
                    Need another block?
                       /             \
                     No               Yes
                     |                 |
                     ↓                 ↓
                  Continue       Allocate block
                                      |
                                      ↓
                                Update mapping
```

The central idea is:

> **The sequence is logically contiguous, but its KV cache does not need to be physically contiguous.**

That single sentence captures PagedAttention.

---

# 58. Final mental model

You can now connect the whole inference story:

```text
MHA
32 Q → 32 KV
        ↓
Large KV cache


GQA
32 Q → 8 KV
        ↓
4 Q heads share each KV head
        ↓
4× smaller KV cache


KV Cache
        ↓
Store K/V so they aren't recomputed


PagedAttention
        ↓
Divide KV cache into fixed-size blocks
        ↓
Allocate blocks dynamically
        ↓
Map logical blocks to physical blocks
        ↓
Avoid contiguous allocation
        ↓
Reduce external fragmentation


Continuous batching
        ↓
Requests enter and leave dynamically
        ↓
Freed blocks are immediately reusable
```

The most important mathematical relationship is:

```text
blocks needed
=
ceil(tokens / tokens_per_block)
```

and the most important fragmentation relationship is:

```text
wasted token slots
=
ceil(tokens / block_size) × block_size
-
tokens
```

while the maximum waste for one request is:

```text
block_size - 1
```

The deeper reason PagedAttention matters is that **LLM inference has a fundamentally dynamic memory workload**. Every generated token grows a request's KV state, different requests have different lengths, requests terminate asynchronously, and multiple users compete for the same GPU memory. A contiguous allocator is poorly matched to that workload.

PagedAttention transforms the problem from:

```text
"Find a sufficiently large contiguous region of GPU memory."
```

into:

```text
"Find enough free fixed-size KV blocks."
```

That seemingly small change is profound. It gives the inference engine a pool of reusable KV blocks, allows requests to grow without moving their existing cache, bounds internal fragmentation, largely eliminates the need to reason about external fragmentation, and provides the memory-management foundation on which high-throughput serving systems can build **continuous batching, prefix sharing, and efficient long-context inference**.

The natural next step after this is **continuous batching + PagedAttention scheduling**, where we can derive how an inference engine decides which requests to admit, how many KV blocks each request needs at every decoding step, and how **tokens/sec, batch size, KV capacity, TTFT, and TPOT** interact mathematically.
