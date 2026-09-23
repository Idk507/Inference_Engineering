# The Inference Lifecycle and Attention Math

## A First-Principles Journey From Prompt to Next Token

Large language model inference is often presented as if it were a single
operation: send a prompt to a model and receive text back. Internally,
however, inference is a carefully structured numerical pipeline.
Human-readable text is first converted into discrete token IDs, those
IDs become vectors, vectors pass through repeated Transformer blocks,
self-attention computes how each position should use information from
other positions, the model produces logits over the vocabulary, and a
decoding policy turns those logits into the next token. The process then
repeats until a stopping condition is reached.

The important idea for inference engineering is that the model does not
directly manipulate words. It manipulates tensors. A prompt is
transformed into integer IDs, integer IDs are transformed into
embeddings, and almost every operation after that is matrix
multiplication, normalization, elementwise transformation, or attention.
Understanding inference therefore means understanding the movement and
transformation of these tensors, as well as understanding which parts
are dominated by computation and which parts are dominated by memory
movement.

This chapter follows one request through the complete decoder-only
Transformer inference lifecycle. The discussion deliberately goes below
API-level abstractions so that a `generate()` call can be mentally
expanded into the actual operations performed by the model.

------------------------------------------------------------------------

## 1. The Complete Inference Lifecycle

Consider the prompt:

``` text
The capital of France is
```

A typical decoder-only language model processes this request
approximately as follows.

``` text
Raw text
   |
   v
Tokenizer
   |
   v
Token IDs
   |
   v
Token embeddings + positional information
   |
   v
Transformer block 1
   |
   v
Transformer block 2
   |
   v
...
   |
   v
Transformer block L
   |
   v
Final normalization
   |
   v
Language-model head
   |
   v
Logits over vocabulary
   |
   v
Temperature / top-k / top-p
   |
   v
Sample or select next token
   |
   v
Append token to sequence
   |
   v
Check stopping criteria
   |
   +------ not stopped ------+
   |                          |
   +--------------------------+
```

The first pass through the prompt is called **prefill**. Once the model
has processed the existing context, generation proceeds token by token
through the **decode** phase. This distinction is fundamental to
inference optimization because prefill and decode have different
computational characteristics.

Prefill is highly parallel. If the prompt contains 1,000 tokens, the
model can process the representations of those 1,000 positions together.
Decode is inherently sequential at the generation level because token
number 1,001 must be generated before token number 1,002 can be
generated.

The model therefore has two different performance regimes. Prefill is
commonly associated with compute-heavy matrix operations and prompt
processing latency. Decode is commonly dominated by repeatedly executing
the model while reading model parameters and the growing key-value cache
from memory. This difference explains why optimizing only FLOPs is
insufficient for modern LLM inference.

------------------------------------------------------------------------

# 2. Tokenization: Turning Text Into Integers

A Transformer does not receive a Python string as its mathematical
input. It receives integer token IDs.

Suppose a tokenizer converts:

``` text
The capital of France is
```

into something conceptually similar to:

``` text
[464, 3139, 286, 4881, 318]
```

The exact numbers depend entirely on the tokenizer vocabulary. These
integers are not semantic values in the mathematical sense. Token ID
4881 is not inherently "more meaningful" than token ID 318. They are
simply indices into a learned vocabulary.

A tokenizer maintains a fixed vocabulary. If the vocabulary contains
50,000 entries, token IDs normally range from 0 through 49,999. The
vocabulary is created during tokenizer training and then associated with
the model. The model's embedding matrix therefore has a corresponding
row for each vocabulary entry.

If the model embedding dimension is 768 and the vocabulary size is
50,000, the token embedding matrix has a conceptual shape of:

``` text
[50000, 768]
```

If token ID 4881 appears in the input, the model retrieves row 4881 from
this matrix.

The important point is that tokenization and embedding are separate
stages. Tokenization converts text into discrete symbols. Embedding
converts those symbols into dense vectors.

------------------------------------------------------------------------

## 2.1 Why Subword Tokenization Exists

A word-level tokenizer would require a vocabulary containing every word
the model might encounter. That is impractical because natural language
has an enormous number of words, names, technical terms, misspellings,
morphological variants, and newly created expressions.

Character-level tokenization has the opposite problem. The vocabulary
becomes small, but sequences become much longer. A 100-character
document could become roughly 100 tokens.

Subword tokenization provides a compromise.

A tokenizer can represent common words as single tokens while
decomposing uncommon words into smaller pieces. Conceptually:

``` text
playing
```

might become:

``` text
play + ing
```

while a common word such as:

``` text
the
```

may remain a single token.

The exact behavior depends on the tokenizer algorithm and vocabulary.

BPE, or Byte Pair Encoding, builds a vocabulary by repeatedly merging
frequently occurring symbol pairs. The training process begins with
small units and learns combinations that occur frequently enough to
justify representing them as larger units. The result is a finite
vocabulary capable of representing arbitrary text through combinations
of known pieces.

The fixed-size vocabulary is important for the neural architecture
because the final language-model head produces one logit per vocabulary
entry. If the vocabulary size is V, the final output for one position
has shape:

``` text
[V]
```

For a vocabulary of 50,000 tokens, the model produces 50,000 logits
before sampling the next token.

------------------------------------------------------------------------

# 3. From Token IDs to Vectors

Once the prompt has become token IDs, each ID is mapped to a dense
vector.

Suppose the token sequence has length n and the model dimension is
d_model. The embedding output has shape:

``` text
[n, d_model]
```

For a batch of B prompts, the shape becomes:

``` text
[B, n, d_model]
```

A decoder-only Transformer also needs positional information because
self-attention by itself does not inherently know whether a token
appeared first, second, or tenth.

Different model families encode positional information differently.
Classical Transformers use explicit positional encodings. Modern
decoder-only architectures frequently use techniques such as Rotary
Position Embeddings, commonly called RoPE.

The exact positional mechanism varies, but the conceptual requirement is
the same: the model needs a representation of token order.

------------------------------------------------------------------------

# 4. The Transformer Block

A simplified decoder-only Transformer block can be thought of as two
major learned transformations.

The first is self-attention. It allows each token position to gather
information from other positions.

The second is the feed-forward network, often called the MLP or FFN. It
transforms each token representation independently after attention has
mixed information across positions.

A simplified conceptual flow is:

``` text
Input
  |
  +--> Normalization
  |
  +--> Self-Attention
  |
  +--> Residual connection
  |
  +--> Normalization
  |
  +--> Feed-Forward Network
  |
  +--> Residual connection
  |
  v
Output
```

Modern architectures vary in ordering and details. Some use
pre-normalization, RMSNorm, SwiGLU, grouped-query attention, multi-query
attention, and other modifications. The central attention mechanism,
however, can be understood from the original scaled dot-product
formulation.

The original Transformer paper defines attention as:

``` text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

The equation is compact, but every symbol represents an important tensor
operation. The rest of this section expands it from first principles.

The original paper introduced scaled dot-product attention and
multi-head attention as the core mechanisms of the Transformer
architecture. It also explains why scaling by the square root of the key
dimension is useful. See the original paper for the canonical
formulation. [Attention Is All You
Need](https://arxiv.org/abs/1706.03762)

------------------------------------------------------------------------

# 5. Query, Key, and Value

Let the input sequence representation be:

``` text
X
```

with shape:

``` text
[n, d_model]
```

The model creates three different representations:

``` text
Q = X W_Q
K = X W_K
V = X W_V
```

Here:

``` text
W_Q = query projection matrix
W_K = key projection matrix
W_V = value projection matrix
```

For a simple single-head example, assume:

``` text
X      = [n, d_model]
W_Q    = [d_model, d_k]
W_K    = [d_model, d_k]
W_V    = [d_model, d_v]
```

Then:

``` text
Q = [n, d_k]
K = [n, d_k]
V = [n, d_v]
```

The query represents what the current token is looking for.

The key represents what information each token can be matched against.

The value represents the actual information that can be retrieved.

These descriptions are conceptual rather than literal symbolic meanings.
The network learns the projections during training.

------------------------------------------------------------------------

# 6. Where the n² Term Comes From

This is the central checkpoint question.

Suppose there are n tokens.

Each token produces one query vector.

Each query needs a compatibility score with every key.

Therefore:

``` text
n queries x n keys = n² query-key comparisons
```

This is exactly where the quadratic term originates.

The matrix multiplication:

``` text
Q K^T
```

has shapes:

``` text
Q     = [n, d_k]
K^T   = [d_k, n]
```

so the result is:

``` text
[n, n]
```

That resulting n-by-n matrix contains one attention score for every
query-key pair.

For every output position, the model asks:

``` text
How relevant is token 1?
How relevant is token 2?
How relevant is token 3?
...
How relevant is token n?
```

There are n questions for each of n query positions.

Therefore:

``` text
n x n = n²
```

This is the exact structural reason self-attention has quadratic
sequence-length scaling.

The dimension d_k contributes to the amount of arithmetic required for
each query-key dot product, but the number of pairwise interactions
comes from n².

------------------------------------------------------------------------

# 7. Deriving the Attention Compute Cost

Consider:

``` text
Q = [n, d_k]
K = [n, d_k]
```

Computing:

``` text
Q K^T
```

requires approximately:

``` text
n x n x d_k
```

multiply-accumulate work.

Therefore the dominant score computation scales as:

``` text
O(n² d_k)
```

Then attention applies the resulting weights to V:

``` text
A V
```

where:

``` text
A = [n, n]
V = [n, d_v]
```

This requires approximately:

``` text
n x n x d_v
```

operations.

Therefore the attention portion is approximately:

``` text
O(n² d_k + n² d_v)
```

If the query, key, and value dimensions are all proportional to the
model dimension, this is commonly summarized as:

``` text
O(n² d_model)
```

The important distinction is that the quadratic factor comes from
sequence interactions, while the hidden dimension determines how
expensive each interaction is.

------------------------------------------------------------------------

# 8. Why Divide by sqrt(d_k)?

The raw query-key score is a dot product:

``` text
q · k
```

Suppose the components of q and k are independent random variables with
mean zero and variance one.

The dot product is:

``` text
q1*k1 + q2*k2 + ... + q_dk*k_dk
```

Each product has approximately unit variance under the simplifying
assumptions.

Adding d_k such terms makes the variance grow approximately with d_k.

Therefore:

``` text
Var(q · k) approximately proportional to d_k
```

and the standard deviation grows approximately as:

``` text
sqrt(d_k)
```

As d_k becomes larger, raw dot products can therefore become numerically
larger in magnitude.

Those values are then passed into softmax.

Softmax is:

``` text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

If one logit becomes much larger than the others, softmax approaches a
nearly one-hot distribution.

For example:

``` text
logits = [1, 2, 3]
```

produce a reasonably distributed probability vector.

But:

``` text
logits = [10, 20, 30]
```

produce a much sharper distribution.

Very sharp softmax outputs can have extremely small gradients for many
positions, making optimization more difficult.

The Transformer therefore divides the dot product by:

``` text
sqrt(d_k)
```

so that the variance of the scaled score remains approximately stable as
d_k changes.

Conceptually:

``` text
raw score magnitude grows with sqrt(d_k)
                  |
                  v
divide by sqrt(d_k)
                  |
                  v
more stable score scale
                  |
                  v
healthier softmax behavior
```

This is not an arbitrary numerical trick. It follows from the
statistical scale of the dot product.

The original Transformer paper explicitly motivates this scaling by
noting that large dot products can push softmax into regions with
extremely small gradients. [Attention Is All You
Need](https://arxiv.org/abs/1706.03762)

------------------------------------------------------------------------

# 9. Softmax Converts Scores Into Weights

After computing:

``` text
S = Q K^T / sqrt(d_k)
```

the model applies softmax independently to each query row.

If S has shape:

``` text
[n, n]
```

then the attention weights also have shape:

``` text
[n, n]
```

Each row sums approximately to one.

For a particular query position i:

``` text
attention_weights_i =
[weight_for_token_1,
 weight_for_token_2,
 ...
 weight_for_token_n]
```

The model then computes a weighted combination of values.

Conceptually:

``` text
output_i =
weight_1 * value_1
+ weight_2 * value_2
+ ...
+ weight_n * value_n
```

Therefore attention is fundamentally a learned weighted information
retrieval mechanism.

The query determines what the current position is looking for. Keys
determine how strongly each available token matches that request. Values
contain the information that gets aggregated.

------------------------------------------------------------------------

# 10. Causal Masking

A decoder-only language model cannot allow token position i to look at
future tokens.

Suppose the sequence is:

``` text
The cat sat
```

When predicting the token after "The", the model cannot use "cat" or
"sat" because those tokens would not exist yet during generation.

A causal mask therefore blocks future positions.

Conceptually, the allowed attention pattern is:

``` text
token 1 -> token 1
token 2 -> token 1, token 2
token 3 -> token 1, token 2, token 3
token 4 -> token 1, token 2, token 3, token 4
```

The attention score matrix can be visualized as:

``` text
[ allowed  blocked  blocked  blocked ]
[ allowed  allowed  blocked  blocked ]
[ allowed  allowed  allowed  blocked ]
[ allowed  allowed  allowed  allowed ]
```

Blocked entries are typically assigned a very large negative value
before softmax, conceptually:

``` text
score = -infinity
```

Then:

``` text
exp(-infinity) = 0
```

so the corresponding attention probability becomes zero.

Causal masking changes which entries are usable, but it does not remove
the fundamental n-by-n structure of the dense attention calculation.
This distinction becomes important when thinking about optimization.

------------------------------------------------------------------------

# 11. Multi-Head Attention

A Transformer does not normally rely on one attention operation.

Instead, the model splits the representation into multiple attention
heads.

Suppose:

``` text
d_model = 768
number_of_heads = 12
```

Then a common head dimension is:

``` text
d_head = 768 / 12 = 64
```

Each head independently constructs its own Q, K, and V representations.

Conceptually:

``` text
Input
  |
  +--> Head 1
  +--> Head 2
  +--> Head 3
  ...
  +--> Head 12
  |
  v
Concatenate
  |
  v
Output projection
```

The intuition is that different heads can learn different interaction
patterns. One head may become useful for local relationships, another
may track syntactic structure, and another may capture long-range
dependencies. These interpretations should be treated as learned
behavior rather than hard-coded semantic roles.

Mathematically, each head computes its own scaled attention operation:

``` text
Head_i = softmax(Q_i K_i^T / sqrt(d_head)) V_i
```

The outputs are concatenated and passed through an output projection.

An important inference-engineering observation is that splitting into
heads does not eliminate the quadratic sequence interaction. Across all
heads, the total attention work remains approximately proportional to:

``` text
O(n² d_model)
```

because the head dimensions collectively sum to approximately the model
dimension.

------------------------------------------------------------------------

# 12. Prefill: Processing the Prompt

Now we can connect the attention mathematics to actual inference.

Suppose the prompt contains 2,000 tokens.

During prefill, the model processes those 2,000 positions together.

Conceptually:

``` text
tokens = [t1, t2, t3, ..., t2000]
```

become:

``` text
X = [x1, x2, x3, ..., x2000]
```

The Transformer can compute Q, K, and V for the entire sequence using
large matrix operations.

This is highly parallelizable on GPUs.

The GPU is extremely good at operations such as:

``` text
matrix multiplication
matrix addition
elementwise transformation
normalization
```

Therefore prefill tends to expose substantial computational parallelism.

------------------------------------------------------------------------

# 13. Why Prefill Is Often Compute-Bound

A simplified linear projection looks like:

``` text
X W
```

If:

``` text
X = [n, d_model]
W = [d_model, d_out]
```

then the result is:

``` text
[n, d_out]
```

The operation requires roughly:

``` text
n * d_model * d_out
```

multiply-accumulate operations.

When n is large, the GPU receives a large amount of arithmetic work that
can be executed in parallel.

The attention score computation also performs large matrix
multiplications:

``` text
Q K^T
```

and:

``` text
A V
```

These are dense operations with substantial arithmetic intensity.

Consequently, during sufficiently large prefill workloads, the GPU can
spend much of its time performing mathematical operations rather than
waiting for individual model weights to arrive from memory.

This is why prompt processing is often described as compute-bound,
although the exact bottleneck depends on model size, hardware, sequence
length, batch size, precision, kernel implementation, and system
configuration.

------------------------------------------------------------------------

# 14. Decode: Generating One Token at a Time

After prefill, suppose the model predicts:

``` text
Paris
```

The sequence becomes:

``` text
The capital of France is Paris
```

Now the model must generate the next token.

It cannot simultaneously know token number N+1 before token number N has
been selected because the next token depends on the sequence that
includes the previous generated token.

Therefore generation proceeds:

``` text
step 1 -> token A
step 2 -> token B
step 3 -> token C
step 4 -> token D
...
```

This sequential dependency is fundamentally different from the parallel
computation inside one forward pass.

------------------------------------------------------------------------

# 15. The Key-Value Cache

Naively recomputing attention for the entire sequence at every decoding
step would be extremely wasteful.

Suppose the current sequence contains:

``` text
1,000 tokens
```

and we generate token 1,001.

The keys and values associated with the first 1,000 tokens have already
been computed.

There is no reason to recompute them from scratch.

The model therefore stores previously computed keys and values in a
structure called the **KV cache**.

At the next decoding step, the model computes:

``` text
Q_new
K_new
V_new
```

for the newly generated token.

The new key and value are appended to the cache:

``` text
K_cache = [K_old, K_new]
V_cache = [V_old, V_new]
```

The new query then attends over all cached keys:

``` text
Q_new K_cache^T
```

and uses the corresponding cached values:

``` text
softmax(Q_new K_cache^T / sqrt(d_k)) V_cache
```

The query length is now effectively one, while the key/value sequence
length continues growing.

This changes the practical compute pattern of attention during decoding.

The model still has to inspect an increasing context, but it avoids
repeatedly reconstructing all previous K and V projections.

------------------------------------------------------------------------

# 16. Why Decode Can Become Memory-Bound

Modern GPUs can perform enormous numbers of arithmetic operations per
second, but model weights and KV-cache data must still be moved through
the memory hierarchy.

During autoregressive decoding, the batch and token dimensions can be
small. A single new token may not provide enough parallel arithmetic
work to fully utilize the GPU's compute capacity.

The system repeatedly reads model parameters, performs matrix operations
for one token, reads the relevant KV-cache data, computes the next
distribution, and repeats.

This can create a memory-bandwidth bottleneck.

The distinction can be summarized conceptually:

``` text
Prefill:
large token dimension
large matrix operations
high parallelism
often compute-heavy

Decode:
one new token at a time
growing KV cache
repeated parameter reads
often memory-sensitive
```

This is one of the most important ideas in inference engineering. "The
model is fast" is not a sufficient description. You need to know whether
you are measuring prefill throughput, decode throughput,
time-to-first-token, inter-token latency, or total request latency.

------------------------------------------------------------------------

# 17. The Language Model Head

After the final Transformer layer, the model produces a hidden
representation for each position.

For the final position, suppose:

``` text
h = [d_model]
```

The language-model head maps this vector into vocabulary space.

Conceptually:

``` text
logits = h W_vocab
```

If:

``` text
W_vocab = [d_model, vocab_size]
```

then:

``` text
logits = [vocab_size]
```

Every vocabulary token gets one score.

For example:

``` text
Paris     -> 8.2
London    -> 5.7
Berlin    -> 4.9
banana    -> -2.3
...
```

These are logits, not probabilities.

The model has not yet selected a token.

------------------------------------------------------------------------

# 18. Why the Model Outputs a Distribution

A language model estimates something conceptually similar to:

``` text
P(next_token | previous_tokens)
```

The logits are converted into probabilities using softmax.

For logits:

``` text
z = [z1, z2, ..., zV]
```

softmax produces:

``` text
P(token_i) = exp(z_i) / sum_j exp(z_j)
```

The result is a probability distribution over the vocabulary.

For example:

``` text
Paris   0.72
London  0.11
Berlin  0.07
Rome    0.04
Other   0.06
```

The model is therefore not inherently saying:

``` text
The answer is Paris.
```

It is saying something closer to:

``` text
Given the context, these are the relative probabilities of possible next tokens.
```

The decoding strategy decides how that distribution is converted into an
actual token.

------------------------------------------------------------------------

# 19. Greedy Decoding

Greedy decoding chooses the highest-probability token.

Conceptually:

``` text
next_token = argmax(logits)
```

If:

``` text
Paris   0.72
London  0.11
Berlin  0.07
```

then:

``` text
Paris
```

is selected.

Greedy decoding is deterministic when the model and numerical execution
are deterministic.

Its weakness is that selecting the locally highest-probability token can
sometimes produce less desirable global sequences. Autoregressive
language generation is a sequence-level process, while greedy decoding
makes a local decision at each step.

------------------------------------------------------------------------

# 20. Temperature

Temperature modifies the sharpness of the probability distribution.

The temperature-adjusted logits are:

``` text
z_scaled = z / T
```

where T is temperature.

Then softmax is applied.

When:

``` text
T = 1
```

the original distribution is preserved.

When:

``` text
T < 1
```

the distribution becomes sharper.

When:

``` text
T > 1
```

the distribution becomes flatter.

Conceptually:

``` text
Low temperature:
one or a few tokens dominate

High temperature:
probability spreads across more tokens
```

Temperature does not change the underlying model weights. It changes the
sampling distribution produced from the model's logits.

------------------------------------------------------------------------

# 21. Top-k Sampling

Top-k sampling first identifies the k highest-logit tokens.

Suppose:

``` text
vocabulary size = 50,000
k = 50
```

Instead of sampling from all 50,000 tokens, the decoder keeps only the
50 highest-scoring candidates.

The remaining logits are masked out.

The surviving logits are then converted into probabilities and sampled.

Top-k therefore imposes a hard candidate-count constraint.

The candidate pool has exactly k tokens, assuming at least k vocabulary
entries exist.

------------------------------------------------------------------------

# 22. Top-p Sampling

Top-p sampling, also called nucleus sampling, uses probability mass
instead of a fixed candidate count.

Suppose the probabilities after softmax are:

``` text
0.50
0.20
0.12
0.08
0.04
0.03
0.02
0.01
...
```

If:

``` text
p = 0.90
```

we sort probabilities from largest to smallest and keep tokens until
cumulative probability reaches at least 0.90.

The candidate set might therefore contain:

``` text
0.50 + 0.20 + 0.12 + 0.08 = 0.90
```

so four candidates survive in this simplified example.

Unlike top-k, top-p does not specify the number of candidates directly.

A confident distribution may reach the probability threshold quickly.

A flat distribution may require many candidates.

This is why top-p dynamically changes the candidate pool.

------------------------------------------------------------------------

# 23. The Important Top-p Confidence Experiment

Your exercise asks for a particularly useful experiment.

Start with logits:

``` text
[4.0, 3.0, 2.0, 1.0, 0.0]
```

Apply softmax.

Then artificially increase confidence by scaling logits:

``` text
logits_confident = logits * 3
```

The relative differences become larger.

Softmax becomes sharper.

The cumulative probability reaches a fixed p threshold using fewer
tokens.

Now flatten the distribution:

``` text
logits_flat = logits * 0.3
```

The probability mass becomes more evenly distributed.

The cumulative probability reaches p only after including more tokens.

Therefore:

``` text
larger logit scale
        ->
sharper softmax
        ->
smaller top-p candidate pool

smaller logit scale
        ->
flatter softmax
        ->
larger top-p candidate pool
```

This is an excellent sanity check because it validates that your
implementation understands top-p as cumulative probability filtering
rather than simply selecting a fixed number of tokens.

------------------------------------------------------------------------

# 24. Why Top-p Must Sort Probabilities

Top-p requires cumulative probability mass.

Therefore the probabilities must be ordered from largest to smallest
before the cumulative sum is computed.

Conceptually:

``` text
probabilities
     |
     v
sort descending
     |
     v
cumulative sum
     |
     v
find first position where cumulative mass >= p
     |
     v
mask everything beyond that boundary
```

One subtle implementation issue is that the token that crosses the
threshold must usually be retained. Otherwise the cumulative mass of the
retained set could remain below p.

Another subtle issue is that the original token indices must be restored
after sorting. Sorting gives you probabilities in ranked order, but the
model's logits remain associated with vocabulary token IDs.

------------------------------------------------------------------------

# 25. Manual Top-p Implementation With PyTorch

The following implementation deliberately avoids a library top-p helper.
It uses tensor operations directly.

``` python
import torch


def top_p_filter(
    logits: torch.Tensor,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering to a single vocabulary logit vector.

    The function sorts tokens by probability, computes cumulative probability
    mass, and masks tokens that fall outside the smallest prefix whose mass
    reaches the requested top_p threshold.

    Args:
        logits: A one-dimensional tensor with shape [vocab_size].
        top_p: Probability mass to retain. Must be in the interval (0, 1].

    Returns:
        A tensor with the same shape as logits. Filtered tokens are assigned
        negative infinity so they receive zero probability after softmax.

    Raises:
        ValueError: If logits is not one-dimensional or top_p is invalid.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional tensor.")

    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1].")

    probabilities = torch.softmax(logits, dim=-1)

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True,
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    # Keep the token that crosses the top_p threshold.
    sorted_mask = cumulative_probabilities > top_p

    if sorted_mask.numel() > 1:
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False

    filtered_logits = logits.clone()

    tokens_to_remove = sorted_indices[sorted_mask]
    filtered_logits[tokens_to_remove] = float("-inf")

    return filtered_logits
```

The key operation is:

``` python
cumulative_probabilities = torch.cumsum(
    sorted_probabilities,
    dim=-1,
)
```

This is the mathematical implementation of nucleus filtering.

The line:

``` python
sorted_mask[1:] = sorted_mask[:-1].clone()
```

is important because it preserves the first token whose inclusion causes
the cumulative probability to cross the threshold.

------------------------------------------------------------------------

# 26. Manual Temperature Scaling

Temperature should be applied to logits before softmax.

``` python
import torch


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Scale logits by temperature before probability conversion.

    Args:
        logits: Tensor containing model logits.
        temperature: Positive temperature value. Values below one sharpen
            the distribution; values above one flatten it.

    Returns:
        Temperature-adjusted logits.

    Raises:
        ValueError: If temperature is not strictly positive.
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    return logits / temperature
```

Do not apply temperature directly to probabilities using the same
formula. Temperature is mathematically defined as a transformation of
the logits before softmax.

------------------------------------------------------------------------

# 27. Manual Sampling

Once the logits have been temperature-scaled and top-p filtered, the
decoder can convert them into probabilities and sample one token.

``` python
import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """
    Sample one token from a vocabulary distribution.

    Args:
        logits: One-dimensional vocabulary logits.
        temperature: Positive sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        The selected token ID as a Python integer.

    Raises:
        ValueError: If logits has the wrong shape or sampling parameters
            are invalid.
    """
    if logits.ndim != 1:
        raise ValueError("logits must have shape [vocab_size].")

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1].")

    scaled_logits = logits / temperature

    filtered_logits = top_p_filter(
        scaled_logits,
        top_p=top_p,
    )

    probabilities = torch.softmax(
        filtered_logits,
        dim=-1,
    )

    next_token = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return int(next_token.item())
```

The important distinction is that `torch.multinomial` is only being used
for the final random draw. The temperature and top-p transformations
themselves are explicitly implemented.

For an even stricter educational implementation, the random draw can be
implemented by generating one random number and manually walking through
the cumulative distribution.

------------------------------------------------------------------------

# 28. A Completely Explicit Sampling Draw

The mathematical sampling process is straightforward.

Suppose:

``` text
P = [0.1, 0.2, 0.5, 0.2]
```

The cumulative distribution is:

``` text
[0.1, 0.3, 0.8, 1.0]
```

Generate:

``` text
u ~ Uniform(0, 1)
```

If:

``` text
u = 0.65
```

then the first cumulative probability greater than or equal to 0.65
corresponds to token index 2.

A manual implementation is:

``` python
import torch


def sample_from_probabilities(
    probabilities: torch.Tensor,
) -> int:
    """
    Draw one token from an already normalized probability distribution.

    Args:
        probabilities: One-dimensional tensor whose values are non-negative
            and sum approximately to one.

    Returns:
        Sampled token index as a Python integer.

    Raises:
        ValueError: If the input is malformed or contains invalid values.
    """
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")

    if torch.any(probabilities < 0):
        raise ValueError("probabilities cannot contain negative values.")

    total = probabilities.sum()

    if not torch.isfinite(total) or total <= 0:
        raise ValueError("probabilities must have a positive finite sum.")

    probabilities = probabilities / total

    cumulative = torch.cumsum(
        probabilities,
        dim=-1,
    )

    random_value = torch.rand(
        (),
        device=probabilities.device,
    )

    token_index = torch.searchsorted(
        cumulative,
        random_value,
    )

    return int(token_index.item())
```

This makes the sampling process concrete: the model produces a
distribution, and the decoder turns that distribution into a discrete
random outcome.

------------------------------------------------------------------------

# 29. The Manual Decode Loop

A decoder-only language model repeatedly performs the following logical
process.

``` text
1. Run the model on the current sequence.
2. Extract logits for the final position.
3. Apply temperature.
4. Apply top-p or top-k filtering.
5. Convert logits to probabilities.
6. Sample or select the next token.
7. Append the token.
8. Check stopping conditions.
9. Repeat.
```

A conceptual implementation without calling `model.generate()` is:

``` python
import torch


@torch.no_grad()
def decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Autoregressively generate tokens from a decoder-only language model.

    This implementation intentionally performs the generation loop manually
    instead of calling a framework-level generate helper.

    Args:
        model: Decoder-only language model returning logits.
        input_ids: Tensor with shape [batch_size, sequence_length].
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Positive sampling temperature.
        top_p: Nucleus sampling threshold.
        eos_token_id: Optional token ID that terminates generation.

    Returns:
        Tensor containing the original prompt followed by generated tokens.

    Raises:
        ValueError: If generation parameters are invalid.
    """
    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch_size, sequence_length]."
        )

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1].")

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        outputs = model(generated)

        logits = outputs.logits

        # Only the final position is used to predict the next token.
        next_token_logits = logits[:, -1, :]

        next_token_logits = next_token_logits / temperature

        next_token_logits = torch.stack(
            [
                top_p_filter(row, top_p=top_p)
                for row in next_token_logits
            ],
            dim=0,
        )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        next_tokens = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        generated = torch.cat(
            [generated, next_tokens],
            dim=-1,
        )

        if eos_token_id is not None:
            if torch.all(next_tokens.squeeze(-1) == eos_token_id):
                break

    return generated
```

This version is intentionally educational rather than
production-optimal.

A real production decoder should use the model's KV cache. Otherwise the
implementation repeatedly feeds the entire generated sequence through
every Transformer layer. That causes unnecessary recomputation.

------------------------------------------------------------------------

# 30. A More Realistic KV-Cache Decode Loop

A production-style architecture conceptually looks like this:

``` python
@torch.no_grad()
def decode_with_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Illustrate the conceptual structure of cached autoregressive decoding.

    The exact cache argument names differ between model implementations,
    so this function is intentionally architectural rather than tied to one
    framework API.

    Args:
        model: Decoder-only language model supporting KV caching.
        input_ids: Initial prompt token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Positive temperature.
        top_p: Nucleus probability threshold.
        eos_token_id: Optional EOS token.

    Returns:
        Generated token IDs.
    """
    generated = input_ids.clone()

    # Prefill processes the entire prompt once.
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
    )

    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        logits = outputs.logits[:, -1, :]
        logits = logits / temperature

        filtered = torch.stack(
            [
                top_p_filter(row, top_p=top_p)
                for row in logits
            ],
            dim=0,
        )

        probabilities = torch.softmax(
            filtered,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=-1,
        )

        if eos_token_id is not None:
            if torch.all(next_token.squeeze(-1) == eos_token_id):
                break

        # Decode only the newly generated token while reusing the cache.
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values

    return generated
```

The precise cache API depends on the model implementation. The
architectural idea is what matters: the prompt is processed once, and
each subsequent step receives only the new token while reusing previous
K and V tensors.

------------------------------------------------------------------------

# 31. Stopping Criteria

Generation cannot continue forever.

The simplest stopping condition is the EOS token.

EOS means end of sequence.

If the model generates:

``` text
<eos>
```

the decoder terminates the sequence.

A second common condition is a maximum number of generated tokens:

``` text
max_new_tokens = 256
```

This protects the system against runaway generation.

A third possibility is a stop string or sequence.

For example, an application might request:

``` text
stop = ["</answer>"]
```

The decoder can terminate when the generated text contains the requested
sequence.

Production systems often combine several criteria:

``` text
EOS
OR
maximum output tokens reached
OR
application-specific stop sequence
OR
request cancellation
OR
server timeout
```

The stopping mechanism is part of inference control, not part of the
neural network itself.

------------------------------------------------------------------------

# 32. Prefill and Decode as Two Different Performance Problems

The entire inference request can therefore be divided into:

``` text
Prefill:
prompt -> KV cache + first next-token distribution

Decode:
KV cache + newest token -> next-token distribution
```

The prefill phase processes many tokens simultaneously.

The decode phase processes one new token at a time.

This distinction explains why two users can send the same model
different workloads and observe very different latency characteristics.

A short prompt with a long generated response may spend relatively
little time in prefill and much more time decoding.

A huge prompt with a short answer can have the opposite profile.

A serving system therefore needs to monitor at least:

``` text
time to first token
prefill throughput
decode throughput
inter-token latency
total generation latency
input token count
output token count
KV-cache usage
GPU memory utilization
GPU compute utilization
```

------------------------------------------------------------------------

# 33. Attention FLOPs in More Detail

It is useful to separate the attention computation into its components.

First, the model creates Q, K, and V.

For one projection:

``` text
X W
```

with:

``` text
X = [n, d_model]
W = [d_model, d_model]
```

requires approximately:

``` text
n * d_model * d_model
```

multiply-accumulate operations.

There are multiple projections, so the projection cost scales
approximately linearly with sequence length:

``` text
O(n d_model²)
```

The attention score calculation is:

``` text
Q K^T
```

which scales approximately as:

``` text
O(n² d_model)
```

The weighted value aggregation:

``` text
A V
```

also scales approximately as:

``` text
O(n² d_model)
```

Therefore a simplified Transformer attention block has both:

``` text
linear-in-sequence terms:
O(n d_model²)

quadratic-in-sequence terms:
O(n² d_model)
```

For sufficiently long sequences, the quadratic term becomes increasingly
important.

This distinction also explains why saying "Transformers are O(n²)" is
incomplete. The quadratic complexity specifically describes the
sequence-interaction component of dense self-attention. The complete
Transformer block includes other operations whose costs depend on model
width and feed-forward dimensions.

------------------------------------------------------------------------

# 34. Why Long Contexts Become Expensive

Suppose attention work is approximately proportional to:

``` text
n²
```

If the sequence length doubles:

``` text
n -> 2n
```

then:

``` text
n² -> (2n)²
     -> 4n²
```

So the pairwise attention work becomes approximately four times larger.

If the sequence length triples:

``` text
n -> 3n
```

then:

``` text
n² -> 9n²
```

The growth is therefore much faster than the number of tokens
themselves.

This is why long-context inference requires careful kernel design,
memory management, cache management, batching strategies, and sometimes
alternative attention structures.

genui{"learning_viz":{"type_id":"BIG_O_TIME_COMPLEXITY","locale_override":"en-US"}}

The visualization above is useful for building intuition about why
quadratic growth becomes increasingly expensive as the sequence length
grows.

------------------------------------------------------------------------

# 35. FlashAttention and the Difference Between FLOPs and Memory Traffic

A common misconception is that FlashAttention makes attention no longer
quadratic.

That is not the core idea.

FlashAttention computes exact attention while reducing expensive memory
movement between GPU high-bandwidth memory and on-chip memory through
tiling. The original FlashAttention paper describes this as an IO-aware
exact attention algorithm. [FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

The naive conceptual implementation is:

``` text
Q
K
 |
v
Q K^T
 |
v
attention matrix
 |
v
softmax
 |
v
attention matrix
 |
v
V
```

A straightforward implementation may materialize large intermediate
tensors, especially the n-by-n attention matrix.

FlashAttention instead divides the computation into blocks.

Conceptually:

``` text
Q block 1     K block 1
Q block 1     K block 2
Q block 1     K block 3
...
Q block 2     K block 1
Q block 2     K block 2
...
```

The algorithm computes partial attention results while keeping useful
intermediate data in faster on-chip memory rather than repeatedly
writing and reading the full attention matrix from HBM.

The key lesson is:

``` text
same mathematical attention
+
better memory access pattern
=
lower memory traffic
+
better practical performance
```

This is an inference-engineering lesson that generalizes far beyond
attention. Algorithmic FLOP counts do not fully predict GPU runtime.
Memory hierarchy and data movement matter.

------------------------------------------------------------------------

# 36. Token Generation End-to-End

We can now trace one generation step from beginning to end.

Assume the prompt has already been tokenized:

``` text
[token_1, token_2, ..., token_n]
```

The embedding layer maps them to:

``` text
X = [n, d_model]
```

Each Transformer layer computes attention and feed-forward
transformations. During prefill, this occurs for all prompt positions.

At the final layer, the hidden state at the final position becomes:

``` text
h_n = [d_model]
```

The language-model head converts this into:

``` text
logits = [vocab_size]
```

Temperature modifies those logits:

``` text
logits_scaled = logits / temperature
```

Top-p optionally removes low-probability candidates.

Softmax converts the remaining logits into a probability distribution.

Sampling or argmax chooses a token.

The token is appended to the sequence.

The decoder checks EOS, maximum length, and application-specific stop
conditions.

If generation has not stopped, the new token is fed into the next decode
iteration.

That loop is the fundamental engine behind autoregressive text
generation.

------------------------------------------------------------------------

# 37. The Most Important Mental Model

You should be able to visualize inference as two nested loops.

The outer loop is the autoregressive generation loop:

``` text
while not stopped:

    run model
    obtain next-token distribution
    choose token
    append token
```

Inside each model execution is the Transformer computation:

``` text
embedding
    ->
Transformer layer 1
    ->
Transformer layer 2
    ->
...
    ->
Transformer layer L
    ->
final normalization
    ->
vocabulary logits
```

Inside each attention layer is another mathematical pipeline:

``` text
X
 |
 +--> Q
 +--> K
 +--> V
 |
 v
Q K^T
 |
 v
scale by sqrt(d_k)
 |
 v
causal mask
 |
 v
softmax
 |
 v
weighted V
 |
 v
attention output
```

Understanding these nested levels gives you a useful mental model for
debugging and optimizing inference.

------------------------------------------------------------------------

# 38. Production Inference Considerations

A hand-written educational decoder is intentionally simple. A production
inference engine must solve substantially more problems.

The first is memory management. Model weights, activations, and KV
caches must fit within available GPU memory or be distributed across
devices. KV-cache growth is particularly important for long contexts and
large concurrent batches.

The second is batching. Multiple requests can be processed together to
improve hardware utilization. However, requests have different prompt
lengths and generation lengths, so serving systems often use dynamic
batching or continuous batching to avoid leaving GPU capacity idle.

The third is numerical precision. FP32, FP16, BF16, FP8, INT8, INT4, and
other formats change memory footprint, throughput, and numerical
behavior. Quantization can substantially reduce memory requirements, but
it must be evaluated for accuracy and stability.

The fourth is observability. Production systems should measure prompt
token counts, generated token counts, time to first token, inter-token
latency, request duration, GPU utilization, memory usage, KV-cache
utilization, queue time, errors, cancellations, and model/version
identifiers. Without these metrics, it is difficult to determine whether
latency problems originate in tokenization, queueing, prefill, decode,
memory pressure, network transport, or downstream sampling.

The fifth is security. Model-serving endpoints should authenticate
requests, authorize access to specific models, rate-limit expensive
workloads, validate maximum input and output lengths, protect sensitive
logs, encrypt data in transit and at rest, and prevent untrusted users
from exhausting GPU resources through pathological context sizes.

------------------------------------------------------------------------

# 39. The Exact Answer to the Checkpoint Question

Given a prompt of length n, dense self-attention forms an
attention-score matrix by multiplying:

``` text
Q [n, d_k]
```

with:

``` text
K^T [d_k, n]
```

The result has shape:

``` text
[n, n]
```

The first n corresponds to every query position.

The second n corresponds to every key position.

Therefore the matrix contains n² query-key interactions.

Each interaction is a dot product of length d_k, so the arithmetic
required to construct the score matrix is approximately:

``` text
n² d_k
```

The weighted value aggregation also has approximately:

``` text
n² d_v
```

work.

Thus the dense self-attention component is approximately:

``` text
O(n² d)
```

when the head dimensions are treated as proportional to the model
dimension.

The n² does not come from softmax. It does not come from tokenization.
It does not come from the vocabulary projection.

It comes from the fact that every query position compares itself with
every key position.

That is the core mathematical reason dense self-attention scales
quadratically with sequence length.

------------------------------------------------------------------------

# 40. Practical Exercise: Build the Decoder Yourself

The most valuable next step is to implement the entire inference loop
without using `model.generate()`.

Start with your Phase-0 nanoGPT model.

First, tokenize a prompt and obtain the initial token IDs.

Second, run one forward pass and inspect the logits at the final
sequence position.

Third, implement temperature scaling manually using:

``` python
scaled_logits = logits / temperature
```

Fourth, implement softmax manually at least once using:

``` python
shifted = logits - logits.max()
exp_values = torch.exp(shifted)
probabilities = exp_values / exp_values.sum()
```

The subtraction of the maximum is important for numerical stability
because exponentials can overflow for large positive logits.

Fifth, implement top-p using sorting, cumulative probability, masking,
and restoration of the original vocabulary indices.

Sixth, sample from the resulting distribution.

Seventh, append the selected token to the sequence.

Eighth, stop when EOS is generated or when the maximum generation length
is reached.

Ninth, repeat the entire procedure.

Finally, compare your implementation with the framework's generation
implementation to verify that the overall behavior is consistent.

------------------------------------------------------------------------

# 41. Top-p Validation Experiment

A good automated validation should test the candidate-set behavior
rather than merely checking whether the code executes.

``` python
import torch


def top_p_candidate_count(
    logits: torch.Tensor,
    top_p: float,
) -> int:
    """
    Count how many tokens survive top-p filtering.

    Args:
        logits: One-dimensional vocabulary logits.
        top_p: Nucleus probability threshold.

    Returns:
        Number of tokens retained by the top-p rule.
    """
    probabilities = torch.softmax(logits, dim=-1)

    sorted_probabilities, _ = torch.sort(
        probabilities,
        descending=True,
    )

    cumulative = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    mask = cumulative <= top_p

    # Include the first token that crosses the threshold.
    crossing = torch.nonzero(
        cumulative > top_p,
        as_tuple=False,
    )

    if crossing.numel() > 0:
        first_crossing = int(crossing[0].item())
        return first_crossing + 1

    return int(mask.sum().item())
```

Now test:

``` python
logits = torch.tensor(
    [4.0, 3.0, 2.0, 1.0, 0.0]
)

confident = logits * 3.0
flat = logits * 0.3

confident_count = top_p_candidate_count(
    confident,
    top_p=0.9,
)

flat_count = top_p_candidate_count(
    flat,
    top_p=0.9,
)

print("Confident candidate count:", confident_count)
print("Flat candidate count:", flat_count)
```

The expected relationship is:

``` text
confident_count < flat_count
```

for an appropriately chosen distribution and top-p threshold.

The exact counts are less important than the direction of the
relationship.

------------------------------------------------------------------------

# 42. What You Should Be Able to Explain Without Looking It Up

After completing this phase, you should be able to explain why
tokenization produces integers, why the vocabulary has a finite size,
and how those integers become embedding vectors.

You should be able to derive:

``` text
Q = X W_Q
K = X W_K
V = X W_V
```

and explain the dimensions of every matrix.

You should be able to derive:

``` text
Q K^T
```

and immediately recognize that:

``` text
[n, d_k] x [d_k, n] = [n, n]
```

which is the structural source of the n² attention interaction.

You should be able to explain why the dot products are divided by the
square root of the key dimension, why causal masking is required for
autoregressive generation, and how multiple attention heads divide the
representation.

You should be able to distinguish prefill from decode and explain why
prefill can exploit large amounts of GPU parallelism while decode has an
autoregressive dependency between generated tokens.

You should understand the role of the KV cache and why recomputing
previous keys and values during every decoding step is wasteful.

You should be able to explain the difference between logits and
probabilities and manually implement temperature, top-k, and top-p
transformations.

Finally, you should be able to explain why FlashAttention improves
practical performance without changing the fundamental mathematical
definition of dense attention: it primarily improves how the computation
moves through the GPU memory hierarchy.

------------------------------------------------------------------------

# 43. Final Mental Model

The complete lifecycle can be reduced to one conceptual pipeline:

``` text
TEXT
 |
 v
TOKENIZER
 |
 v
TOKEN IDs
 |
 v
EMBEDDINGS
 |
 v
TRANSFORMER
 |
 +-----------------------------+
 |                             |
 |       SELF-ATTENTION        |
 |                             |
 |  X -> Q, K, V               |
 |       |                     |
 |       v                     |
 |  Q K^T                      |
 |       |                     |
 |       v                     |
 |  / sqrt(d_k)                |
 |       |                     |
 |       v                     |
 |  causal mask                |
 |       |                     |
 |       v                     |
 |  softmax                    |
 |       |                     |
 |       v                     |
 |  attention weights x V      |
 |                             |
 +-----------------------------+
 |
 v
FEED-FORWARD NETWORK
 |
 v
REPEATED TRANSFORMER LAYERS
 |
 v
FINAL HIDDEN STATE
 |
 v
VOCABULARY LOGITS
 |
 v
TEMPERATURE
 |
 v
TOP-K / TOP-P
 |
 v
PROBABILITY DISTRIBUTION
 |
 v
SAMPLING / ARGMAX
 |
 v
NEXT TOKEN
 |
 v
STOP?
 |
 +---- NO ----> DECODE AGAIN
 |
 +---- YES ---> OUTPUT
```

The deepest lesson of this phase is that inference is not simply
"running a neural network." It is the interaction of token
representation, matrix multiplication, attention, probability
transformation, sequential decision-making, caching, memory movement,
and stopping logic.

Once you understand the tensor shapes and the origin of the n² term, the
behavior of modern LLM inference becomes much less mysterious. You can
reason about why long context costs more, why KV caching matters, why
prefill and decode behave differently, why temperature changes
diversity, why top-p changes its candidate count dynamically, and why
GPU-aware attention implementations can dramatically improve real-world
performance without changing the model's mathematical function.

## Recommended Reading

The original Transformer paper is the canonical source for scaled
dot-product attention and multi-head attention:

https://arxiv.org/abs/1706.03762

For the inference and GPU-memory perspective on attention, read the
FlashAttention paper:

https://arxiv.org/abs/2205.14135

While reading FlashAttention, focus especially on the distinction
between arithmetic complexity and IO complexity. The key insight is that
an algorithm can perform the same mathematical computation yet run
substantially faster when it reduces movement between different levels
of the memory hierarchy.

------------------------------------------------------------------------

## Phase-1 Completion Criteria

Phase 1 is complete when you can take an arbitrary prompt, explain its
token IDs, describe the tensor shape at every major Transformer stage,
derive the attention score matrix dimensions without reference material,
explain the origin of the n² term, explain the purpose of sqrt(d_k)
scaling, describe causal masking, distinguish prefill from decode,
explain KV caching, manually implement temperature scaling, manually
implement top-p filtering, and run an autoregressive decoding loop
without relying on `model.generate()`.

The most important checkpoint is not memorizing the attention equation.
It is being able to reconstruct it from tensor shapes:

``` text
Q        = [n, d]
K^T      = [d, n]

Q K^T    = [n, n]

[n, n] x [n, d]
         =
[n, d]
```

The first multiplication creates all pairwise query-key interactions.
The second uses those interactions to aggregate information from the
values. Once that chain is clear, the quadratic attention cost is no
longer something you need to memorize. You can derive it directly.
