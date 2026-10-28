# LLM Sampling Strategies in Full

## Introduction

At every autoregressive decoding step, an LLM produces one logit for each vocabulary token. Decoding transforms those logits into the next token.

```text
model -> logits -> constraints/penalties -> temperature -> top-k/top-p -> sample
```

Beam search follows a different path: it maintains multiple partial sequences, expands them, scores them, and repeatedly prunes them.

This README covers exact top-k and top-p algorithms, temperature, categorical sampling, beam-search scoring and pruning, repetition/frequency/presence penalties, no-repeat n-grams, implementation subtleties, and production implications. All equations use plain text.

## 1. Logits and softmax

For vocabulary size V:

```text
logits = [z_1, z_2, ..., z_V]

p_i = exp(z_i - m) / sum_j exp(z_j - m)

m = max(logits)
```

Subtracting the maximum provides numerical stability without changing softmax probabilities.

## 2. Greedy decoding

```text
next_token = argmax(logits)
```

Greedy decoding commits to the locally highest-scoring token at every step. It is cheap and deterministic under deterministic execution, but a locally optimal token need not produce the globally highest-scoring sequence.

## 3. Temperature

```text
adjusted_logit_i = logit_i / T
```

For `T < 1`, the distribution becomes sharper. For `T > 1`, it becomes flatter. As T approaches zero, behavior approaches greedy decoding. Production implementations should use an explicit deterministic mode rather than literally divide by zero.

## 4. Exact top-k algorithm

Top-k keeps exactly the k highest-scoring candidates, subject to tie-handling details.

```text
1. Obtain logits.
2. Find indices of the k largest logits.
3. Set all other logits to -infinity.
4. Softmax the surviving logits.
5. Sample from the renormalized categorical distribution.
```

Formally:

```text
S_k = indices of k largest logits

filtered_i =
    logit_i      if i is in S_k
    -infinity    otherwise

probabilities = softmax(filtered)
next_token ~ Categorical(probabilities)
```

Example:

```text
A 5
B 4
C 3
D 2
E 1

k = 3
```

Keep A, B, C. Their shifted exponentials are approximately:

```text
A: exp(0)  = 1
B: exp(-1) = 0.3679
C: exp(-2) = 0.1353

sum = 1.5032
```

Renormalized:

```text
A ≈ 0.6652
B ≈ 0.2447
C ≈ 0.0900
D = 0
E = 0
```

Top-k's limitation is that candidate count is fixed even when model uncertainty changes.

## 5. Exact top-p / nucleus algorithm

Top-p retains the smallest highest-probability prefix whose cumulative probability reaches or exceeds p.

```text
1. Convert logits to probabilities.
2. Sort token indices by probability descending.
3. Compute cumulative probability in that order.
4. Find the first token where cumulative mass >= p.
5. Keep every token through that boundary token.
6. Remove all later tokens.
7. Renormalize.
8. Sample.
```

Example:

```text
A 0.40
B 0.25
C 0.15
D 0.10
E 0.06
F 0.04

top_p = 0.80
```

Cumulative mass:

```text
A         0.40
A+B       0.65
A+B+C     0.80
```

Keep A, B, C. Renormalized:

```text
A = 0.40 / 0.80 = 0.50
B = 0.25 / 0.80 = 0.3125
C = 0.15 / 0.80 = 0.1875
```

The boundary token must be retained. Top-p does not mean retaining exactly p mass; it means the smallest prefix reaching or exceeding p.

## 6. Why top-p adapts to uncertainty

Confident distribution:

```text
A 0.92
B 0.03
C 0.02
...
```

With top-p 0.9, A alone can survive.

Uncertain distribution:

```text
A 0.12
B 0.11
C 0.10
D 0.09
...
```

Many candidates may be needed to reach 0.9.

Therefore top-p adapts candidate-set size to distribution concentration, whereas top-k always retains k candidates.

## 7. Combining top-k and top-p

A common conceptual pipeline is:

```text
logits
 -> penalties
 -> temperature
 -> top-k
 -> top-p
 -> softmax/renormalize
 -> sample
```

Order matters. Applying top-k before top-p means the nucleus is computed only among top-k survivors. Applying transformations in another order can produce another distribution.

## 8. Categorical sampling

For:

```text
A 0.50
B 0.30
C 0.20
```

draw `u` uniformly from `[0, 1)`:

```text
A [0.00, 0.50)
B [0.50, 0.80)
C [0.80, 1.00)
```

If `u = 0.72`, choose B. This is the final stochastic selection step.

## 9. Beam search

Beam search keeps `B` partial hypotheses rather than committing to one.

For generated sequence:

```text
y_1, ..., y_T
```

the sequence probability factors autoregressively. Multiplication is replaced by addition in log space:

```text
sequence_score =
sum_t log P(y_t | prompt, y_<t)
```

Higher score is better.

## 10. Exact beam-search expansion and pruning

For beam width B:

```text
1. Start with prompt, score 0.
2. Run the model for active beams.
3. Obtain next-token log probabilities.
4. Expand each beam with candidate next tokens.
5. new_score = parent_score + token_log_probability.
6. Apply configured sequence-score normalization/penalty.
7. Rank candidates globally.
8. Keep the best B active hypotheses.
9. Store EOS hypotheses as finished according to stopping policy.
10. Repeat.
11. Return the best finished hypothesis.
```

With B beams and vocabulary V, the conceptual candidate score tensor is `[B, V]`.

A flattened selected candidate maps back using:

```text
parent_beam = flat_index // V
token_id    = flat_index % V
```

## 11. Beam-search numerical example

Beam width:

```text
B = 2
```

First step:

```text
A logp -0.40
B logp -0.60
C logp -1.50
```

Keep A and B.

Expand A:

```text
A X: -0.40 + -0.70 = -1.10
A Y: -0.40 + -1.00 = -1.40
A Z: -0.40 + -2.00 = -2.40
```

Expand B:

```text
B X: -0.60 + -0.20 = -0.80
B Y: -0.60 + -0.90 = -1.50
B Z: -0.60 + -1.20 = -1.80
```

Global ranking:

```text
B X -0.80
A X -1.10
A Y -1.40
B Y -1.50
B Z -1.80
A Z -2.40
```

Prune back to B=2:

```text
B X
A X
```

Ordinary beam pruning is global. Both survivors may originate from the same parent if they are the best-scoring hypotheses.

## 12. EOS and finished hypotheses

When a beam emits EOS, it becomes a completed hypothesis. It should not continue as an ordinary active beam.

Implementations differ in when they terminate. A sophisticated implementation can stop when enough finished hypotheses exist and no active beam can beat them under the configured scoring rule. Exact EOS and early-stopping semantics should be treated as part of decoding configuration.

## 13. Length penalty

Raw log-probability tends to become more negative as sequences grow, creating a short-sequence bias.

A generic normalization is:

```text
normalized_score =
raw_log_probability / length^alpha
```

But libraries do not universally use the same formula. Some use transformed length functions. Always verify the exact implementation before comparing beam-search outputs.

## 14. Beam search and KV-cache reordering

After pruning, new beams may come from arbitrary old parents:

```text
new 0 <- old 2
new 1 <- old 2
new 2 <- old 0
new 3 <- old 3
```

Their KV-cache state must logically follow those parent mappings. Beam search therefore increases KV-cache storage, cache-management complexity, memory traffic, and serving cost.

## 15. Multiplicative repetition penalty

A common Hugging Face-style repetition penalty uses a sign-aware multiplicative transformation for previously seen token IDs.

For penalty `r > 1`:

```text
if logit > 0:
    adjusted = logit / r

if logit < 0:
    adjusted = logit * r
```

Example with r=1.2:

```text
3.0  -> 2.5
-2.0 -> -2.4
```

Both become less attractive.

Blindly dividing a negative logit would be wrong:

```text
-2 / 1.2 = -1.67
```

which makes it more attractive.

Depending on implementation, this penalty is generally based on whether a token appears rather than scaling with every occurrence. That differs from frequency penalties.

## 16. Frequency penalty

A generic additive frequency penalty is:

```text
adjusted_logit_i =
logit_i - frequency_penalty * count_i
```

If:

```text
frequency_penalty = 0.5
count = 3
logit = 4.0
```

then:

```text
adjusted = 4.0 - 1.5
         = 2.5
```

More occurrences produce stronger suppression.

## 17. Presence penalty

A generic presence penalty is:

```text
adjusted_logit_i =
logit_i
-
presence_penalty * indicator(count_i > 0)
```

It only asks whether the token appeared.

```text
count 0 -> no penalty
count 1 -> penalty
count 8 -> same penalty
```

Presence encourages new token choices; frequency increasingly discourages repeated token use.

## 18. No-repeat n-gram constraint

This is a hard constraint rather than a soft penalty.

For `no_repeat_ngram_size = 3`, determine whether a candidate token would recreate an existing trigram. If yes:

```text
candidate_logit = -infinity
```

Example:

```text
history: A B C D A B
current suffix: A B
candidate: C
```

Appending C recreates `A B C`, so C is banned.

## 19. Penalties operate on tokens, not words

Tokenizer units do not map perfectly to human words. Strong penalties can damage legitimate repetition in:

```text
code
JSON
URLs
names
technical terminology
tables
structured schemas
```

For code generation, repeated identifiers and syntax are often necessary. Penalty settings should therefore be workload-specific.

## 20. Prompt tokens versus generated tokens

Implementations can apply history-based penalties to:

```text
generated tokens only
```

or:

```text
prompt + generated tokens
```

Including prompt tokens can discourage the model from repeating entities or terminology from the user's request, which may hurt QA, extraction, summarization, and code editing. Verify library semantics.

## 21. Exact ordering matters

These pipelines are not equivalent:

```text
penalty -> temperature -> top-k -> top-p
```

and:

```text
top-p -> penalty -> temperature
```

Penalties can change which tokens enter the nucleus. Temperature can change how many tokens are needed to reach a top-p threshold.

For reproducibility, record the entire decoding pipeline and implementation version.

## 22. Temperature and top-p interaction

Low temperature sharpens probabilities, so fewer candidates may be needed to reach a given top-p threshold.

High temperature flattens probabilities, so the same top-p may retain more tokens.

Therefore sampling parameters interact; they are not independent knobs.

## 23. Top-p can become deterministic

Suppose:

```text
A 0.70
B 0.20
C 0.10

top_p = 0.5
```

The smallest prefix reaching 0.5 is just A. After renormalization:

```text
P(A) = 1
```

Sampling is enabled but this step is deterministic.

## 24. Logits processors and warpers

A useful conceptual distinction is:

```text
processors:
history/constraint-driven logit modifications

warpers:
reshape stochastic sampling distribution
```

Examples:

```text
repetition penalty -> processor
no-repeat n-gram   -> processor
grammar mask       -> processor

temperature        -> warper
top-k              -> warper
top-p              -> warper
```

Library terminology can differ.

## 25. Structured generation

For JSON or grammar-constrained generation:

```text
valid token   -> eligible
invalid token -> -infinity
```

Then sampling occurs over grammar-valid candidates. The filtering pipeline must preserve at least one valid continuation or provide a defined fallback.

Sampling configuration is not a security boundary. Tool calls and structured outputs still require independent validation and authorization.

## 26. Efficient top-p implementation subtlety

A common vectorized implementation calculates cumulative probability after sorting. If it simply removes every token where:

```text
cumulative > p
```

it can incorrectly remove the token that first crosses the threshold.

Correct nucleus sampling retains that boundary token. Vectorized implementations often shift the removal mask by one position to achieve this behavior.

## 27. Vocabulary-size performance

Modern vocabularies can contain 100,000 or more tokens. Every decode step may therefore involve a large logits vector.

Potential work includes:

```text
penalties
grammar masking
softmax
top-k selection
top-p candidate discovery
sampling
```

A naive full sort costs approximately:

```text
O(V log V)
```

and is not always necessary. Production GPU implementations use parallel selection and fused kernels where possible.

## 28. Sampling and TPOT

Sampling occurs every autoregressive decode iteration:

```text
model decode
 -> logits processing
 -> sampling
 -> next token
 -> model decode
```

At high concurrency, CPU round trips, full-vocabulary sorts, history scans, and synchronization can materially affect TPOT.

High-performance inference systems therefore optimize the decoding path as well as transformer kernels.

## 29. Educational NumPy implementation

```python
"""Educational implementation of common LLM sampling operations."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Return numerically stable softmax probabilities."""
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits must be a non-empty 1D array.")

    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    total = np.sum(exponentials)

    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Invalid softmax normalization.")

    return exponentials / total


def apply_repetition_penalty(
    logits: np.ndarray,
    previous_tokens: Iterable[int],
    penalty: float,
) -> np.ndarray:
    """Apply sign-aware multiplicative repetition penalty."""
    if penalty <= 0:
        raise ValueError("penalty must be positive.")

    adjusted = logits.copy()

    for token_id in set(previous_tokens):
        if token_id < 0 or token_id >= adjusted.size:
            raise IndexError(f"Invalid token id: {token_id}")

        if adjusted[token_id] > 0:
            adjusted[token_id] /= penalty
        else:
            adjusted[token_id] *= penalty

    return adjusted


def apply_frequency_penalty(
    logits: np.ndarray,
    previous_tokens: Iterable[int],
    penalty: float,
) -> np.ndarray:
    """Apply additive frequency penalty."""
    adjusted = logits.copy()

    for token_id, count in Counter(previous_tokens).items():
        if token_id < 0 or token_id >= adjusted.size:
            raise IndexError(f"Invalid token id: {token_id}")
        adjusted[token_id] -= penalty * count

    return adjusted


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep only k largest logits."""
    if k <= 0 or k >= logits.size:
        return logits.copy()

    indices = np.argpartition(logits, -k)[-k:]
    adjusted = np.full_like(logits, -np.inf)
    adjusted[indices] = logits[indices]
    return adjusted


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Keep smallest highest-probability prefix with mass >= p."""
    if not 0 < p <= 1:
        raise ValueError("p must satisfy 0 < p <= 1.")

    if p == 1:
        return logits.copy()

    probabilities = stable_softmax(logits)
    sorted_indices = np.argsort(-probabilities)
    cumulative = np.cumsum(probabilities[sorted_indices])

    crossing = np.searchsorted(cumulative, p, side="left")
    keep_count = min(int(crossing) + 1, logits.size)
    keep_indices = sorted_indices[:keep_count]

    adjusted = np.full_like(logits, -np.inf)
    adjusted[keep_indices] = logits[keep_indices]
    return adjusted


def sample_next_token(
    logits: np.ndarray,
    *,
    previous_tokens: Iterable[int] = (),
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
) -> int:
    """Sample one token through a transparent decoding pipeline."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling.")

    adjusted = apply_repetition_penalty(
        logits,
        previous_tokens,
        repetition_penalty,
    )
    adjusted /= temperature
    adjusted = top_k_filter(adjusted, top_k)
    adjusted = top_p_filter(adjusted, top_p)

    probabilities = stable_softmax(adjusted)
    rng = np.random.default_rng(seed)

    return int(rng.choice(logits.size, p=probabilities))


def main() -> None:
    """Run a small sampling demonstration."""
    logits = np.array(
        [5.0, 4.2, 3.4, 2.1, 1.0, -0.5],
        dtype=np.float64,
    )

    try:
        token = sample_next_token(
            logits,
            previous_tokens=[0, 0, 2],
            temperature=0.8,
            repetition_penalty=1.15,
            top_k=4,
            top_p=0.9,
            seed=42,
        )
        print(f"Selected token id: {token}")
    except (ValueError, IndexError, FloatingPointError) as exc:
        print(f"Sampling failed: {exc}")


if __name__ == "__main__":
    main()
```

This is educational code. Production inference engines use optimized batched implementations.

## 30. Beam-search implementation sketch

```python
"""Educational beam-search implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Beam:
    """Represent one beam-search hypothesis."""

    tokens: tuple[int, ...]
    score: float
    finished: bool = False


def beam_search(
    initial_tokens: tuple[int, ...],
    next_log_probs: Callable[[tuple[int, ...]], np.ndarray],
    *,
    beam_width: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> Beam:
    """Run a simple unnormalized educational beam search."""
    if beam_width <= 0:
        raise ValueError("beam_width must be positive.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    beams = [Beam(initial_tokens, 0.0, False)]

    for _ in range(max_new_tokens):
        candidates: list[Beam] = []

        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue

            log_probs = next_log_probs(beam.tokens)

            if log_probs.ndim != 1:
                raise ValueError(
                    "next_log_probs must return a 1D array."
                )

            candidate_count = min(beam_width, log_probs.size)
            token_ids = np.argpartition(
                log_probs,
                -candidate_count,
            )[-candidate_count:]

            for token_id in token_ids:
                token = int(token_id)
                candidates.append(
                    Beam(
                        beam.tokens + (token,),
                        beam.score + float(log_probs[token]),
                        token == eos_token_id,
                    )
                )

        candidates.sort(
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        beams = candidates[:beam_width]

        if all(beam.finished for beam in beams):
            break

    return max(beams, key=lambda beam: beam.score)
```

A production implementation needs stronger finished-hypothesis handling, length penalties, batched model execution, cache reordering, stopping rules, and efficient GPU candidate selection.

## 31. Choosing a strategy

```text
Deterministic extraction:
greedy or constrained decoding

Open-ended chat:
temperature + top-p, optionally top-k

Creative generation:
more diversity, while monitoring coherence

Translation / sequence ranking:
beam search can be useful when sequence likelihood aligns with quality

Code / JSON:
structured constraints, conservative randomness,
careful repetition penalties
```

These are tendencies, not universal parameter prescriptions.

## 32. Production observability

Record decoding settings with inference telemetry:

```text
model/version
tokenizer/version
temperature
top_k
top_p
repetition_penalty
frequency_penalty
presence_penalty
beam_width
length_penalty
max_new_tokens
seed policy
grammar mode
```

Correlate them with:

```text
TTFT
TPOT
output length
output tokens/s
request latency
GPU utilization
sampling-kernel time
```

Otherwise behavior changes may be incorrectly attributed to the model rather than the decoder.

## 33. Security

Do not treat low temperature, top-k, top-p, or repetition penalties as safety controls.

Production systems still require independent:

```text
authentication
authorization
tool permission checks
structured-output validation
safety policies
rate limiting
tenant isolation
secret handling
```

A sampled tool call must be validated regardless of its probability.

## 34. Complete mental model

```text
PROMPT
  |
MODEL
  |
RAW LOGITS
  |
  +-- hard constraints
  |
  +-- repetition/frequency/presence penalties
  |
  +-- temperature
  |
  +-- top-k
  |
  +-- top-p
  |
FINAL DISTRIBUTION
  |
  +-----------------------+
  |                       |
SAMPLE                 SEARCH
  |                       |
next token            greedy/beam
  |                       |
  +-----------+-----------+
              |
         append token
              |
        update KV cache
              |
        next decode step
```

## 35. Key takeaways

```text
TOP-K
Keep k highest-scoring candidates, renormalize, sample.

TOP-P
Keep the smallest highest-probability prefix whose cumulative
mass reaches or exceeds p, renormalize, sample.

TEMPERATURE
Divide logits by T to control distribution sharpness.

BEAM SEARCH
Maintain B hypotheses, expand, add token log-probabilities,
globally rank, and prune back to B each step.

MULTIPLICATIVE REPETITION PENALTY
Common sign-aware score transformation for previously seen tokens.

FREQUENCY PENALTY
Penalty grows with occurrence count.

PRESENCE PENALTY
Penalty depends only on whether a token appeared.

NO-REPEAT N-GRAM
Hard-bans a continuation that recreates an existing n-gram.

ORDER MATTERS
These transformations do not generally commute.

IMPLEMENTATION MATTERS
EOS handling, length penalties, prompt-history semantics,
minimum candidates, and exact formulas vary across libraries.
```

## Conclusion

Sampling is the algorithmic bridge between an LLM's next-token distribution and the actual sequence a user sees.

Top-k controls candidate count. Top-p controls retained probability mass. Temperature controls sharpness. Repetition, frequency, and presence penalties modify scores using token history. Beam search is fundamentally different: it performs bounded search across multiple partial sequences using cumulative log-probability scoring and repeated expansion and pruning.

For production inference:

```text
same model
+ same prompt
+ different decoding pipeline
= potentially very different output
```

The decoder should therefore be treated as versioned serving configuration. Because its operations execute once per generated token, efficient sampling and beam-state management also matter for TPOT, throughput, KV-cache pressure, and overall serving cost.
