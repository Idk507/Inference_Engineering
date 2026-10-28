# Speculative Decoding: Draft, Verify, Residual Resampling, and Exactness

## 1. Core idea

Ordinary autoregressive decoding advances one target-model token at a time:

```text
target -> token 1 -> target -> token 2 -> target -> token 3 -> ...
```

Speculative decoding changes the execution schedule:

```text
cheap draft model
    |
    v
propose d1, d2, ..., dK autoregressively
    |
    v
expensive target model
    |
    v
verify all proposed positions in one parallel pass
    |
    v
accept a valid prefix
or correct the first rejection
```

The classical stochastic algorithm is **mathematically exact**: its final output distribution is the same target distribution that ordinary target-model sampling would produce.

The draft model affects efficiency, not the authoritative output distribution.

---

## 2. Notation

At a particular token position let:

```text
p(x) = target-model probability of token x
q(x) = draft-model probability of token x
```

The draft proposes:

```text
X ~ q
```

We ultimately require:

```text
output ~ p
```

The problem is therefore:

```text
How can we exploit a cheap proposal from q
while preserving p exactly?
```

---

## 3. Draft phase

Given committed prefix `c`, draft K tokens:

```text
d1 ~ q1(. | c)

d2 ~ q2(. | c, d1)

d3 ~ q3(. | c, d1, d2)

...

dK ~ qK(. | c, d1, ..., dK-1)
```

Store the proposal distributions, or the information required by the exact verifier.

Drafting is still autoregressive, but the draft model is deliberately much cheaper than the target.

---

## 4. Verify phase

Once the entire speculative sequence is known:

```text
c, d1, d2, ..., dK
```

the target transformer can evaluate the proposed positions together.

Conceptually it produces:

```text
p1(. | c)

p2(. | c, d1)

p3(. | c, d1, d2)

...

pK(. | c, d1, ..., dK-1)

and usually

pK+1(. | c, d1, ..., dK)
```

The target distributions are then checked from left to right.

This parallel verification is the performance opportunity: several otherwise sequential target decode steps are replaced by one target verification workload.

---

## 5. Exact acceptance rule

For a proposed token `x`:

```text
acceptance_probability =
min(1, p(x) / q(x))
```

Draw:

```text
u ~ Uniform(0,1)
```

Accept if:

```text
u <= min(1, p(x)/q(x))
```

Equivalent:

```text
u * q(x) <= p(x)
```

There are two cases.

If:

```text
p(x) >= q(x)
```

then:

```text
acceptance = 1
```

If:

```text
p(x) < q(x)
```

then:

```text
acceptance = p(x)/q(x)
```

---

## 6. Accepted probability mass

Token x is proposed with probability:

```text
q(x)
```

and accepted with probability:

```text
min(1, p(x)/q(x))
```

Therefore:

```text
P(propose x AND accept x)
=
q(x) * min(1, p(x)/q(x))
```

which simplifies exactly to:

```text
min(p(x), q(x))
```

This identity is the foundation of speculative sampling.

The accepted branch captures the pointwise overlap between p and q.

---

## 7. Total acceptance probability

Summing accepted mass across the vocabulary:

```text
P(accept)
=
sum_x min(p(x), q(x))
```

Therefore:

```text
P(reject)
=
1 - sum_x min(p(x), q(x))
```

If p and q are close, overlap is high and proposals are frequently accepted.

If they differ strongly, rejection is frequent.

---

## 8. What mass is missing?

The desired final mass for x is:

```text
p(x)
```

The accepted branch already provides:

```text
min(p(x),q(x))
```

The missing amount is:

```text
p(x) - min(p(x),q(x))
```

which equals:

```text
max(0, p(x)-q(x))
```

Therefore a rejection must be corrected using the **positive residual**.

---

## 9. Exact residual-resampling distribution

Define:

```text
residual_unnormalized(x)
=
max(0, p(x)-q(x))
```

Normalize it:

```text
r(x)
=
max(0, p(x)-q(x))
/
sum_y max(0, p(y)-q(y))
```

On rejection:

```text
replacement ~ r
```

This is not an optional heuristic. It is what makes the stochastic algorithm distribution-preserving.

---

## 10. Why the residual denominator is exactly rejection probability

For every token:

```text
p(x)
=
min(p(x),q(x))
+
max(0,p(x)-q(x))
```

Sum over x:

```text
1
=
sum_x min(p(x),q(x))
+
sum_x max(0,p(x)-q(x))
```

Therefore:

```text
sum_x max(0,p(x)-q(x))
=
1 - sum_x min(p(x),q(x))
=
P(reject)
```

So the residual's normalization constant is exactly the probability of entering the rejection branch.

---

## 11. Exactness proof

A final token x can arise in two ways.

### Accepted branch

```text
P(output x via acceptance)
=
min(p(x),q(x))
```

### Rejected branch

```text
P(output x via rejection)
=
P(reject) * r(x)
```

Substitute the residual:

```text
P(output x via rejection)
=
max(0,p(x)-q(x))
```

Total:

```text
P(output x)
=
min(p(x),q(x))
+
max(0,p(x)-q(x))
```

But:

```text
min(p,q) + max(0,p-q) = p
```

Hence:

```text
P(output x) = p(x)
```

for every token x.

Therefore the speculative procedure samples **exactly from the target distribution**.

---

## 12. Numerical proof

Let:

```text
          A     B     C

p       0.50  0.30  0.20
q       0.40  0.50  0.10
```

Acceptance probabilities:

```text
A: min(1, 0.50/0.40) = 1
B: min(1, 0.30/0.50) = 0.60
C: min(1, 0.20/0.10) = 1
```

Accepted mass:

```text
A: 0.40 * 1    = 0.40
B: 0.50 * 0.60 = 0.30
C: 0.10 * 1    = 0.10

total = 0.80
```

So:

```text
P(reject) = 0.20
```

Residual:

```text
p-q:

A =  0.10
B = -0.20
C =  0.10
```

Positive part:

```text
[0.10, 0, 0.10]
```

Normalize:

```text
r = [0.50, 0, 0.50]
```

Rejected branch contributes:

```text
0.20 * r
=
[0.10, 0, 0.10]
```

Final mass:

```text
accepted + residual contribution

A: 0.40 + 0.10 = 0.50
B: 0.30 + 0    = 0.30
C: 0.10 + 0.10 = 0.20
```

Exactly:

```text
[0.50, 0.30, 0.20] = p
```

---

## 13. Why sampling from p after rejection is wrong

An incorrect implementation might do:

```text
proposal rejected
    |
    v
sample replacement from p
```

But the accepted branch has already contributed:

```text
min(p,q)
```

Sampling from all of p again would add mass to regions that already received their accepted share.

The residual must contain only the target mass still missing:

```text
max(0,p-q)
```

---

## 14. Why argmax matching is not exact stochastic speculation

This is also wrong for stochastic target sampling:

```text
if draft token == target argmax:
    accept
else:
    use target argmax
```

That changes the target distribution toward greedy decoding.

Exact speculative sampling uses the full target and proposal probabilities:

```text
min(1,p/q)
```

not merely argmax agreement.

---

## 15. Multiple-token verification

For draft:

```text
d1, d2, ..., dK
```

verify left to right.

At position i:

```text
accept di with probability:

min(1, p_i(di)/q_i(di))
```

If accepted:

```text
continue to i+1
```

If rejected:

```text
r_i =
normalize(max(0,p_i-q_i))

replacement ~ r_i
```

Then:

```text
commit replacement
discard d_i+1 ... d_K
stop this verification round
```

---

## 16. Why verification stops at the first rejection

Suppose:

```text
draft = A B C D
```

A and B are accepted.

C is rejected and replaced with X.

D was proposed under:

```text
prefix A B C
```

but the actual committed prefix is now:

```text
prefix A B X
```

Therefore D's proposal and target conditional distributions no longer correspond to the real autoregressive context.

D must be discarded.

This follows directly from autoregressive conditional dependence.

---

## 17. Why the full block remains exact

The one-token proof applies conditionally at every speculative position.

At position 1, the procedure samples exactly from:

```text
p1(. | committed prefix)
```

If d1 is accepted, that token becomes part of the real prefix.

At position 2, the same proof applies under that conditional prefix.

Continue inductively.

At the first rejection, residual resampling produces exactly the target conditional distribution for that position and invalid later proposals are discarded.

Thus the joint generated sequence follows the same autoregressive target distribution as ordinary target-model sampling.

---

## 18. The extra token

If all K proposals are accepted, target verification normally also yields:

```text
pK+1(. | prefix,d1,...,dK)
```

Then:

```text
xK+1 ~ pK+1
```

So the best case is:

```text
K draft tokens accepted
+
1 ordinary target sample
=
K+1 committed tokens
```

from one target verification iteration.

---

## 19. Complete exact algorithm

```text
DRAFT

for i = 1 ... K:
    compute q_i
    sample d_i ~ q_i
    append d_i to temporary prefix
    retain q_i information


VERIFY

run target on the known speculative block
obtain p_1 ... p_K and usually p_K+1


ACCEPT / REJECT

for i = 1 ... K:

    a_i =
    min(1, p_i(d_i)/q_i(d_i))

    u ~ Uniform(0,1)

    if u <= a_i:
        commit d_i
        continue

    residual =
    normalize(max(0,p_i-q_i))

    replacement ~ residual

    commit replacement
    discard later draft suffix
    stop


IF ALL K ACCEPTED

sample one extra token from p_K+1
```

---

## 20. Numerical stability

Rather than directly divide tiny probabilities, use log probabilities.

Given:

```text
log_p = log p(token)
log_q = log q(token)
```

then:

```text
log(p/q) = log_p - log_q
```

Acceptance can be tested with:

```text
log(u) <= min(0, log_p-log_q)
```

This avoids explicitly forming a potentially awkward tiny probability ratio.

---

## 21. Total variation interpretation

For discrete distributions:

```text
TV(p,q)
=
0.5 * sum_x |p(x)-q(x)|
```

A standard identity is:

```text
sum_x min(p(x),q(x))
=
1 - TV(p,q)
```

Since:

```text
P(accept)
=
sum_x min(p(x),q(x))
```

we get:

```text
P(accept)
=
1 - TV(p,q)
```

for the one-token coupling.

This gives a powerful intuition:

```text
draft closer to target
        |
        v
lower total variation
        |
        v
more distribution overlap
        |
        v
higher acceptance
```

---

## 22. Maximal-coupling interpretation

Imagine trying to construct:

```text
X ~ q
Y ~ p
```

while maximizing:

```text
P(X = Y)
```

The maximum possible agreement probability is:

```text
sum_x min(p(x),q(x))
```

The speculative acceptance mechanism captures precisely this overlap.

The residual branch then generates the target-only mass.

This is a deeper probabilistic explanation for why the algorithm is exact.

---

## 23. Expected accepted prefix length

For intuition only, suppose every position has approximately constant independent acceptance probability:

```text
alpha
```

Then:

```text
P(at least 1 accepted) ≈ alpha
P(at least 2 accepted) ≈ alpha^2
...
P(at least K accepted) ≈ alpha^K
```

Expected accepted draft tokens:

```text
E[A]
≈ alpha + alpha^2 + ... + alpha^K
```

For alpha not equal to 1:

```text
E[A]
≈ alpha * (1-alpha^K)/(1-alpha)
```

If all K survive, one extra target token can be emitted, so approximate expected advancement is:

```text
E[L]
≈
alpha + alpha^2 + ... + alpha^K
+ alpha^K
```

Real acceptance probabilities are context- and position-dependent, so this is a simplified performance model rather than the exact production expectation.

---

## 24. Example

For:

```text
alpha = 0.8
K = 4
```

expected accepted draft tokens:

```text
0.8 + 0.64 + 0.512 + 0.4096
=
2.3616
```

All-four acceptance probability:

```text
0.8^4 = 0.4096
```

Approximate advancement including the extra target token:

```text
2.3616 + 0.4096
=
2.7712 tokens per verification
```

This does **not** imply 2.77x speedup because drafting and verification themselves have costs.

---

## 25. Real speedup condition

Let:

```text
C_T(1) = cost of one ordinary target decode step
C_D(K) = cost of drafting K tokens
C_V(K) = target verification cost
E[L]   = expected committed tokens per speculative iteration
```

Approximate speculative cost per committed token:

```text
(C_D(K) + C_V(K)) / E[L]
```

Speculation helps when:

```text
(C_D(K) + C_V(K)) / E[L]
<
C_T(1)
```

Acceptance rate alone therefore does not determine speedup.

---

## 26. Why verification is hardware-friendly

K ordinary target steps look like:

```text
target -> sync -> target -> sync -> target -> ...
```

Verification looks more like:

```text
target([d1,d2,...,dK])
```

The latter can expose a larger token dimension and improve:

```text
parallelism
GPU utilization
weight reuse
arithmetic intensity
kernel efficiency
```

Speculation partially transforms several decode-like operations into a small prefill-like target workload.

---

## 27. Relationship to batching

Batching creates parallelism across:

```text
different sequences
```

Speculation creates parallelism across:

```text
future positions within a sequence
```

A serving engine can combine both:

```text
many requests
x
multiple speculative positions
```

This creates a richer scheduling problem involving latency, cache capacity, batch occupancy, and verification shape.

---

## 28. Relationship to TTFT and TPOT

Speculation mainly optimizes generation.

Its most direct benefits are:

```text
lower TPOT
higher output-token throughput
fewer sequential target iterations
```

It does not inherently remove prompt-prefill cost, so TTFT improvements may be smaller or workload-dependent.

---

## 29. KV-cache commit and rollback

Draft and target paths maintain autoregressive state.

Suppose:

```text
A accepted
B accepted
C rejected -> X
D speculative
```

Committed sequence:

```text
A B X
```

State that assumes:

```text
A B C D
```

cannot remain authoritative.

Production engines need:

```text
temporary speculative allocation
commit accepted prefix
rollback/truncate rejected suffix
correct replacement state
```

Paged KV-cache systems can make these operations more manageable.

---

## 30. Sampling transformations must be included in p and q

Suppose target decoding uses:

```text
temperature
top-k
top-p
repetition penalty
grammar constraints
```

Then `p` in the proof must mean the **actual target sampling distribution after those transformations**.

Likewise q must be the distribution actually used to sample the draft token.

The exact acceptance rule is then:

```text
min(1,p_transformed(d)/q_transformed(d))
```

and residual:

```text
normalize(max(0,p_transformed-q_transformed))
```

Using inconsistent distributions invalidates the simple exactness proof.

---

## 31. History-dependent processors

At speculative position i, penalties and constraints must use the correct speculative history.

This includes:

```text
repetition penalties
frequency penalties
presence penalties
no-repeat n-grams
grammar state
stop constraints
```

If these processors see the wrong history, p or q changes and distributional equivalence can be lost.

---

## 32. Exact does not mean bit-identical

Exactness means:

```text
the generated sequence distribution matches the target decoder
```

It does not necessarily mean:

```text
the same seed produces the same token sequence
as ordinary non-speculative decoding
```

Speculation consumes randomness differently:

```text
draft sampling
accept/reject uniforms
residual samples
extra target samples
```

Kernel order and floating-point reductions can also differ.

Distributional exactness is different from bitwise execution identity.

---

## 33. Greedy speculative decoding is a different case

If the target decoder is deterministic greedy decoding, verification can simply test target-selected tokens along the proposed path.

That is different from stochastic speculative sampling.

Do not confuse:

```text
greedy verification
```

with:

```text
p/q acceptance + residual correction
```

The latter is needed to preserve a stochastic target distribution.

---

## 34. Exact versus approximate speculative methods

Some methods deliberately relax exactness:

```text
threshold-based acceptance
approximate verifier
truncated correction
heuristic divergence tests
```

These may trade quality/distributional fidelity for speed.

They should not be called mathematically exact unless their particular algorithm has an appropriate proof.

---

## 35. Extreme case: q = p

If:

```text
q = p
```

then:

```text
p(x)/q(x) = 1
```

for all sampled tokens.

Therefore:

```text
acceptance = 100%
```

and:

```text
max(0,p-q) = 0
```

This is statistically ideal.

It is only computationally useful if q is cheaper than the target.

---

## 36. Poor draft model

If p and q have little overlap:

```text
P(accept)
=
sum min(p,q)
```

is small.

Then the system performs substantial:

```text
drafting
verification
suffix discard
rollback
```

while committing few tokens.

Speculation can become slower than ordinary decoding.

---

## 37. Adaptive speculation length

Larger K provides more potential parallel advancement but also:

```text
more draft latency
more verification work
more temporary KV state
more discarded suffix work after rejection
```

A production controller can adapt K based on:

```text
recent acceptance
TPOT target
batch occupancy
KV-cache pressure
draft cost
verification cost
```

---

## 38. Context-dependent acceptance

Draft quality may vary across:

```text
common prose
code
multilingual text
rare knowledge
reasoning
high-entropy transitions
structured output
```

Monitor acceptance by workload rather than relying only on a global average.

---

## 39. Production metrics

Useful telemetry:

```text
draft tokens proposed
draft tokens accepted
token acceptance rate
accepted prefix length
first rejection position
fully accepted block rate
tokens committed per verification
draft latency
verification latency
residual-resampling count
rollback count
KV-cache usage
TPOT
output tokens/s
```

Token acceptance rate and full-block acceptance rate are distinct metrics.

---

## 40. Continuous batching

Ordinary continuous batching often advances each request by roughly one decode token per iteration.

With speculation, a request can advance by:

```text
1 ... K+1 tokens
```

A scheduler may need states such as:

```text
RUNNING
  |
DRAFT
  |
VERIFY
  |
commit variable-length prefix
  |
RUNNING
```

Different requests reject at different positions, creating irregular progress and fairness considerations.

---

## 41. Interaction with quantization

A quantized draft may be much faster, but quantization can alter q.

The real objective is not:

```text
fastest draft in isolation
```

It is:

```text
best balance of
draft cost
+
target agreement
+
end-to-end speedup
```

Measure whether lower draft latency compensates for any acceptance-rate reduction.

---

## 42. Tree-based speculation

Classical speculation drafts one chain:

```text
A -> B -> C -> D
```

Advanced methods can propose trees:

```text
        A
      /   \
     B     C
    / \     \
   D   E     F
```

This can increase candidate coverage but adds tree attention masks, branch selection, cache complexity, and verification work.

Exactness depends on the concrete tree algorithm and cannot be assumed from the linear-chain proof alone.

---

## 43. Self-speculative decoding

A separate model is not mandatory.

A target model can sometimes draft using a cheaper internal computation:

```text
fewer layers
early exit
lightweight draft heads
shared representations
```

Then the full target verifies.

This can reduce separate-model memory, but the draft path must still be cheap and well aligned.

---

## 44. Medusa/EAGLE-style direction

Modern speculative systems include proposal mechanisms beyond a conventional small draft LM.

Examples conceptually include:

```text
multiple future-token heads
candidate trees
feature-level autoregressive drafting
```

The common systems goal remains:

```text
cheap high-quality proposals
+
efficient target verification
```

The exact acceptance/verification semantics should be studied per method.

---

## 45. CPU speculative-execution analogy

CPU:

```text
predict future control flow
execute ahead
commit if correct
discard if wrong
```

LLM:

```text
predict future tokens
compute ahead
verify with target
commit accepted prefix
discard invalid suffix
```

For stochastic LLM sampling there is an extra probabilistic requirement: rejection must redistribute probability mass correctly.

---

## 46. Educational exact implementation

```python
"""Exact one-token speculative sampling."""

from __future__ import annotations

import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    """Validate and normalize a probability vector."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected a non-empty 1D vector.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Values must be finite.")
    if np.any(values < 0):
        raise ValueError("Probabilities cannot be negative.")

    total = float(values.sum())

    if total <= 0:
        raise ValueError("Total probability mass must be positive.")

    return values / total


def speculative_sample(
    target: np.ndarray,
    draft: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, bool]:
    """
    Return an exact target-distributed token using a draft proposal.

    Returns:
        token_id and whether the draft proposal was accepted.
    """
    p = normalize(target)
    q = normalize(draft)

    if p.shape != q.shape:
        raise ValueError("Target and draft vocabularies must match.")

    proposal = int(rng.choice(q.size, p=q))

    if q[proposal] <= 0:
        raise FloatingPointError(
            "Sampled proposal has zero draft probability."
        )

    acceptance = min(
        1.0,
        float(p[proposal] / q[proposal]),
    )

    if float(rng.random()) <= acceptance:
        return proposal, True

    residual = np.maximum(p - q, 0.0)
    residual_mass = float(residual.sum())

    if residual_mass <= 0:
        raise FloatingPointError(
            "Rejection occurred with zero residual mass."
        )

    residual /= residual_mass

    replacement = int(
        rng.choice(
            p.size,
            p=residual,
        )
    )

    return replacement, False
```

---

## 47. Monte Carlo verification

```python
"""Empirically verify exact speculative sampling."""

from __future__ import annotations

import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize a non-negative probability vector."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected non-empty 1D values.")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid probabilities.")

    total = float(values.sum())

    if total <= 0:
        raise ValueError("Probability mass must be positive.")

    return values / total


def speculative_sample(
    p: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, bool]:
    """Sample exactly from p using proposal q."""
    p = normalize(p)
    q = normalize(q)

    proposal = int(rng.choice(q.size, p=q))

    acceptance = min(
        1.0,
        float(p[proposal] / q[proposal]),
    )

    if rng.random() <= acceptance:
        return proposal, True

    residual = normalize(
        np.maximum(p - q, 0.0)
    )

    return int(rng.choice(p.size, p=residual)), False


def main() -> None:
    """Compare empirical output with the target distribution."""
    p = np.array([0.50, 0.30, 0.20])
    q = np.array([0.40, 0.50, 0.10])

    trials = 500_000
    rng = np.random.default_rng(42)

    counts = np.zeros(p.size, dtype=np.int64)
    accepted = 0

    try:
        for _ in range(trials):
            token, was_accepted = speculative_sample(
                p,
                q,
                rng,
            )
            counts[token] += 1
            accepted += int(was_accepted)

        empirical = counts / trials
        theoretical_acceptance = np.minimum(p, q).sum()

        print("Target:", p)
        print("Empirical:", empirical)
        print("Absolute error:", np.abs(empirical - p))
        print("Empirical acceptance:", accepted / trials)
        print("Expected acceptance:", theoretical_acceptance)

    except (ValueError, FloatingPointError) as exc:
        print(f"Experiment failed: {exc}")


if __name__ == "__main__":
    main()
```

Expected:

```text
empirical distribution -> [0.50, 0.30, 0.20]
acceptance rate         -> 0.80
```

as the number of trials becomes large.

---

## 48. Correctness tests

A production implementation should test:

```text
q = p
-> 100% acceptance

tiny known distributions
-> hand-calculated output matches

poor-overlap p/q
-> rejection increases

Monte Carlo
-> empirical output converges to target p

EOS
-> only committed EOS terminates generation

top-k/top-p/temperature
-> verifier uses actual transformed distributions

history penalties
-> correct speculative history is used

grammar constraints
-> invalid tokens remain impossible

first rejection
-> later draft suffix is discarded

KV cache
-> only correct committed state survives
```

---

## 49. Side-effect safety

Speculative tokens are temporary.

They must not trigger irreversible external actions.

For an agent:

```text
draft proposes:
delete_resource()
```

The proposal might later be rejected.

Therefore:

```text
drafting
=
prediction only

verification
=
establish committed sequence

tool execution
=
only after committed, authorized output
```

Temporary speculative state must also preserve tenant and KV-cache isolation.

---

## 50. Complete mental model

```text
CURRENT PREFIX
      |
      v
CHEAP DRAFT
      |
      v
d1 -> d2 -> ... -> dK
      |
      v
TARGET VERIFICATION
      |
      v
p1, p2, ..., pK, pK+1
      |
      v
LEFT-TO-RIGHT CHECK

for each di:

    accept with:
    min(1,p_i(di)/q_i(di))

        |
   +----+----+
   |         |
accept     reject
   |         |
continue     v
        residual =
        normalize(max(0,p_i-q_i))
             |
             v
        sample correction
             |
             v
        discard later drafts
             |
             v
        restart

if all accepted:
sample extra token from pK+1
```

---

## 51. Four-line proof to memorize

```text
proposal mass:
q(x)

accepted mass:
q(x) * min(1,p(x)/q(x))
= min(p(x),q(x))

missing target mass:
max(0,p(x)-q(x))

final mass:
min(p,q) + max(0,p-q)
= p
```

That is why exact speculative sampling is **not an approximation**.

---

## 52. Interview-ready answer

```text
Speculative decoding uses a cheap draft model to propose K future tokens and
the expensive target model to verify those proposed positions in parallel.

For a proposed token x, exact stochastic speculation accepts it with
probability min(1,p(x)/q(x)), where p is the target distribution and q is the
proposal distribution. Since x was proposed with probability q(x), accepted
mass is q(x)*min(1,p(x)/q(x)) = min(p(x),q(x)).

If the proposal is rejected, the replacement is sampled from the normalized
positive residual max(0,p-q). The rejection branch therefore contributes
exactly max(0,p(x)-q(x)). Adding accepted and residual mass gives p(x)
exactly.

Verification proceeds left to right and stops at the first rejection because
later draft tokens were conditioned on a now-invalid prefix. If all K
proposals survive, the target verification pass can normally provide one
additional target-sampled token.
```

---

## 53. Key takeaways

```text
1. Drafting proposes; the target remains authoritative.

2. Target verification parallelizes multiple known speculative positions.

3. Exact acceptance is min(1,p/q).

4. Accepted mass is min(p,q).

5. Rejected mass is corrected with normalize(max(0,p-q)).

6. Accepted + residual mass equals p exactly.

7. The first rejection invalidates every later speculative token.

8. Full acceptance can yield K+1 committed tokens from one verification.

9. Acceptance probability equals distribution overlap:
   sum min(p,q) = 1-TV(p,q).

10. Draft quality changes efficiency, not target fidelity, when exact
    correction is implemented correctly.

11. Speedup depends on draft cost, verification cost, acceptance,
    hardware utilization, scheduling, and KV-cache management.

12. p and q must represent the actual sampling distributions after relevant
    temperature, truncation, penalties, and constraints.

13. Distributional exactness does not imply bit-identical seeded output.

14. Speculative tokens must never cause external side effects before commit.
```

---

## Conclusion

Speculative decoding attacks one of LLM inference's hardest constraints: sequential autoregressive target-model execution.

Instead of asking the expensive model to discover every next token sequentially, a cheap proposal mechanism guesses several future tokens. The target then verifies those known positions together.

The probability correction is what makes classical speculative sampling special:

```text
proposal:
x ~ q

accept:
min(1,p(x)/q(x))

accepted mass:
min(p(x),q(x))

residual:
normalize(max(0,p-q))

final distribution:
min(p,q) + max(0,p-q)
= p
```

So the optimization changes the **execution algorithm**, not the intended target distribution.

When the draft is cheap, target verification is efficient, and p and q have high overlap, several tokens can be committed per expensive target iteration. This reduces the sequential decode bottleneck and can improve TPOT and output throughput while retaining exact target-model sampling semantics.
