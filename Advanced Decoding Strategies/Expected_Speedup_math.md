# Expected Speedup Math for Speculative Decoding

## Acceptance Rate, Draft Length, and the Real Speedup Equation

This guide derives how speculative-decoding acceptance rate and draft
length interact, and why expected accepted tokens are not the same as
actual end-to-end speedup.

## 1. Definitions

``` text
K       = speculative draft length
alpha   = conditional per-token acceptance probability
A       = accepted draft tokens before the first rejection
L       = committed output tokens in one speculative round
C_T     = cost of one ordinary target decode step
C_D(K)  = cost of drafting K tokens
C_V(K)  = target verification cost
C_O(K)  = other speculative overhead
```

A standard speculative round always commits at least one token.

For K=4:

``` text
reject d1:
replacement -> L=1

accept d1, reject d2:
d1 + replacement -> L=2

accept d1,d2, reject d3:
d1,d2 + replacement -> L=3

accept d1,d2,d3, reject d4:
d1,d2,d3 + replacement -> L=4

accept all:
d1,d2,d3,d4 + extra target token -> L=5
```

Thus:

``` text
1 <= L <= K+1
```

## 2. Acceptance-rate formula

For exact speculative sampling, proposal x from draft distribution q is
accepted with:

``` text
a(x) = min(1, p(x)/q(x))
```

where p is the target distribution.

Because x is proposed according to q:

``` text
alpha
=
sum_x q(x) * min(1,p(x)/q(x))

=
sum_x min(p(x),q(x))
```

Therefore alpha is the probability-mass overlap between target and
draft.

Using total variation distance:

``` text
TV(p,q)
=
0.5 * sum_x |p(x)-q(x)|
```

we have:

``` text
alpha = 1 - TV(p,q)
```

A better-aligned draft distribution therefore produces a higher exact
acceptance rate.

## 3. Constant-alpha model

For the first derivation assume each speculative position has the same
conditional acceptance probability:

``` text
alpha
```

This is a simplifying performance model. Real acceptance is context- and
position-dependent.

The probability that the first i draft tokens all survive is:

``` text
P(first i accepted)
=
alpha^i
```

## 4. Accepted-prefix distribution

For:

``` text
A = number of accepted drafts before first rejection
```

and:

``` text
a = 0,...,K-1
```

the probability is:

``` text
P(A=a)
=
alpha^a * (1-alpha)
```

All K accepted:

``` text
P(A=K)
=
alpha^K
```

So A behaves like a truncated geometric random variable.

For:

``` text
K=4
alpha=0.8
```

``` text
A=0 -> 0.2000
A=1 -> 0.1600
A=2 -> 0.1280
A=3 -> 0.1024
A=4 -> 0.4096
```

These sum to 1.

## 5. Expected accepted draft tokens

Using the tail-sum identity:

``` text
E[A]
=
P(A>=1)
+ P(A>=2)
+ ...
+ P(A>=K)
```

therefore:

``` text
E[A]
=
alpha
+ alpha^2
+ ...
+ alpha^K
```

For alpha not equal to 1:

``` text
E[A]
=
alpha * (1-alpha^K)/(1-alpha)
```

If:

``` text
alpha = 1
```

then:

``` text
E[A] = K
```

## 6. Expected committed tokens

This is the quantity most people accidentally undercount.

If a rejection occurs after A accepted drafts:

``` text
committed tokens
=
A accepted drafts
+
1 residual replacement
```

If all K survive:

``` text
committed tokens
=
K accepted drafts
+
1 extra target token
```

Therefore in the standard algorithm:

``` text
L = A + 1
```

for every outcome.

Thus:

``` text
E[L]
=
1 + E[A]
```

and:

``` text
E[L]
=
1
+ alpha
+ alpha^2
+ ...
+ alpha^K
```

For alpha not equal to 1:

``` text
E[L]
=
(1-alpha^(K+1))/(1-alpha)
```

If alpha=1:

``` text
E[L] = K+1
```

This is the central expected-progress formula.

## 7. Example: alpha=0.8, K=4

``` text
E[A]
=
0.8 + 0.64 + 0.512 + 0.4096
=
2.3616
```

But:

``` text
E[L]
=
1 + E[A]
=
3.3616
```

So one verification round advances about:

``` text
3.36 committed tokens
```

under this simplified model.

## 8. Full distribution of committed tokens

For:

``` text
l = 1,...,K
```

``` text
P(L=l)
=
alpha^(l-1) * (1-alpha)
```

For the all-accepted case:

``` text
P(L=K+1)
=
alpha^K
```

For alpha=0.8 and K=4:

``` text
L=1 -> 0.2000
L=2 -> 0.1600
L=3 -> 0.1280
L=4 -> 0.1024
L=5 -> 0.4096
```

Expected value:

``` text
1*0.20
+ 2*0.16
+ 3*0.128
+ 4*0.1024
+ 5*0.4096

=
3.3616
```

## 9. Why the series starts with 1

The series:

``` text
alpha + alpha^2 + ... + alpha^K
```

counts accepted draft tokens.

The series:

``` text
1 + alpha + alpha^2 + ... + alpha^K
```

counts committed output tokens.

Even if the first proposal is rejected, the residual correction produces
one valid target-distributed token.

Therefore every speculative round makes at least one token of progress.

## 10. Marginal benefit of increasing draft length

For draft length K:

``` text
E[L_K]
=
1 + alpha + ... + alpha^K
```

For K+1:

``` text
E[L_K+1]
=
1 + alpha + ... + alpha^K + alpha^(K+1)
```

Therefore:

``` text
Delta E[L]
=
alpha^(K+1)
```

This is a critical result.

The expected benefit of every additional speculative position shrinks
geometrically unless alpha=1.

## 11. Diminishing returns example

For alpha=0.8:

``` text
additional position    marginal expected progress

1                      0.800000
2                      0.640000
3                      0.512000
4                      0.409600
5                      0.327680
6                      0.262144
8                      0.167772
12                     0.068719
```

The draft and verification costs continue growing while marginal useful
progress declines.

Therefore:

``` text
larger K != automatically faster
```

## 12. Infinite-draft upper bound

For:

``` text
0 <= alpha < 1
```

as:

``` text
K -> infinity
```

``` text
E[L]
->
1/(1-alpha)
```

Examples:

``` text
alpha=0.50 -> ceiling 2
alpha=0.70 -> ceiling 3.333
alpha=0.80 -> ceiling 5
alpha=0.90 -> ceiling 10
alpha=0.95 -> ceiling 20
```

This is a statistical ceiling on expected progress under the
constant-alpha model.

A poor acceptance rate cannot be fixed merely by making K enormous.

## 13. Draft length interacts strongly with alpha

For alpha=0.5:

``` text
alpha^1 = 0.5000
alpha^2 = 0.2500
alpha^4 = 0.0625
alpha^8 = 0.0039
```

Later positions are almost never reached.

For alpha=0.95:

``` text
alpha^1 = 0.9500
alpha^2 = 0.9025
alpha^4 = 0.8145
alpha^8 = 0.6634
```

Long speculative blocks remain useful.

Therefore:

``` text
low alpha  -> short K usually makes sense
high alpha -> larger K can remain valuable
```

## 14. Expected-progress table

    alpha    K=1    K=2    K=4    K=8
  ------- ------ ------ ------ ------
     0.50   1.50   1.75   1.94   2.00
     0.70   1.70   2.19   2.77   3.20
     0.80   1.80   2.44   3.36   4.33
     0.90   1.90   2.71   4.10   6.13
     0.95   1.95   2.85   4.52   7.40

The numerator improves rapidly with K only when acceptance is
sufficiently high.

## 15. Expected progress is not speedup

A common mistake is:

``` text
speedup = E[L]
```

That would only be true if an entire speculative round cost exactly one
ordinary target decode step.

A real round performs:

``` text
draft generation
target verification
accept/reject logic
residual correction
KV-cache operations
scheduler work
```

These costs must be included.

## 16. Baseline work

One ordinary target decode step costs:

``` text
C_T
```

Producing E\[L\] tokens conventionally costs approximately:

``` text
C_baseline
=
E[L] * C_T
```

## 17. Speculative work

One speculative round costs:

``` text
C_spec(K)
=
C_D(K)
+
C_V(K)
+
C_O(K)
```

where C_O can include:

``` text
sampling
rollback
KV bookkeeping
scheduler overhead
synchronization
```

## 18. Fundamental expected-speedup equation

``` text
Speedup(K)
=
E[L_K] * C_T
/
(
    C_D(K)
    + C_V(K)
    + C_O(K)
)
```

Substituting the constant-alpha expected progress:

``` text
Speedup(K)
=
[
    (1-alpha^(K+1))/(1-alpha)
    * C_T
]
/
[
    C_D(K)
    + C_V(K)
    + C_O(K)
]
```

This is much more informative than acceptance rate alone.

## 19. Break-even condition

Speculation helps when:

``` text
Speedup(K) > 1
```

or equivalently:

``` text
E[L_K] * C_T
>
C_D(K)
+
C_V(K)
+
C_O(K)
```

This says:

``` text
baseline work avoided
>
speculative work performed
```

## 20. Normalize costs

Set one ordinary target decode step to:

``` text
C_T = 1
```

Define normalized costs:

``` text
d(K) = C_D(K)/C_T
v(K) = C_V(K)/C_T
o(K) = C_O(K)/C_T
```

Then:

``` text
Speedup(K)
=
E[L_K]
/
(d(K)+v(K)+o(K))
```

## 21. Simple draft-cost model

If the draft model costs:

``` text
c_d
```

per drafted token relative to one target decode step, then:

``` text
C_D(K)
≈
K*c_d
```

So:

``` text
Speedup(K)
≈
E[L_K]
/
(
    K*c_d
    + v(K)
    + o(K)
)
```

The numerator approaches a finite geometric ceiling while the
denominator continues growing.

This is why a finite optimal K normally exists.

## 22. Numerical speedup example

Suppose:

``` text
alpha = 0.8
K = 4

C_T = 1.00

draft cost/token = 0.05
C_D(4) = 0.20

C_V(4) = 1.10
C_O(4) = 0.05
```

Expected progress:

``` text
E[L] = 3.3616
```

Speculative cost:

``` text
0.20 + 1.10 + 0.05
=
1.35
```

Equivalent baseline cost:

``` text
3.3616 * 1.00
=
3.3616
```

Speedup:

``` text
3.3616 / 1.35
≈
2.49x
```

So:

``` text
expected progress = 3.36 tokens/round
```

does not mean:

``` text
speedup = 3.36x
```

because speculative execution is not free.

## 23. Verification cost

Do not assume:

``` text
C_V(K) = C_T
```

Target verification handles K known positions.

It normally costs more than one tiny target decode step, but can be much
cheaper than K sequential target steps because the GPU processes those
positions together.

A useful metric is:

``` text
verification efficiency ratio
=
C_V(K)/(K*C_T)
```

Example:

``` text
K=4
C_V=1.3*C_T
```

then:

``` text
ratio
=
1.3/4
=
0.325
```

The target verifies four candidate positions at 32.5% of the cost of
four separate target decode steps.

Acceptance then determines how much of that verified work becomes useful
output.

## 24. Two independent requirements for speedup

Speculation needs:

``` text
STATISTICAL EFFICIENCY
high acceptance
-> useful speculative work

HARDWARE EFFICIENCY
cheap drafting + efficient verification
-> low speculative cost
```

High alpha with an expensive draft can lose.

A cheap draft with poor alpha can also lose.

## 25. Marginal condition for increasing K

Adding speculative position K+1 gives expected progress:

``` text
alpha^(K+1)
```

Its baseline value is approximately:

``` text
alpha^(K+1) * C_T
```

Let incremental speculative cost be:

``` text
Delta C_spec
=
C_spec(K+1)-C_spec(K)
```

A useful first-order decision rule is:

``` text
increase K while:

alpha^(K+1) * C_T
>
Delta C_spec
```

Real systems should still benchmark because changing K can change GPU
efficiency and batch scheduling.

## 26. Why optimal K grows with acceptance

At K=5:

``` text
alpha=0.50:
alpha^6 = 0.015625

alpha=0.80:
alpha^6 = 0.262144

alpha=0.95:
alpha^6 ≈ 0.735092
```

The sixth speculative token has almost no expected value at 50%
acceptance but remains highly valuable at 95%.

## 27. General position-dependent acceptance

Real systems have:

``` text
alpha_1
alpha_2
...
alpha_K
```

where:

``` text
alpha_i
=
P(position i accepted |
  positions 1...i-1 accepted)
```

Define survival:

``` text
S_i
=
alpha_1 * alpha_2 * ... * alpha_i
```

Then:

``` text
E[A]
=
S_1 + S_2 + ... + S_K
```

and:

``` text
E[L]
=
1
+ S_1
+ S_2
+ ...
+ S_K
```

Explicitly:

``` text
E[L]
=
1
+ alpha_1
+ alpha_1*alpha_2
+ alpha_1*alpha_2*alpha_3
+ ...
+ alpha_1*...*alpha_K
```

If every alpha_i equals alpha, this becomes the geometric-series
formula.

## 28. Why global average acceptance can mislead

Consider K=4.

System A:

``` text
[0.95, 0.95, 0.60, 0.60]
```

System B:

``` text
[0.60, 0.60, 0.95, 0.95]
```

Both have arithmetic mean:

``` text
0.775
```

System A:

``` text
E[L]
=
1
+ 0.95
+ 0.95*0.95
+ 0.95*0.95*0.60
+ 0.95*0.95*0.60*0.60

=
3.7189
```

System B:

``` text
E[L]
=
1
+ 0.60
+ 0.60*0.60
+ 0.60*0.60*0.95
+ 0.60*0.60*0.95*0.95

=
2.6269
```

Same mean acceptance values, dramatically different useful progress.

Why?

An early rejection destroys the usefulness of every later proposal.

Therefore monitor:

``` text
conditional acceptance by speculative position
```

not only global acceptance.

## 29. Full-block acceptance

Constant alpha:

``` text
P(all K accepted)
=
alpha^K
```

General case:

``` text
P(all K accepted)
=
alpha_1*alpha_2*...*alpha_K
```

For alpha=0.8:

``` text
K=1  -> 0.8000
K=2  -> 0.6400
K=4  -> 0.4096
K=8  -> 0.1678
K=16 -> 0.0281
```

Full-block success falls exponentially even when many early tokens are
frequently useful.

## 30. Wasted suffix work

If rejection happens at position r:

``` text
d_r is rejected
d_r+1 ... d_K are discarded
```

The later draft work has already been performed.

Larger K therefore increases both:

``` text
potential useful progress
```

and:

``` text
potential wasted speculative work
```

This is another reason K must be optimized rather than maximized.

## 31. Draft-model size trade-off

A tiny draft model may have:

``` text
low draft cost
low alpha
```

A larger draft model may have:

``` text
higher draft cost
higher alpha
```

Compare:

``` text
Speedup_A
=
E[L_A]*C_T/C_spec_A
```

with:

``` text
Speedup_B
=
E[L_B]*C_T/C_spec_B
```

The best draft is the one that maximizes the end-to-end objective, not
the one with the highest acceptance or highest standalone draft
throughput.

## 32. High acceptance can justify a slower draft

For K=8:

``` text
alpha=0.70:
E[L] ≈ 3.20

alpha=0.90:
E[L] ≈ 6.13
```

The higher-quality draft nearly doubles expected progress.

A somewhat slower draft can still win if the increase in useful progress
exceeds its extra cost.

## 33. A nearly free bad draft can still be weak

For:

``` text
alpha=0.30
```

the infinite-K progress ceiling is:

``` text
1/(1-0.30)
=
1.4286
```

Even arbitrarily long speculation cannot create much expected progress.

Distributional alignment matters.

## 34. Idealized acceptance-only speedup ceiling

Imagine an unrealistic ideal where:

``` text
C_spec = C_T
```

regardless of K.

Then:

``` text
Speedup_ideal = E[L]
```

For infinite K:

``` text
Speedup_ideal,max
=
1/(1-alpha)
```

``` text
alpha=0.50 -> 2x
alpha=0.80 -> 5x
alpha=0.90 -> 10x
alpha=0.95 -> 20x
```

Real speedups are lower because draft and verification costs are
nonzero.

## 35. Hardware-aware model

Costs depend on workload state:

``` text
B = serving batch size
S = context length
```

A more realistic model is:

``` text
C_spec(K,B,S)
=
C_D(K,B,S)
+
C_V(K,B,S)
+
C_KV(K,B,S)
+
C_sched(K,B)
+
C_sample(K)
```

and:

``` text
Speedup
=
E[L] * C_T(B,S)
/
C_spec(K,B,S)
```

Therefore optimal K can change with:

``` text
context length
concurrency
GPU
precision
parallelism
KV-cache pressure
```

## 36. Context length

Long contexts increase KV-cache traffic.

This can change both:

``` text
ordinary target decode cost
```

and:

``` text
speculative verification cost
```

So the best K at 1K context may not be the best K at 32K context.

Benchmark by context bucket.

## 37. Concurrency

At low concurrency, ordinary decode may underutilize the GPU.

Speculation introduces token-level parallelism and can substantially
improve utilization.

At high concurrency, continuous batching already supplies considerable
parallel work.

The incremental benefit of speculation can therefore change with server
load.

## 38. Throughput versus latency

The K that maximizes server tokens/s may not minimize user TPOT.

Large K can increase:

``` text
drafting delay before verification
verification kernel duration
temporary KV allocation
```

A production optimizer may therefore solve:

``` text
maximize throughput

subject to:

P99 TPOT <= SLO
memory <= capacity
KV usage <= capacity
```

rather than maximizing raw expected speedup.

## 39. Adaptive K

A production controller can estimate:

``` text
recent alpha_i
draft cost
verification cost
batch state
KV pressure
```

For each candidate K:

``` text
estimate E[L_K]
estimate C_spec(K)
estimate speedup(K)
```

Then choose the best K under latency and memory constraints.

This is more robust than one fixed global draft length.

## 40. General marginal gain

For position-dependent acceptance:

``` text
S_K
=
alpha_1*alpha_2*...*alpha_K
```

Adding position K+1 contributes:

``` text
Delta E[L]
=
S_K * alpha_K+1
```

or:

``` text
Delta E[L]
=
alpha_1*alpha_2*...*alpha_K+1
```

This is the general version of:

``` text
alpha^(K+1)
```

## 41. Temperature and sampling configuration change acceptance

For stochastic speculation:

``` text
alpha
=
sum min(p,q)
```

where p and q must be the actual distributions used for sampling.

Changing:

``` text
temperature
top-k
top-p
repetition penalties
grammar constraints
```

changes p and/or q and therefore can change alpha.

Benchmark acceptance under the actual production decoding policy.

## 42. Greedy speculation

For deterministic greedy decoding, alpha has a different interpretation.

It can represent the conditional probability that the draft token
matches the target greedy token.

The geometric survival intuition can still apply:

``` text
P(first i matches)
≈ alpha^i
```

but this is not the same alpha as the exact stochastic overlap:

``` text
sum min(p,q)
```

Keep the two cases conceptually separate.

## 43. Expected target verification calls per token

A speculative round commits E\[L\] tokens on average.

So idealized verification rounds per output token are:

``` text
1/E[L]
```

For:

``` text
alpha=0.8
K=4
E[L]=3.3616
```

``` text
1/E[L]
≈
0.2975
```

For a 100-token output:

``` text
baseline:
~100 sequential target decode rounds

speculative:
~100/3.3616
≈ 29.75 verification rounds
```

This reduction in sequential target dependency depth is the main
opportunity.

It is not itself a latency speedup because each verification round has a
different cost from one baseline decode round.

## 44. End-of-sequence effects

If only two output tokens remain, a K=8 speculative block cannot realize
its full expected progress.

Short outputs therefore reduce amortization.

Speculative decoding tends to become more attractive as generation
length increases enough to amortize setup and boundary effects.

## 45. Verification cost is nonlinear

Do not assume:

``` text
C_V(K) proportional to K
```

For small K, increasing the token dimension may greatly improve GPU
utilization.

For large K, compute, activation traffic, and KV work eventually grow
substantially.

Measure:

``` text
C_V(1)
C_V(2)
C_V(4)
C_V(8)
...
```

on the actual serving engine.

## 46. Practical benchmark sweep

Sweep:

``` text
K = 1,2,3,4,6,8,12,16
```

For every K collect:

``` text
conditional alpha_1 ... alpha_K
global token acceptance
full-block acceptance
measured E[L]
draft latency
verification latency
rollback latency
total round latency
TPOT
output tokens/s
GPU utilization
HBM bandwidth
KV-cache occupancy
```

Then compare:

``` text
K vs E[L]
K vs round latency
K vs TPOT
K vs throughput
K vs full-block acceptance
```

Expected progress rises and saturates, while cost generally keeps
increasing. Speedup therefore commonly has an interior optimum.

## 47. Python calculator

``` python
from __future__ import annotations


def expected_committed_tokens(
    acceptance: float,
    draft_length: int,
) -> float:
    """Compute 1 + alpha + ... + alpha^K."""
    if not 0.0 <= acceptance <= 1.0:
        raise ValueError(
            "acceptance must be between 0 and 1."
        )
    if draft_length < 0:
        raise ValueError(
            "draft_length cannot be negative."
        )

    if acceptance == 1.0:
        return float(draft_length + 1)

    return (
        1.0
        - acceptance ** (draft_length + 1)
    ) / (1.0 - acceptance)


def expected_accepted_drafts(
    acceptance: float,
    draft_length: int,
) -> float:
    """Compute expected accepted draft-token count."""
    return (
        expected_committed_tokens(
            acceptance,
            draft_length,
        )
        - 1.0
    )


def expected_speedup(
    acceptance: float,
    draft_length: int,
    target_decode_cost: float,
    draft_cost: float,
    verification_cost: float,
    overhead_cost: float = 0.0,
) -> float:
    """Estimate speculative speedup from measured costs."""
    if target_decode_cost <= 0:
        raise ValueError(
            "target_decode_cost must be positive."
        )

    if any(
        cost < 0
        for cost in (
            draft_cost,
            verification_cost,
            overhead_cost,
        )
    ):
        raise ValueError("Costs cannot be negative.")

    speculative_cost = (
        draft_cost
        + verification_cost
        + overhead_cost
    )

    if speculative_cost <= 0:
        raise ValueError(
            "Speculative cost must be positive."
        )

    progress = expected_committed_tokens(
        acceptance,
        draft_length,
    )

    baseline_cost = (
        progress * target_decode_cost
    )

    return baseline_cost / speculative_cost


def main() -> None:
    """Sweep several example draft lengths."""
    acceptance = 0.80
    target_cost = 1.0
    draft_cost_per_token = 0.05
    overhead = 0.05

    # Illustrative values only.
    verification_costs = {
        1: 0.90,
        2: 0.98,
        4: 1.10,
        8: 1.35,
    }

    try:
        print(
            "K | E[L] | full accept | speedup"
        )
        print("-" * 42)

        for k, verification in (
            verification_costs.items()
        ):
            progress = expected_committed_tokens(
                acceptance,
                k,
            )

            full_accept = acceptance ** k

            speedup = expected_speedup(
                acceptance=acceptance,
                draft_length=k,
                target_decode_cost=target_cost,
                draft_cost=(
                    k * draft_cost_per_token
                ),
                verification_cost=verification,
                overhead_cost=overhead,
            )

            print(
                f"{k:>1} | "
                f"{progress:>4.2f} | "
                f"{full_accept:>11.3f} | "
                f"{speedup:>6.2f}x"
            )

    except ValueError as exc:
        print(f"Calculation failed: {exc}")


if __name__ == "__main__":
    main()
```

Replace the illustrative verification costs with measured engine values.

## 48. Position-dependent calculator

``` python
from __future__ import annotations

from collections.abc import Sequence


def expected_progress(
    conditional_acceptance: Sequence[float],
) -> float:
    """
    Compute:
    1 + a1 + a1*a2 + ... + product(a1...aK).
    """
    survival = 1.0
    expected = 1.0

    for position, alpha in enumerate(
        conditional_acceptance,
        start=1,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"Invalid alpha at position "
                f"{position}."
            )

        survival *= alpha
        expected += survival

    return expected


def main() -> None:
    """Compare two equal-mean acceptance profiles."""
    system_a = [
        0.95,
        0.95,
        0.60,
        0.60,
    ]

    system_b = [
        0.60,
        0.60,
        0.95,
        0.95,
    ]

    try:
        print(
            "System A:",
            expected_progress(system_a),
        )
        print(
            "System B:",
            expected_progress(system_b),
        )

    except ValueError as exc:
        print(f"Calculation failed: {exc}")


if __name__ == "__main__":
    main()
```

Expected approximately:

``` text
System A: 3.7189
System B: 2.6269
```

## 49. Interview derivation

Memorize this sequence:

``` text
1. Let alpha be conditional draft-token acceptance.

2. Probability first i drafts survive:
   alpha^i

3. A speculative round always commits one token.

4. Each surviving speculative position adds another
   committed token.

5. Therefore:

   E[L]
   =
   1 + alpha + alpha^2 + ... + alpha^K

6. Hence:

   E[L]
   =
   (1-alpha^(K+1))/(1-alpha)

7. Marginal gain from increasing K by one:

   alpha^(K+1)

8. Real speedup:

   Speedup(K)
   =
   E[L] * C_T
   /
   (C_D(K)+C_V(K)+C_O(K))
```

Therefore optimal K balances statistical survival against increasing
system cost.

## 50. Common mistakes

### Mistake 1: confusing accepted drafts with committed tokens

``` text
E[A]
=
alpha + ... + alpha^K
```

but:

``` text
E[L]
=
1 + E[A]
```

### Mistake 2: saying speedup equals E\[L\]

E\[L\] measures expected progress, not wall-clock cost.

### Mistake 3: maximizing K

Marginal progress falls as:

``` text
alpha^(K+1)
```

while costs continue increasing.

### Mistake 4: using only average acceptance

Early acceptance matters disproportionately because a rejection
invalidates the remaining suffix.

### Mistake 5: optimizing only full-block acceptance

Partially accepted blocks are still useful.

### Mistake 6: choosing the highest-acceptance draft

A higher-acceptance draft can be too expensive. Optimize end-to-end
speedup.

## 51. Formula sheet

``` text
Exact stochastic acceptance:

alpha
=
sum_x min(p(x),q(x))

=
1-TV(p,q)
```

``` text
First i proposals survive:

P =
alpha^i
```

``` text
Exactly a<K drafts accepted:

P(A=a)
=
alpha^a * (1-alpha)
```

``` text
All K accepted:

P(A=K)
=
alpha^K
```

``` text
Expected accepted drafts:

E[A]
=
alpha + alpha^2 + ... + alpha^K

=
alpha*(1-alpha^K)/(1-alpha)
```

``` text
Expected committed tokens:

E[L]
=
1 + alpha + alpha^2 + ... + alpha^K

=
(1-alpha^(K+1))/(1-alpha)
```

``` text
Marginal expected progress:

Delta E[L]
=
alpha^(K+1)
```

``` text
Infinite-K ceiling:

E[L] -> 1/(1-alpha)
```

``` text
Expected speedup:

Speedup(K)
=
E[L_K] * C_T
/
(C_D(K)+C_V(K)+C_O(K))
```

``` text
Break-even:

E[L_K] * C_T
>
C_D(K)+C_V(K)+C_O(K)
```

``` text
Position-dependent case:

E[L]
=
1
+ alpha_1
+ alpha_1*alpha_2
+ ...
+ alpha_1*...*alpha_K
```

## 52. Complete mental model

``` text
draft-target overlap
        |
        v
acceptance alpha
        |
        v
prefix survival
alpha, alpha^2, ..., alpha^K
        |
        v
expected progress
1 + alpha + ... + alpha^K
        |
        +--------------------+
        |                    |
        v                    v
 statistical benefit      system cost
                          draft
                          verify
                          KV
                          scheduler
        |                    |
        +---------+----------+
                  |
                  v
               SPEEDUP

E[L] * target decode cost
-------------------------
draft + verify + overhead
```

## Conclusion

The key mathematical interaction is geometric.

With constant conditional acceptance alpha and draft length K:

``` text
E[L]
=
1 + alpha + alpha^2 + ... + alpha^K
```

Every later speculative position is useful only if the entire earlier
prefix survives.

Therefore:

``` text
high alpha
-> slow survival decay
-> longer drafts can be worthwhile

low alpha
-> fast survival decay
-> long drafts mostly add wasted work
```

The marginal expected progress of adding one more draft position is:

``` text
alpha^(K+1)
```

so returns diminish with K.

But expected progress is only the numerator of the performance problem.
The real speedup is approximately:

``` text
Speedup(K)
=
E[L_K] * C_T
/
(C_D(K)+C_V(K)+C_O(K))
```

The production-optimal draft length therefore depends on acceptance
profile, draft cost, target verification efficiency, context length,
concurrency, KV-cache pressure, hardware, and latency objectives.

This is why adaptive speculation is more powerful than blindly choosing
the largest possible draft length.
