Affine quantization is the basic mapping used to convert floating-point values such as FP32 or FP16 tensors into low-bit integers such as INT8. The core idea is simple: instead of storing every real number directly, we describe the tensor using an integer value plus two parameters called the scale and the zero-point. Once you understand those three pieces, symmetric quantization, asymmetric quantization, per-tensor quantization, per-channel quantization, and even KV-cache quantization become much easier to reason about.

The forward affine quantization equation is:

`q = round(x / scale) + zero_point`

followed by clipping:

`q = clamp(q, q_min, q_max)`

The approximate inverse, called dequantization, is:

`x_hat = scale * (q - zero_point)`

Here, `x` is the original floating-point number, `q` is the stored integer, `scale` tells us how much real-number distance one integer step represents, `zero_point` tells us which integer corresponds to the real value `0.0`, and `x_hat` is the reconstructed approximation of the original value.

Suppose we are using signed INT8. Its representable integer range is normally:

`q_min = -128`

`q_max = 127`

That gives us 256 distinct integer values. But our floating-point tensor may contain values such as:

`[-1.0, -0.31, 0.0, 0.48, 1.0]`

There are infinitely many possible floating-point numbers between `-1` and `1`, but only 256 INT8 codes. Quantization therefore divides the real-number range into discrete intervals.

The scale controls the width of those intervals.

Imagine a very simple quantizer where:

`scale = 0.1`

`zero_point = 0`

Then the mapping is approximately:

```text
float value        integer value

-0.3        ->      -3
-0.2        ->      -2
-0.1        ->      -1
 0.0        ->       0
 0.1        ->       1
 0.2        ->       2
 0.3        ->       3
```

Each increment of `1` in integer space means an increment of `0.1` in floating-point space.

For example:

```text
x = 0.37

q = round(0.37 / 0.1)
  = round(3.7)
  = 4
```

When we reconstruct it:

```text
x_hat = 0.1 * 4
      = 0.4
```

So the original value was:

`0.37`

but after quantization and dequantization we obtain:

`0.40`

The difference:

`0.40 - 0.37 = 0.03`

is quantization error.

This illustrates one of the most important ideas in quantization: quantization is lossy compression. Multiple nearby floating-point numbers collapse into the same integer code.

For example, with `scale = 0.1`:

```text
0.36
0.37
0.38
0.39
0.40
0.41
0.42
0.43
0.44
```

may all quantize to only a few integer values.

The scale therefore determines the resolution of the quantizer. Smaller scales provide finer numerical precision, while larger scales cover a larger numeric range but with coarser precision.

The interesting question is therefore: how do we calculate the scale?

For asymmetric affine quantization, suppose the floating-point range we want to represent is:

`[x_min, x_max]`

and the integer range is:

`[q_min, q_max]`

We want to map the endpoints approximately like this:

```text
x_min  -> q_min
x_max  -> q_max
```

The scale is therefore:

`scale = (x_max - x_min) / (q_max - q_min)`

This is simply the ratio between the width of the real-number interval and the width of the integer interval.

Consider a tensor whose values range from:

```text
x_min = -1
x_max = 3
```

Suppose we quantize to unsigned INT8:

```text
q_min = 0
q_max = 255
```

The floating-point range has width:

```text
3 - (-1) = 4
```

The integer range has width:

```text
255 - 0 = 255
```

Therefore:

```text
scale = 4 / 255
      ≈ 0.015686
```

Each integer increment therefore represents roughly:

`0.015686`

in the original floating-point domain.

But now we encounter another problem. Where should floating-point zero go?

This is exactly why the zero-point exists.

The affine transformation can be rearranged as:

`q ≈ x / scale + zero_point`

We want:

`x = 0`

to correspond exactly to some integer.

Therefore:

```text
zero_point =
q_min - x_min / scale
```

Using our example:

```text
zero_point =
0 - (-1 / 0.015686)

≈ 63.75
```

Because integer storage requires an integer:

```text
zero_point = round(63.75)
           = 64
```

Therefore the quantizer becomes:

```text
scale ≈ 0.015686
zero_point = 64
```

Now observe what happens to zero:

```text
q = round(0 / 0.015686) + 64

q = 64
```

And dequantization gives:

```text
x_hat =
0.015686 * (64 - 64)

= 0
```

This is important because neural networks contain many operations where representing zero accurately matters. Padding, ReLU outputs, sparsity, masking, convolution padding, and many matrix operations benefit from having an exact representation of zero.

The mapping now looks conceptually like:

```text
Real domain                   Integer domain

-1.0  ---------------------->   0

 0.0  ---------------------->  64

 1.0  ----------------------> 128

 2.0  ----------------------> 191

 3.0  ----------------------> 255
```

Notice something important: the integer value `0` does not represent floating-point zero.

Instead:

```text
integer 64 <-> float 0
```

That is the meaning of the zero-point.

Affine quantization can therefore be visualized as translating and scaling a number line:

```text
Floating point

-1 -------- 0 -------- 1 -------- 2 -------- 3

             |
             |
             v

Integer

0 --------- 64 ------- 128 ------ 191 ------ 255
```

The scale controls stretching or compression, while the zero-point controls translation.

This is why the mapping is called an affine transformation rather than merely a linear scaling.

A purely linear transformation would look like:

`q = x / scale`

but affine quantization adds a translation:

`q = x / scale + zero_point`

Now we can distinguish symmetric and asymmetric quantization.

In symmetric quantization, the floating-point range is forced to be symmetric around zero:

```text
[-A, +A]
```

For example:

```text
[-5.0, +5.0]
```

or:

```text
[-0.87, +0.87]
```

The zero-point is normally fixed to:

`0`

for signed integers.

The formula therefore simplifies to:

```text
q = round(x / scale)
```

and:

```text
x_hat = scale * q
```

Suppose our tensor has:

```text
x_min = -3
x_max = 2
```

The largest magnitude is:

```text
max_abs = max(abs(-3), abs(2))
        = 3
```

Symmetric quantization chooses the range:

```text
[-3, +3]
```

even though the tensor never reaches `+3`.

For INT8, many symmetric implementations use approximately:

```text
[-127, +127]
```

instead of `[-128,127]`, because the symmetric range around zero is convenient.

Then:

```text
scale = max_abs / 127

scale = 3 / 127
      ≈ 0.023622
```

Now:

```text
x = 2

q = round(2 / 0.023622)

≈ 85
```

Dequantization gives:

```text
x_hat = 85 * 0.023622
      ≈ 2.008
```

The mapping looks like:

```text
float                INT8

-3.0  -------------> -127
-2.0  ------------->  -85
-1.0  ------------->  -42
 0.0  ------------->    0
 1.0  ------------->   42
 2.0  ------------->   85
 3.0  ------------->  127
```

The key advantage is that zero maps naturally to zero:

```text
float 0 -> integer 0
```

There is no offset arithmetic required.

This makes symmetric quantization particularly attractive for neural-network weights.

Weights are commonly distributed approximately around zero:

```text
             *
          *  *  *
       *  *  *  *  *
-----*--*--*--*--*--*-----
            0
```

For example:

```text
-0.38
-0.14
-0.03
 0.02
 0.18
 0.41
```

Because positive and negative values are roughly balanced, a symmetric representation uses the integer range efficiently.

Asymmetric quantization behaves differently.

Suppose an activation tensor after ReLU has values:

```text
[0, 0.2, 0.5, 1.7, 3.8, 6.0]
```

Its range is:

```text
[0, 6]
```

If we used symmetric quantization, the range would become:

```text
[-6, +6]
```

But the negative half is completely unused.

Conceptually:

```text
INT8 range

-127 ........... 0 ........... +127
 |               |               |
-6               0               6

<------ wasted ------->
```

Half of our available integer codes represent numbers that never occur.

Instead, asymmetric quantization could map:

```text
0 -> 0
6 -> 255
```

when using UINT8.

Then:

```text
scale = 6 / 255
      ≈ 0.02353
```

All 256 codes represent useful values between zero and six.

This is why asymmetric quantization is often used for activations, especially highly skewed or non-negative activation distributions.

Consider another asymmetric tensor:

```text
[-1, 4]
```

Symmetric quantization would need:

```text
[-4, 4]
```

because it must preserve the largest magnitude.

That wastes:

```text
[-4, -1)
```

which never occurs.

Asymmetric quantization can instead use the complete integer space:

```text
-1 -------------------- 4
 |                       |
q_min                  q_max
```

giving better effective resolution.

The trade-off is computational complexity.

With symmetric quantization, matrix multiplication is conceptually:

```text
x ≈ scale_x * q_x
w ≈ scale_w * q_w
```

Therefore:

```text
x * w
≈
scale_x * scale_w * q_x * q_w
```

The integer kernel can accumulate:

```text
sum(q_x * q_w)
```

and multiply by scales afterward.

With asymmetric quantization:

```text
x ≈ scale_x * (q_x - zp_x)

w ≈ scale_w * (q_w - zp_w)
```

Therefore:

```text
x * w
≈
scale_x * scale_w *
(q_x - zp_x) *
(q_w - zp_w)
```

Expanding the product:

```text
(q_x - zp_x)(q_w - zp_w)

=
q_x q_w
- zp_w q_x
- zp_x q_w
+ zp_x zp_w
```

So the kernel potentially has to handle additional correction terms. Modern inference libraries can optimize these operations heavily, but symmetric quantization remains simpler and often faster.

This is why a very common production pattern is:

```text
Weights     -> symmetric quantization
Activations -> asymmetric or symmetric quantization
```

depending on the hardware and inference engine.

There is another subtle detail worth understanding: clipping.

Suppose the quantizer was calibrated for:

```text
[-1, 1]
```

but later receives:

```text
x = 1.4
```

If the maximum integer is:

`127`

then the calculated integer might exceed it.

For example:

```text
scale = 1 / 127

q = round(1.4 / scale)
  ≈ 178
```

But INT8 cannot represent `178`.

Therefore:

```text
q = clamp(178, -127, 127)
```

giving:

```text
q = 127
```

Dequantization produces:

```text
x_hat = 127 * scale
      = 1
```

The original:

`1.4`

has therefore been clipped to:

`1.0`

This is called saturation error.

There are consequently two major sources of quantization error.

One is rounding error:

```text
0.373
-> quantized
-> 0.370 or 0.375
```

The other is clipping error:

```text
1.4
-> clipped
-> 1.0
```

Choosing the quantization range therefore involves a trade-off. A very wide range avoids clipping but creates a large scale, which reduces precision. A narrow range gives fine precision but may clip outliers.

You can think about it like adjusting the zoom level of a camera.

If your range is:

```text
[-100, 100]
```

and you have only 256 integer levels, each step is roughly:

```text
200 / 255 ≈ 0.784
```

That is relatively coarse.

If most values actually lie inside:

```text
[-1, 1]
```

then nearly all of the integer codes are wasted representing rarely used regions.

Using:

```text
[-1, 1]
```

would instead give:

```text
2 / 255 ≈ 0.00784
```

which is around 100 times finer resolution.

This is why quantization calibration methods often ignore extreme outliers using percentiles, MSE minimization, KL-divergence-based calibration, or other optimization strategies rather than always choosing the absolute minimum and maximum.

For inference engineering, the most useful mental model is therefore:

```text
FLOAT DOMAIN
     |
     | divide by scale
     v
NORMALIZED DOMAIN
     |
     | round
     v
INTEGER GRID
     |
     | add zero-point
     v
INTEGER STORAGE
     |
     | clamp
     v
INT8 / INT4 / etc.
```

Dequantization reverses this:

```text
INTEGER
   |
   | subtract zero-point
   v
CENTERED INTEGER
   |
   | multiply by scale
   v
APPROXIMATE FLOAT
```

The difference between symmetric and asymmetric quantization can then be summarized conceptually as:

```text
SYMMETRIC

float range:
       -A -------- 0 -------- +A

integer range:
      -127 ------- 0 -------- 127

zero_point = 0
```

while asymmetric quantization looks like:

```text
ASYMMETRIC

float range:
       xmin -------- 0 ---------------- xmax

integer range:
       0 ----------- Z ---------------- 255

zero_point = Z
```

Symmetric quantization sacrifices some range efficiency in exchange for simpler arithmetic. Asymmetric quantization shifts the integer grid so that it can fit arbitrary floating-point distributions more tightly.

For modern LLM inference, this distinction becomes particularly important when we move into weight quantization and KV-cache quantization. Weight-only INT8/INT4 systems frequently prefer symmetric per-channel or group-wise quantization because transformer weights are generally centered around zero and matrix multiplication becomes straightforward. Activations and KV caches can have more irregular distributions, so the choice between symmetric and asymmetric schemes becomes more dependent on the quantization algorithm and hardware kernel.

A compact equation worth remembering is therefore:

```text
Quantization:

q = clamp(
    round(x / s) + z,
    q_min,
    q_max
)
```

and:

```text
Dequantization:

x_hat = s * (q - z)
```

where:

```text
s = scale
z = zero-point
```

Everything else in basic affine quantization is essentially about choosing good values for `s` and `z`.

In the context of the inference-engineering sequence you have been studying, the natural next concept is **per-tensor vs per-channel vs group-wise quantization**, because the affine formula itself remains exactly the same; what changes is how many separate `scale` and `zero_point` values we maintain across a tensor. That is the step that leads directly toward practical INT8/INT4 weight quantization and KV-cache quantization.
