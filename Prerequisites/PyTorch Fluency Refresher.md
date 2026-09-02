# PyTorch Fluency Refresher

## 1. Tensors — the base data structure

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])   # from data
x = torch.zeros(3, 4)                     # shape (3,4)
x = torch.randn(3, 4)                     # standard normal
x = torch.arange(10).reshape(2, 5)

x.shape          # torch.Size([2, 5])
x.dtype           # torch.float32 by default for randn/zeros
x.device           # cpu or cuda
```

Key mental model: a tensor is (data pointer, shape, dtype, device, strides). Operations either return new tensors or mutate in place (methods ending in `_`, e.g. `x.add_(1)` mutates, `x.add(1)` doesn't).

**Reshaping gotchas:**
```python
x.view(10)      # requires contiguous memory, shares storage
x.reshape(10)   # like view but copies if needed — safer default
x.transpose(0,1)  # non-contiguous after this; .view() will fail, use .reshape() or .contiguous()
```

**Broadcasting** follows numpy rules: shapes align from the right, dims of size 1 stretch.
```python
a = torch.randn(3, 1)
b = torch.randn(1, 4)
a + b   # shape (3, 4)
```

## 2. Autograd basics — the thing everything else sits on

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()          # computes dy/dx
print(x.grad)          # tensor([7.])   (2*2 + 3)
```

Every tensor with `requires_grad=True` builds a computation graph as it's used. `.backward()` walks it in reverse (this *is* the backward pass from the FLOP discussion — literally computing $\partial \mathcal{L}/\partial x$ for every leaf).

Gradients **accumulate** by default — you must zero them between steps (this is why `optimizer.zero_grad()` exists; forgetting it is the single most common PyTorch bug).

## 3. `nn.Module` — the standard way to structure a model

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                       # ALWAYS call this first
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

model = MLP(784, 256, 10)
out = model(x)     # calls model.forward(x) via __call__ — don't call .forward() directly
```

**Why not call `.forward()` directly?** `__call__` (invoked via `model(x)`) also triggers registered hooks (pre-forward, forward, backward hooks). Calling `.forward()` skips them silently — a subtle bug source.

**Why `nn.Module` at all, mechanically:** subclassing gives you automatic parameter registration. Any `nn.Parameter` or `nn.Module` attribute assigned in `__init__` gets tracked:

```python
list(model.parameters())        # all learnable tensors, recursively, incl. submodules
model.state_dict()               # OrderedDict of name -> tensor, for saving/loading
sum(p.numel() for p in model.parameters())   # total param count — tie back to N from FLOP formulas
```

## 4. Writing a forward pass by hand (no `nn.Linear`)

Worth doing once to demystify what `nn.Linear` actually is — it's just `y = xW^T + b` with registered parameters:

```python
class MyLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x):
        return x @ self.weight.T + self.bias   # this matmul is the 2*T*d_in*d_out FLOPs
```

`nn.Parameter` is just a `Tensor` subclass with `requires_grad=True` by default, and it auto-registers with the parent `Module` (a plain tensor attribute would NOT show up in `.parameters()`).

A minimal transformer block by hand, to connect back to the FLOP-counting material:

```python
class SimpleAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.d, self.h = d, n_heads
        self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)     # fused QKV projection
        self.out = nn.Linear(d, d)

    def forward(self, x):                   # x: (B, T, d)
        B, T, d = x.shape
        qkv = self.qkv(x)                    # (B, T, 3d)  -- this is 2*T*d*3d FLOPs (per batch elem)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)  # (B, h, T, dh)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / self.dh**0.5     # (B, h, T, T) -- the 2*T^2*d term
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v                                      # (B, h, T, dh) -- another 2*T^2*d term
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.out(out)
```

Every line here maps directly onto a term in last session's FLOP derivation — `self.qkv(x)` and `self.out(out)` are the $2NT$ linear-layer terms, `q @ k.transpose(...)` and `attn @ v` are the two $2T^2d$ attention terms.

## 5. Running on GPU: `.to("cuda")`

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)     # moves all parameters + buffers to GPU
x = x.to(device)              # you must ALSO move your input tensors — separately

out = model(x)                 # now runs on GPU
```

**Critical rule: every tensor in an operation must be on the same device.** `model.to(device)` does NOT move data you pass in later — that's a separate call, and mismatched devices throw a `RuntimeError: Expected all tensors to be on the same device`.

`model.to(device)` mutates the module in place for `nn.Module` (unlike plain tensors — `x.to(device)` returns a new tensor, doesn't mutate `x`). So this is a common asymmetry:

```python
model.to(device)          # works — model is now on device, no reassignment needed
x = x.to(device)          # tensors: must reassign, .to() doesn't mutate in place
```

**Other device gotchas:**
- `.cuda()` and `.to("cuda")` are roughly equivalent; `.to()` is more general (also handles dtype: `.to(torch.float16)`, or both at once: `.to(device, dtype=torch.bfloat16)`).
- New tensors created inside `forward()` (e.g. `torch.zeros(...)`) default to CPU unless you specify `device=x.device` — a very common bug when writing custom layers.
- `.item()` and `.cpu().numpy()` force a sync + copy back to host — expensive if called in a hot loop (e.g. logging loss every step naively can bottleneck GPU utilization).

## 6. `torch.no_grad()` — what it actually does and why

```python
with torch.no_grad():
    out = model(x)
```

**What it does:** disables graph construction for the duration of the block. Operations still run and produce tensors, but PyTorch skips recording the operations needed for `.backward()` — no computation graph is built, so intermediate activations aren't kept around for the backward pass.

**Why this matters, concretely, tying back to Session 1's FLOP breakdown:**
- Forward pass with grad tracking ON: PyTorch stores activations needed later for backward (memory cost scales with depth × batch × sequence length — this is "activation memory," separate from parameter memory).
- Forward pass under `no_grad()`: none of that is stored. Memory usage drops substantially, and you skip the bookkeeping overhead of graph construction (small compute savings too, but **memory** is the dominant win).

**When to use it:**
- **Inference / evaluation** — you never call `.backward()`, so building the graph is pure waste: `model.eval()` + `torch.no_grad()` is the standard inference pattern.
- **Anywhere you compute something you don't want to backprop through** — e.g. computing evaluation metrics, logging, or manually updating parameters inside a custom optimizer step (`with torch.no_grad(): param -= lr * param.grad`).

```python
model.eval()                    # switches BatchNorm/Dropout to eval-mode behavior — NOT related to autograd
with torch.no_grad():           # this is what actually disables the graph
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
```

**Common confusion:** `model.eval()` and `torch.no_grad()` are independent and do different things — `eval()` changes layer *behavior* (dropout off, batchnorm uses running stats instead of batch stats), `no_grad()` changes *graph tracking*. You almost always want both together for inference, but they're not redundant, and forgetting either one is a distinct bug (forgetting `eval()` with dropout on gives noisy/wrong inference; forgetting `no_grad()` during eval just wastes memory, doesn't corrupt correctness — but can OOM on large validation sets).

**Related but distinct: `requires_grad_(False)` / freezing.** `no_grad()` is a *context* (temporary, applies to any tensor created inside it). Setting `param.requires_grad = False` is a *permanent* property of a specific parameter, used for freezing layers (e.g. freezing a pretrained backbone during fine-tuning) rather than disabling gradient tracking for one forward pass.

## 7. Putting it together: a minimal but complete training step

```python
model = MLP(784, 256, 10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()                         # dropout/batchnorm in train mode
for x, y in train_loader:
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()              # clear accumulated grads
    out = model(x)                       # forward pass — builds graph (grad tracking ON by default)
    loss = criterion(out, y)
    loss.backward()                      # backward pass — this is the "3x forward" work from Session 1
    optimizer.step()                     # parameter update using .grad

# ---- evaluation ----
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        # compute metrics, no graph, no .grad needed
```

This loop structure — `zero_grad → forward → loss → backward → step` — is the one pattern you'll see in essentially every training script you ever read.

---

## 1. Mixed Precision (autocast + GradScaler / bf16)

**The idea:** run most ops in fp16/bf16 (half the memory, ~2x throughput on tensor cores) while keeping numerically sensitive ops (like softmax, loss reduction, some norms) in fp32.

```python
scaler = torch.cuda.amp.GradScaler()   # only needed for fp16, not bf16

for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x)
        loss = criterion(out, y)

    scaler.scale(loss).backward()       # scales loss up before backward to avoid underflow
    scaler.step(optimizer)              # unscales grads, checks for inf/nan, steps if OK
    scaler.update()                     # adjusts scale factor for next iteration
```

**Why `GradScaler` exists:** fp16 has a tiny dynamic range (min normal ~$6\times10^{-5}$). Small gradients underflow to zero silently. The scaler multiplies the loss by a large factor (e.g. 65536) before `.backward()`, so gradients get scaled up proportionally and stay representable — then unscales before the optimizer step. It also skips the step automatically if it detects inf/nan (which signals the scale is too high) and adapts the scale down.

**bf16 doesn't need this.** bf16 has the same exponent range as fp32 (just less mantissa precision), so it doesn't have the underflow problem fp16 has — no `GradScaler` needed:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(x)
    loss = criterion(out, y)
loss.backward()
optimizer.step()
```

Trade-off: bf16 has less precision per value (fewer mantissa bits) than fp16, so it's less accurate at a given magnitude, but far more numerically stable for training — this is why bf16 is the default choice for large-model training today (fp16 was more common before bf16-capable hardware was widespread).

**Tie-back to FLOP counting:** mixed precision doesn't change the FLOP *count* from Session 1's formulas — it changes FLOP *throughput* (GPUs do fp16/bf16 matmuls roughly 2x faster than fp32, sometimes more with tensor cores), so it directly determines whether you're hitting good MFU. Master weights are often still kept in fp32 (or at least an fp32 optimizer state) even when compute is bf16 — that's a memory cost, not a FLOP cost.

## 2. Gradient Checkpointing (activation recomputation)

**The problem it solves:** Section 6 above noted that `no_grad()` saves memory by not storing activations — but you obviously *need* gradients during training, so you can't just turn it off. Activation memory scales with depth × batch × sequence length, and for deep models this dominates GPU memory, often more than parameters themselves.

**The trick:** don't store *all* intermediate activations during the forward pass. Store only a subset (checkpoints), and when backward needs an activation that wasn't saved, **recompute it** by re-running the forward pass for that segment.

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)   # don't store block's internal activations
        return x
```

`checkpoint(block, x)` runs `block(x)` under `no_grad()`-like conditions during the forward pass (discarding intermediates), but wraps it so that during `.backward()`, PyTorch **re-runs the forward pass for that block** to regenerate the needed activations just-in-time, then immediately backprops through that recomputation.

**Direct connection to your FLOP framework:** this is a textbook compute-memory trade-off.
- Memory: activation storage drops roughly from $O(L)$ (one segment per layer) to $O(\sqrt{L})$ or $O(1)$ depending on checkpointing granularity — huge savings for deep models.
- Compute: you pay for the forward pass of each checkpointed segment **twice** — once during the real forward, once during recompute in backward. So instead of the "3x forward" rule (1 fwd + 2 bwd) from Session 1, checkpointed layers cost roughly **4x forward** (1 fwd + 1 recompute-fwd + 2 bwd-equivalent) for the checkpointed portion. This is exactly why people describe it as "trading compute for memory" — you're explicitly increasing $\text{FLOPs}_{\text{fwd+bwd}}$ (roughly $6NT \to 8NT$ in the limit of checkpointing everything) to fit a bigger model or longer sequence in the same GPU memory.
- In practice you checkpoint selectively (e.g. every other block, or just the MLP not the attention) to tune this trade-off rather than applying it uniformly.

## 3. Custom `autograd.Function` — writing your own backward pass

`nn.Module` + built-in ops gives you autograd for free, chaining backward automatically. `autograd.Function` is for when you want to **define forward AND backward explicitly yourself** — needed for custom CUDA kernels, non-differentiable-looking ops that actually have a defined gradient, or numerically stable custom gradients.

```python
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)      # stash what backward will need
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output * 2 * x    # d(x^2)/dx = 2x, chain rule with grad_output
        return grad_input

x = torch.tensor([3.0], requires_grad=True)
y = MySquare.apply(x)      # NOTE: .apply(), not calling forward() directly
y.backward()
print(x.grad)    # tensor([6.])
```

**Mechanics to internalize:**
- `forward(ctx, *inputs)`: `ctx` is a context object for stashing anything backward needs (`ctx.save_for_backward(...)` for tensors specifically — preferred over `ctx.x = x` because it interacts correctly with memory management and avoids reference cycles).
- `backward(ctx, *grad_outputs)`: receives $\partial \mathcal{L}/\partial y$ (upstream gradient) for each output, must return $\partial \mathcal{L}/\partial x$ (via chain rule: $\frac{\partial \mathcal L}{\partial x} = \frac{\partial \mathcal L}{\partial y}\cdot\frac{\partial y}{\partial x}$) for each input, in the same order as `forward`'s inputs.
- Number of `backward` return values must match number of `forward` input arguments (return `None` for inputs that don't need gradients, e.g. non-tensor config args).
- Call via `MyFunction.apply(x)`, never instantiate and call `.forward()` directly — `.apply()` is what wires it into the autograd graph.

**A realistic use case tying back to memory/compute trade-offs:** this is exactly the mechanism gradient checkpointing is built on under the hood (`torch.utils.checkpoint` is implemented as a custom `autograd.Function` whose `backward` re-invokes `forward`). It's also how people implement **fused kernels** — e.g., a fused "linear + GELU" custom function that computes both forward and a hand-derived combined backward in one CUDA kernel, saving memory bandwidth (fewer intermediate tensors written to/read from HBM) even though the FLOP count is unchanged — a good concrete instance of the arithmetic-intensity/roofline idea from earlier: same FLOPs, less memory traffic, better wall-clock.

---

