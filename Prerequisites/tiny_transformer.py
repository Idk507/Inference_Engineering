"""
A tiny transformer, from scratch, in NumPy. No autograd — every gradient
is derived and coded by hand, so nothing is hidden behind `.backward()`.

Task: character-level next-token prediction on a short piece of text.
Architecture: embeddings -> [1 transformer block: attention + MLP] -> output head.

Run: python3 tiny_transformer.py
"""
import numpy as np

np.random.seed(0)

# ---------------------------------------------------------------------------
# 1. Data: turn text into integer tokens
# ---------------------------------------------------------------------------
text = "to be or not to be that is the question " * 20
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
data = np.array([stoi[c] for c in text])

block_size = 16   # context length (how many tokens attend to each other)
d_model = 32      # embedding dimension
d_ff = 64         # hidden size of the feed-forward MLP
lr = 0.05

def get_batch(batch_size=32):
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# ---------------------------------------------------------------------------
# 2. Parameters
# ---------------------------------------------------------------------------
def init(shape):
    return (np.random.randn(*shape) * 0.02).astype(np.float64)

params = dict(
    tok_emb=init((vocab_size, d_model)),      # token embedding table
    pos_emb=init((block_size, d_model)),      # positional embedding table
    Wq=init((d_model, d_model)), Wk=init((d_model, d_model)), Wv=init((d_model, d_model)),
    Wo=init((d_model, d_model)),              # attention output projection
    W1=init((d_model, d_ff)), b1=np.zeros(d_ff),
    W2=init((d_ff, d_model)), b2=np.zeros(d_model),
    Wh=init((d_model, vocab_size)), bh=np.zeros(vocab_size),  # output head
)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layernorm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return xhat, mu, var

# ---------------------------------------------------------------------------
# 3. Forward pass (caches every intermediate needed for backprop)
# ---------------------------------------------------------------------------
causal_mask = np.triu(np.ones((block_size, block_size)), k=1).astype(bool)  # True = block

def forward(x_idx):
    B, T = x_idx.shape
    cache = {}

    tok = params["tok_emb"][x_idx]                     # (B,T,d)
    pos = params["pos_emb"][:T][None, :, :]             # (1,T,d)
    h0 = tok + pos                                      # embeddings step
    cache["h0"] = h0

    # --- self-attention ---
    h0n, mu1, var1 = layernorm(h0)
    cache.update(h0n=h0n, mu1=mu1, var1=var1)
    Q = h0n @ params["Wq"]; K = h0n @ params["Wk"]; V = h0n @ params["Wv"]
    cache.update(Q=Q, K=K, V=V)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)   # (B,T,T)
    scores = np.where(causal_mask[None], -1e9, scores)     # causal masking
    attn = softmax(scores, axis=-1)
    cache["attn"] = attn
    att_out = attn @ V                                   # (B,T,d)
    att_proj = att_out @ params["Wo"]
    cache.update(att_out=att_out, att_proj=att_proj)
    h1 = h0 + att_proj                                    # residual connection
    cache["h1"] = h1

    # --- feed-forward MLP ---
    h1n, mu2, var2 = layernorm(h1)
    cache.update(h1n=h1n, mu2=mu2, var2=var2)
    ff1 = h1n @ params["W1"] + params["b1"]
    relu = np.maximum(ff1, 0)
    ff2 = relu @ params["W2"] + params["b2"]
    cache.update(ff1=ff1, relu=relu, ff2=ff2)
    h2 = h1 + ff2                                         # residual connection
    cache["h2"] = h2

    # --- output head ---
    logits = h2 @ params["Wh"] + params["bh"]
    cache["logits"] = logits
    return logits, cache

def loss_and_grads(x_idx, y_idx):
    B, T = x_idx.shape
    logits, c = forward(x_idx)
    probs = softmax(logits, axis=-1)
    loss = -np.log(probs[np.arange(B)[:, None], np.arange(T)[None, :], y_idx] + 1e-9).mean()

    grads = {k: np.zeros_like(v) for k, v in params.items()}

    # dL/dlogits for softmax cross-entropy
    dlogits = probs.copy()
    dlogits[np.arange(B)[:, None], np.arange(T)[None, :], y_idx] -= 1
    dlogits /= (B * T)

    grads["Wh"] = c["h2"].reshape(-1, d_model).T @ dlogits.reshape(-1, vocab_size)
    grads["bh"] = dlogits.sum(axis=(0, 1))
    dh2 = dlogits @ params["Wh"].T

    # MLP backward
    dff2 = dh2
    grads["W2"] = c["relu"].reshape(-1, d_ff).T @ dff2.reshape(-1, d_model)
    grads["b2"] = dff2.sum(axis=(0, 1))
    drelu = dff2 @ params["W2"].T
    dff1 = drelu * (c["ff1"] > 0)
    grads["W1"] = c["h1n"].reshape(-1, d_model).T @ dff1.reshape(-1, d_ff)
    grads["b1"] = dff1.sum(axis=(0, 1))
    dh1n = dff1 @ params["W1"].T
    dh1 = dh2 + _layernorm_backward(dh1n, c["h1n"], c["mu2"], c["var2"], c["h1"])

    # attention backward
    datt_proj = dh1
    grads["Wo"] = c["att_out"].reshape(-1, d_model).T @ datt_proj.reshape(-1, d_model)
    datt_out = datt_proj @ params["Wo"].T
    dattn = datt_out @ c["V"].transpose(0, 2, 1)
    dV = c["attn"].transpose(0, 2, 1) @ datt_out
    dscores = _softmax_backward(dattn, c["attn"])
    dscores = np.where(causal_mask[None], 0, dscores) / np.sqrt(d_model)
    dQ = dscores @ c["K"]
    dK = dscores.transpose(0, 2, 1) @ c["Q"]
    grads["Wq"] = c["h0n"].reshape(-1, d_model).T @ dQ.reshape(-1, d_model)
    grads["Wk"] = c["h0n"].reshape(-1, d_model).T @ dK.reshape(-1, d_model)
    grads["Wv"] = c["h0n"].reshape(-1, d_model).T @ dV.reshape(-1, d_model)
    dh0n = dQ @ params["Wq"].T + dK @ params["Wk"].T + dV @ params["Wv"].T
    dh0 = dh1 + _layernorm_backward(dh0n, c["h0n"], c["mu1"], c["var1"], c["h0"])

    # embeddings backward
    grads["pos_emb"][:T] += dh0.sum(axis=0)
    np.add.at(grads["tok_emb"], x_idx, dh0)

    return loss, grads

def _layernorm_backward(dxhat, xhat, mu, var, x, eps=1e-5):
    N = x.shape[-1]
    std_inv = 1.0 / np.sqrt(var + eps)
    dvar_term = dxhat.sum(-1, keepdims=True) / N
    dxhat_x_term = (dxhat * xhat).sum(-1, keepdims=True) * xhat / N
    return std_inv * (dxhat - dvar_term - dxhat_x_term)

def _softmax_backward(dout, softmax_out):
    s = softmax_out
    dot = (dout * s).sum(-1, keepdims=True)
    return s * (dout - dot)

# ---------------------------------------------------------------------------
# 4. Training loop (plain SGD)
# ---------------------------------------------------------------------------
for step in range(400):
    x, y = get_batch()
    loss, grads = loss_and_grads(x, y)
    for k in params:
        params[k] -= lr * grads[k]
    if step % 50 == 0:
        print(f"step {step:4d}  loss {loss:.3f}")

# ---------------------------------------------------------------------------
# 5. Sample text from the trained model
# ---------------------------------------------------------------------------
def generate(prompt, n=60):
    idx = [stoi[c] for c in prompt]
    for _ in range(n):
        ctx = idx[-block_size:]
        ctx = [0] * (block_size - len(ctx)) + ctx  # left-pad
        logits, _ = forward(np.array([ctx]))
        next_id = np.random.choice(vocab_size, p=softmax(logits[0, -1]))
        idx.append(next_id)
    return "".join(itos[i] for i in idx)

print("\nSample:", generate("to be "))
