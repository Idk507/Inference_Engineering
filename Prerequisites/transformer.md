# How a Transformer Works

A transformer turns tokens into predictions in three stages:

**1. Embeddings** — Each input token (word/subword) gets mapped to a vector via a lookup table. Since attention has no inherent sense of order, positional information is added (via learned or sinusoidal positional embeddings) so the model knows *where* each token sits in the sequence.

**2. Transformer blocks** (stacked N times) — Each block has two sub-layers, each wrapped with a residual connection + layer norm:
- **Self-attention**: every token looks at every other token (or previous tokens, if causal) and pulls in relevant context. Each token produces a Query, Key, and Value vector; attention weights = softmax(Q·Kᵀ/√d), then weighted-sum the Values.
- **Feed-forward network (MLP)**: a per-token 2-layer MLP (expand → nonlinearity → contract) that processes the mixed information independently for each position.

Attention moves information *between* tokens; the MLP processes information *within* a token.

**3. Output head** — After the final block, a linear layer projects each token's vector to vocabulary-size logits, and softmax turns those into a probability distribution over the next token.

Let me show this as a diagram, then build a tiny working transformer so this isn't just abstract.Here's the intuition picture, then the code so it's not just a diagram.Now let's build one. This is a character-level mini-GPT — small enough to train on CPU in a couple minutes, but with every real component: token+positional embeddings, multi-head self-attention, causal masking, MLP, residual connections, layer norm, and an output head.Disk space is too tight to reinstall torch, so I'll write it in plain NumPy with manual forward/backward passes instead — arguably a better learning exercise anyway, since nothing is hidden behind `.backward()`.It trains and loss drops steadily — the sample is already picking up word fragments from the repeated training string ("to", "be", "that"), which is exactly what you'd expect from ~400 steps on a tiny repetitive corpus.Mapping the code to the diagram:
<img width="367" height="422" alt="image" src="https://github.com/user-attachments/assets/5a2a0215-77d5-4bd8-9312-6f7de152fea6" />

- **Embeddings**: `tok_emb[x_idx] + pos_emb[:T]` — token identity plus position, since attention alone is order-blind.
- **Attention**: `Q,K,V = h0n @ Wq,Wk,Wv`, scores `Q @ K.T / sqrt(d)`, causal mask (a token can't see the future), softmax, weighted sum of `V`, projected by `Wo`, added back via residual.
- **MLP**: two linear layers with a ReLU in between, also added back via residual.
- **Output head**: final hidden state projected to vocab-size logits, trained with softmax cross-entropy.
- **Backward pass**: every gradient (`_layernorm_backward`, `_softmax_backward`, the chain through `Wq/Wk/Wv`) is written by hand — no `.backward()` — so you can see exactly how error flows from the loss back to the embeddings.

One real architectural thing this toy skips: it's a single block with one attention head. Production transformers stack many blocks and split attention into multiple heads (each learning a different kind of relationship — syntax, coreference, etc.) run in parallel, then concatenated. The math per head is identical to what's here, just done H times on smaller slices.

