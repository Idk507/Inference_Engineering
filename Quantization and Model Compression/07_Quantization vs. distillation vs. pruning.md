# Quantization vs. Distillation vs. Pruning: Three Fundamentally Different Ways to Optimize AI Models

Quantization, knowledge distillation, and pruning are three major techniques used to make neural networks smaller, cheaper, and more efficient. They are frequently grouped together under model compression, but they solve the problem in fundamentally different ways.

The easiest distinction to remember is:

Quantization changes how precisely numbers are represented. Pruning removes parts of a model. Distillation trains another model to reproduce the behavior of a stronger model.

All three can reduce deployment costs, but their effects on parameter count, memory consumption, training requirements, hardware utilization, and inference latency are different.

Understanding those differences is particularly important in LLM inference engineering because a model that occupies fewer bytes does not necessarily require fewer arithmetic operations, and a model with fewer mathematical operations does not necessarily execute faster on a GPU.

## 1. The fundamental difference

Imagine we start with a transformer containing eight billion parameters.
<img width="475" height="277" alt="image" src="https://github.com/user-attachments/assets/7c971707-5461-41ad-88c4-2beaf8cc8c13" />

The parameter counts and proportions are illustrative, not claims about a particular real model or compression recipe.

Quantization typically preserves the architecture. The model still has eight billion parameters, but each parameter is represented using fewer bits.

Pruning removes weights, neurons, channels, attention heads, or whole layers. Depending on how pruning is performed, the actual executable architecture may become smaller or may remain dense with many zero-valued weights.

Distillation usually constructs a student model, often smaller than the original teacher, and trains that student to reproduce the teacher's predictions or internal behavior.

These differences lead to very different deployment characteristics.

## 2. Quantization: Reduce numerical precision

We have already examined affine quantization, GPTQ, AWQ, QAT, and GGUF. They all connect to the same broad principle: representing the model's parameters, activations, or caches using fewer bits.

For an FP16 weight matrix:

`W ∈ FP16`

A quantization operation approximates it using lower-bit integer codes:

`q = clamp(round(W / scale) + zero_point, q_min, q_max)`

The approximate reconstructed matrix is:

`W_hat = scale × (q - zero_point)`

Consider:

```
Original weights:

[0.124, 0.386, -0.742, 0.054]

Quantized approximation:

[0.125, 0.375, -0.750, 0.000]
```

The important point is that the weights still exist in the same positions. Their values have simply been approximated using a lower-precision numerical representation.

### What happens to parameter count?

Suppose the original model has eight billion parameters.

After INT4 weight quantization:

`parameter_count = 8 billion`

The parameter count remains unchanged.

But memory consumption can fall substantially:

`FP16 = 8B × 16 bits = 16 GB`

`INT4 = 8B × 4 bits = 4 GB`

Those are idealized weight-payload figures. Actual quantized checkpoints require additional storage for scales, zero-points, alignment, and sometimes higher-precision tensors.

### What happens to computation?

This is where quantization gets misunderstood.

If a linear layer originally performs:

`Y = X × W.T`

the quantized model still performs the corresponding linear operation using the same logical matrix dimensions.

The number of mathematical multiply-accumulate operations is not automatically reduced.

Instead, the computation uses a different numerical representation and potentially different hardware instructions.

An optimized inference engine may use INT8 tensor-core operations, native FP8 computation, packed INT4 processing, or fused dequantization and matrix multiplication.

Quantization can therefore reduce memory bandwidth consumption and improve hardware throughput without reducing the underlying number of logical matrix multiplication terms.

Quantization makes the arithmetic cheaper to store and potentially cheaper to execute; it does not inherently remove the arithmetic.

## 3. Pruning: Remove weights or network structures

Pruning tries to eliminate unnecessary parts of a neural network.

Consider a weight matrix:

```
Original:

W = [
    [0.9, 0.1, 0.7, 0.2],
    [0.3, 0.8, 0.1, 0.6],
    [0.2, 0.4, 0.9, 0.1]
]
```

Suppose we consider weights with small absolute magnitudes less important.

An unstructured pruning operation could replace them with zeros:

```
Pruned:

W = [
    [0.9, 0.0, 0.7, 0.0],
    [0.3, 0.8, 0.0, 0.6],
    [0.0, 0.4, 0.9, 0.0]
]
```

We have removed five nonzero weight connections.

However, the matrix still has three rows and four columns.

This distinction leads to the two major categories of pruning.

### 3.1 Unstructured pruning

Unstructured pruning removes individual weights, irrespective of their position.

For example, a model with one billion weights might have 50% of those weights set to zero.

Its mathematical sparsity becomes 50%.

But if the pruned weights remain stored in a conventional dense FP16 tensor, the tensor still contains one billion FP16 elements.

The original dense matrix multiplication may still execute the same number of operations, including multiplication by zero.

Therefore, 50% sparsity does not automatically produce a 50% smaller checkpoint or twice the inference speed.

This is explicitly noted in PyTorch's sparsity documentation: exploiting sparsity requires suitable storage layouts and kernels.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

torchao 0.17 documentation



To achieve real benefits, the pruned weights need a suitable compressed representation and an inference kernel that can skip them efficiently.

Sparse formats such as CSR or CSC store nonzero values together with information describing their positions.

The trade-off is additional indexing metadata, irregular memory access, and potentially poor GPU utilization.

At moderate sparsity, an optimized dense kernel may outperform a generic sparse kernel.

### 3.2 Structured pruning

Structured pruning removes entire architectural units instead of arbitrary individual weights.

For a transformer, these may include attention heads, feed-forward channels, or complete transformer blocks.

For example, consider a linear layer:

```
Original:

W.shape = [4096, 4096]
```

If we remove 1,024 output channels and propagate that change through all dependent operations, the resulting layer could have:

```
W_pruned.shape = [3072, 4096]
```

The original matrix contains:

`4096 × 4096 = 16,777,216 weights`

The smaller matrix contains:

`3072 × 4096 = 12,582,912 weights`

That is a 25% reduction in this particular matrix.

Unlike unstructured pruning, the resulting operation can use an ordinary dense matrix multiplication with smaller dimensions.

This makes structured pruning attractive for hardware acceleration.

However, removing transformer dimensions requires preserving architectural compatibility. For example, attention head sizes, residual connections, normalization dimensions, and subsequent layers may depend on the removed channels.

Pruning one matrix independently is not sufficient when its outputs feed multiple other operations.

### 3.3 Semi-structured pruning: NVIDIA's 2:4 sparsity

A third important category is semi-structured sparsity.

A well-known example is NVIDIA's 2:4 sparsity pattern.

For every group of four weights along the supported reduction dimension, at least two must be zero.

2:4 semi-structured sparsity
<img width="492" height="91" alt="image" src="https://github.com/user-attachments/assets/3d9b3658-f1f2-4523-a7b9-cdd5cb7e7a0b" />

Every group of four contains two nonzero weights. The regular pattern allows compatible sparse hardware to skip zero-valued work.

On supported NVIDIA Sparse Tensor Cores, this structure permits accelerated computation using compressed representations and position metadata.

The theoretical sparse arithmetic throughput may approach twice the corresponding dense arithmetic throughput for supported operations, but real model speedups depend on kernel selection and workload characteristics.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://developer.nvidia.com\&sz=32)

NVIDIA Technical Blog

+1



An additional subtlety is that 2:4 pruning need not reduce the transformer's hidden dimension or logical matrix shape. Instead, it reduces the nonzero elements inside a supported fixed-size structure.

That is why it is called semi-structured rather than conventional structured pruning.

## 4. Knowledge distillation: Transfer behavior into a student model

Distillation is fundamentally different because it is a training method.

Instead of directly reducing the storage precision or deleting connections in the original network, we train a student network using information from a teacher.

The teacher is generally larger, stronger, or more computationally expensive.

The student is usually smaller or cheaper to serve, although knowledge distillation does not mathematically require the student to have fewer parameters.

Consider:

```
Teacher model:
8 billion parameters

Student model:
2 billion parameters
```

The teacher produces predictions on training examples. The student learns to imitate those predictions, sometimes in combination with ground-truth labels and intermediate representations.

Once training is complete, the student can be deployed independently.

The teacher is generally not needed during student inference.

### 4.1 Why soft targets contain useful information

Suppose a classification teacher predicts:

```
Cat: 0.65
Dog: 0.30
Car: 0.05
```

The ground-truth label is Cat.

Ordinary supervised training may use a one-hot label:

```
Cat: 1.0
Dog: 0.0
Car: 0.0
```

But the teacher distribution contains additional information.

It tells us that Dog is more plausible than Car for this image.

This relationship between classes is sometimes called dark knowledge.

A student trained only on the hard label does not directly receive that information. A student trained on the teacher's probability distribution does.

The classic paper by Geoffrey Hinton, Oriol Vinyals, and Jeff Dean introduced temperature-softened probability distributions as a central mechanism for distillation.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://research.google\&sz=32)

Google Research



### 4.2 Temperature scaling

Let the teacher produce logits:

`z = [3, 2, 0]`

Ordinary softmax converts these logits into probabilities:

`p_i = exp(z_i) / sum(exp(z_j))`

Distillation introduces a temperature `T`:

`p_i(T) = exp(z_i / T) / sum(exp(z_j / T))`

When `T = 1`, this is ordinary softmax.

When `T` is larger than 1, the probability distribution becomes softer, revealing more information about the relative scores of classes that would otherwise have very low probabilities.

Explore distillation temperature

<img width="487" height="222" alt="image" src="https://github.com/user-attachments/assets/79323801-626b-441c-bff6-cb56e0b4f4c8" />


Fixed teacher logits [3, 2, 0]. Higher temperature reveals more information about lower-ranked classes; it does not change their ordering.

### 4.3 Distillation loss

A common distillation objective combines supervised learning with teacher imitation:

`Loss = alpha × CrossEntropy(student, labels) + (1 - alpha) × T² × KL(teacher_T || student_T)`

Here, KL divergence measures how different the student's softened output probability distribution is from the teacher's.

The factor `T²` is conventionally used to compensate for changes in gradient magnitude introduced by temperature scaling.

The mixing coefficient `alpha` controls the relative importance of hard labels and teacher predictions.

For a large language model, logits are produced over vocabulary tokens rather than just a few classification labels.

Distillation may therefore train the student to match the teacher's next-token distribution.

However, full vocabulary-logit distillation can be expensive because storing or transferring dense distributions over very large vocabularies consumes substantial memory and bandwidth.

For that reason, modern LLM distillation can also involve teacher-generated responses, intermediate hidden-state matching, selected logits, preference signals, and other objectives.

### 4.4 Sequence-level distillation for LLMs

Suppose the teacher is asked:

> Explain how a KV cache reduces attention computation.

The teacher generates a high-quality answer.

That generated answer becomes a supervised training target for the student.

Over many examples, the student learns patterns in the teacher's responses.

The distinction between this and ordinary supervised fine-tuning is the provenance of the training signal: the responses are generated or scored by a teacher rather than exclusively obtained from independently labeled data.

But imitation also has limitations. A student may copy the teacher's mistakes, stylistic biases, or unsupported claims. Teacher-generated outputs should therefore be filtered and evaluated, especially for tasks requiring factual correctness.

## 5. Comparing the three methods mathematically

We can now express the key differences with equations.

For quantization, start with a floating-point matrix `W` and construct a low-precision approximation:

`W_quantized = Dequantize(Quantize(W))`

The objective is to preserve the original computation while lowering numerical storage and possibly execution cost.

For pruning, introduce a mask `M`:

`W_pruned = W × M`

where mask elements are zero or one.

The objective is to remove weight connections or structures while retaining acceptable model quality.

For distillation, create a separate student with parameters `theta_student` and optimize:

`min_theta_student Loss(student, teacher, training_data)`

The objective is to transfer useful teacher behavior into the student.

These expressions correspond to different operations: numeric approximation, structural deletion, and behavioral learning.

## 6. Side-by-side engineering comparison

| Dimension                    | Quantization    | Pruning                              | Distillation               |
| ---------------------------- | --------------- | ------------------------------------ | -------------------------- |
| Changes numerical precision  | Yes             | Not necessarily                      | Not necessarily            |
| Reduces nonzero weights      | Not inherently  | Yes                                  | Depends on student         |
| Reduces architecture size    | Generally no    | With structured pruning              | Often                      |
| Requires training            | PTQ no; QAT yes | Sometimes fine-tuning                | Yes                        |
| Uses teacher model           | Not required    | Not required                         | Typically yes              |
| Reduces weight memory        | Usually         | Only with suitable structure/storage | Usually if student smaller |
| Reduces dense FLOPs          | Not inherently  | Structured pruning can               | Usually if student smaller |
| Requires specialized kernels | For speed often | For sparse pruning often             | Usually not                |
| Principal risk               | Precision loss  | Capacity/connectivity loss           | Knowledge-transfer failure |

One common misconception is that a quantized eight-billion-parameter model becomes a two-billion-parameter model because its memory footprint is reduced by approximately four times.

That is incorrect.

A four-bit eight-billion-parameter model still has eight billion logical parameters.

It is not equivalent to a dense two-billion-parameter model.

The two models have different architectures, compute requirements, and numerical behavior.

## 7. A worked example: Three ways to optimize an 8B model

Suppose our original model has eight billion parameters and FP16 weights.

Its raw weight storage is 16 GB.

We want to deploy it on a memory-constrained inference server.

Illustrative compression outcomes

Raw weight storage across strategies

Illustrative scenarios; sparse case assumes physically compacted surviving weights and ignores indexing metadata


<img width="492" height="265" alt="image" src="https://github.com/user-attachments/assets/903cdd4d-6cc1-4c19-8de3-c7f542db75a0" />


The first strategy is quantization. We retain the eight-billion-parameter architecture and quantize the weights to four bits.

The ideal raw weight payload becomes 4 GB. The logical number of parameters and dense matrix dimensions remain the same.

The second strategy is structured pruning. Imagine we successfully remove enough channels, blocks, or other components to produce a valid four-billion-parameter architecture. At FP16, that model requires approximately 8 GB for weights.

This is a hypothetical outcome, not a guarantee that 50% pruning will preserve model quality. If we instead merely set half the weights to zero in the original dense tensor, its stored size may remain 16 GB.

The third strategy is distillation. We train a separate two-billion-parameter student. Its FP16 weights occupy approximately 4 GB.

The fourth strategy combines distillation with quantization. The two-billion-parameter student is subsequently quantized to INT4, reducing ideal weight storage to approximately 1 GB.

Notice that the eight-billion-parameter INT4 model and the two-billion-parameter FP16 student can have similar raw weight-memory requirements.

But their compute requirements are very different.

The 8B model still has the original large architecture, while the 2B student executes a smaller architecture.

This is why distillation can be especially valuable when both low latency and small memory footprint matter.

## 8. Why pruning does not automatically make LLM inference faster

Consider a matrix multiplication:

`Y = X × W.T`

Let:

`W.shape = [4096, 4096]`

The weight matrix contains approximately 16.8 million elements.

Suppose we prune 50% of them by setting individual weights to zero.

If the inference engine still calls an ordinary dense matrix multiplication, it may perform operations over every logical matrix position.

The GPU does not automatically skip all zero weights.

In contrast, if structured pruning produces:

`W_new.shape = [2048, 4096]`

then the matrix multiplication has fewer output channels, and its arithmetic workload decreases.

For a matrix multiplication with input shape `[B, K]` and weight shape `[N, K]`, the approximate floating-point operation count is:

`FLOPs ≈ 2 × B × N × K`

If we halve `N` while keeping everything else fixed, the arithmetic workload of that operation is approximately halved.

But model-level improvements depend on whether subsequent layers can also be reduced and whether overheads remain significant.

This is why structured pruning is often more straightforward to accelerate than unstructured pruning on conventional dense inference backends.

Semi-structured formats such as 2:4 sparsity offer a middle ground, but they require compatible hardware and inference kernels. NVIDIA's TensorRT documentation notes that some sparse-eligible layers may still use dense kernels when those are faster.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)





## 9. Why distillation can reduce compute more naturally

Consider two dense decoder-only transformers.

The first has eight billion parameters, while the second has two billion parameters.

If the smaller student has fewer layers, smaller hidden dimensions, or narrower feed-forward networks, each forward pass generally executes less dense computation.

The student therefore has the potential to reduce weight traffic, arithmetic work, and KV-cache requirements.

However, parameter count alone is not a perfect predictor of inference latency.

A deeper narrow model and a shallower wide model with the same parameter count can behave differently, especially during decoding.

KV-cache memory depends on the number of layers, number of KV heads, head dimension, context length, and datatype.

For example:

`KV bytes = 2 × layers × tokens × kv_heads × head_dim × bytes_per_element`

A student that reduces layers and KV dimensions can reduce runtime cache memory as well as weight memory.

But a student with fewer parameters does not necessarily have a proportionally smaller KV cache. Its particular attention architecture matters.

This is another difference from simple weight-only quantization, which reduces the weights without automatically changing the size of the KV cache.

## 10. Can quantization, pruning, and distillation be combined?

Yes. In fact, they are often complementary because they operate along different dimensions of model efficiency.

Consider a production deployment where we want a compact reasoning assistant.

We could begin with a large teacher model and distill its behavior into a smaller student.

Next, we could structurally prune the student, removing selected channels or layers while fine-tuning to recover performance.

Finally, we could quantize the remaining parameters using AWQ, GPTQ, or QAT, depending on the required accuracy and deployment environment.
<img width="478" height="407" alt="image" src="https://github.com/user-attachments/assets/0566633c-cb85-4626-835a-c20b35c958d8" />


This pipeline is illustrative. There is no universally optimal order.

For some models, pruning before distillation makes sense. For others, distillation can help recover knowledge lost through pruning. QAT can then adapt the remaining model to low-precision execution.

The final model must be evaluated after every major compression stage because errors and architectural changes can interact.

## 11. Which technique should you choose?

The decision depends on the actual bottleneck.

| Your objective                                | First technique to investigate     | Why                                            |
| --------------------------------------------- | ---------------------------------- | ---------------------------------------------- |
| Model does not fit in VRAM                    | Quantization                       | Reduces weight-storage precision               |
| Need a smaller independently deployable model | Distillation                       | Transfers behavior into a smaller architecture |
| Need fewer dense matrix operations            | Structured pruning or distillation | Reduces executable architecture                |
| Have sparse-hardware support                  | 2:4 pruning                        | Exploits sparse acceleration                   |
| Need local CPU inference                      | Quantization and/or distillation   | Reduces memory demand and potentially compute  |
| PTQ causes unacceptable accuracy loss         | QAT                                | Trains the model to tolerate quantization      |
| Model must fit on an edge device              | Distillation plus quantization     | Can reduce both computation and weight memory  |

For a consumer-hardware LLM running through llama.cpp, quantization is generally the simplest first optimization because it often requires no retraining.

For a latency-sensitive production application, a smaller distilled model can be more attractive if its quality is sufficient.

For a GPU serving infrastructure with efficient structured-sparsity kernels, pruning may offer an additional compute-performance advantage.

In every case, the right decision requires actual measurements rather than assumptions based on checkpoint size.

## 12. How to evaluate all three fairly

A valid experiment should begin with one clear task-specific quality baseline.

For example, suppose a model answers technical questions using retrieval-augmented generation. Measure factual correctness, retrieval-grounded answer quality, instruction adherence, and latency on a representative test dataset.

Then compare optimized variants using the same evaluation inputs.

For quantization, inspect changes in model quality, memory usage, and inference throughput.

For pruning, inspect both the sparsity ratio and the actual executable representation. A model with 50% zero weights is not necessarily a model with 50% fewer stored parameters.

For distillation, evaluate whether the student reproduces useful teacher behavior and generalizes beyond the examples used during training.

At the system level, monitor time to first token, time per output token, throughput under concurrency, peak RAM and VRAM, and quality degradation.

For deployed systems, also ensure that compressed checkpoints are integrity-verified, permissions are controlled, and model-version metadata identifies the optimization method and calibration or training dataset version. Quantization and pruning can produce unexpected quality regressions, while distillation can reproduce teacher errors, making traceable evaluation particularly important.

## 13. The distinctions to remember for interviews and inference engineering

Quantization — fewer bits

FP16 to INT4 changes weight precision, usually preserving the original architecture and parameter count. Its primary direct advantage is lower numerical storage and memory traffic.

Pruning — fewer active connections or structures

Removing individual weights creates sparsity. Removing channels or layers can create a physically smaller architecture. Execution speed improves only when the representation and kernels exploit the reduction.

Distillation — transfer learned behavior

A student learns from a teacher using predictions, generated outputs, or internal representations. When the student is smaller, it can reduce both memory requirements and computation.

The distinction becomes clear with three examples.

If someone says, "I converted Llama 8B from FP16 to INT4, so now I have a 2B model," that is incorrect. They still have an 8B model with lower-precision weights.

If someone says, "I pruned 50% of my weights, so inference must be twice as fast," that is also incorrect. Performance depends on whether the runtime actually skips the removed weights.

If someone says, "I trained a 2B model using outputs from a 70B teacher, so I quantized the 70B model," that is incorrect as well. They performed distillation, not quantization.

The three techniques can be summarized in one sentence:

Quantization compresses the numbers, pruning compresses the network structure or connections, and distillation compresses learned behavior into another model.

That distinction is foundational for designing efficient LLM inference systems because it separates three very different optimization targets: numerical representation, computational structure, and model capacity.
