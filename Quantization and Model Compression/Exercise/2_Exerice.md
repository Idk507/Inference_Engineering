""" Exercise 2: Take a real small open-weight model (something that fits on whatever GPU you have, e.g., a 1–3B model) and quantize it with two different libraries/methods — e.g., bitsandbytes INT8 and AutoGPTQ or AutoAWQ INT4. Compare: model file size on disk, VRAM usage at inference, and output quality on a handful of prompts (even just eyeballing coherence). Write down what you'd trade off in a real deployment decision.
"""
# Exercise 2: Real-Model INT8 vs INT4 Quantization Benchmark

For this exercise, we'll use a real 1.54-billion-parameter open-weight model and compare two different quantization methods on identical prompts.

The experiment will measure checkpoint size on disk, peak GPU VRAM, generation latency, throughput, and output quality. It will also preserve the generated answers in a JSON report so that we can inspect where INT4 loses quality, if at all.

## 1. Model and quantization methods

[Qwen Ai Logo Vector Download | Logowik](https://images.openai.com/static-rsc-4/okiq-hk9_J5wcoDk1tZ34gGp7IGNs3Vrin3EYhvVsE34kDub7Un1lHmQU6WUlfbo8mXEA5AP07lYxxeEyISIATZaqp-QNpVeVWSPTaaQLRLsnl3G_kYVR5ku4PUprso3P8uTNV2_SxSI7yV9cvfKedjWIGM1Urn3feSjlICZTKQ?purpose=inline)

Qwen2.5-1.5B-Instruct

1.54B parameters · 28 transformer layers · Apache 2.0

A relatively small instruction-following model suitable for experimenting with local inference, quantization, and generation quality.

[Original model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

INT8 baseline

bitsandbytes LLM.int8()

Loads the original model and quantizes supported linear layers to eight bits, preserving higher-precision handling for sensitive operations.

INT4 comparison

GPTQModel GPTQ INT4

Loads an officially published GPTQ-quantized version of the same model, with four-bit packed weights and calibrated error compensation.

The base checkpoint is approximately 3.09 GB. The official Qwen GPTQ INT4 checkpoint is available at [Qwen2.5-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Qwen/Qwen2.5-1.5B-Instruct at main

+1



We will use [GPTQModel](https://github.com/ModelCloud/GPTQModel) rather than the old AutoGPTQ ecosystem. Likewise, AutoAWQ has been officially deprecated and archived, so I would not build a new experiment around it without a specific compatibility requirement.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



One important experimental distinction: bitsandbytes will quantize the original model on load, whereas the GPTQ INT4 checkpoint has already been quantized by Qwen. We are benchmarking two actual quantized inference implementations, not rerunning the entire GPTQ calibration procedure.

## 2. Expected memory behavior before benchmarking

<img width="493" height="322" alt="image" src="https://github.com/user-attachments/assets/7a0d85c5-0cb6-44c4-8f8f-dc6250a15ab4" />

The actual numbers will differ because quantization scales, unquantized layers, kernel buffers, CUDA allocations, and the KV cache contribute additional memory. In particular, an INT4 checkpoint is not expected to occupy exactly 0.77 GB.

These are estimates, not measured results. I cannot access your local GPU or run these model checkpoints in this session, so the experiment below is designed to generate the actual results on your machine.

## 3. Set up the environment

Because the two libraries can require different CUDA kernels and dependencies, we will use two isolated virtual environments. This avoids one quantization framework unintentionally changing the other's dependency versions.

For Windows PowerShell with Python 3.11:

```
mkdir quantization-exercise-2
cd quantization-exercise-2

py -3.11 -m venv .venv-bnb
py -3.11 -m venv .venv-gptq
```

Install PyTorch using the wheel appropriate for your GPU and driver from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Install that first in each environment, then install the corresponding libraries.

For the bitsandbytes environment:

```
.\.venv-bnb\Scripts\python.exe -m pip install -U pip

.\.venv-bnb\Scripts\python.exe -m pip install `
    transformers accelerate bitsandbytes `
    huggingface_hub safetensors
```

For GPTQModel:

```
.\.venv-gptq\Scripts\python.exe -m pip install -U pip

.\.venv-gptq\Scripts\python.exe -m pip install `
    gptqmodel transformers accelerate `
    huggingface_hub safetensors
```

Run `pip check` in each environment after installation. GPTQModel may also require supported compiler tooling for its CUDA/JIT backend, so check its installation documentation if the selected kernel cannot initialize. The official bitsandbytes documentation lists NVIDIA Turing or newer for its LLM.int8() path.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)





The benchmark will explicitly require an available CUDA GPU rather than silently spilling the model to CPU, since CPU offloading would invalidate the VRAM and GPU-throughput comparison.

## 4. Complete benchmark implementation

We will use one Python file with a `--method` argument. Run it once in each environment so the memory measurements are isolated.

The script downloads the appropriate checkpoint, measures its actual local disk footprint, loads the model on GPU, runs identical prompts using greedy decoding, tracks GPU memory, measures decode throughput, and saves the results as JSON.

It also performs a small warm-up to reduce one-time initialization effects.

Create `benchmark_quantization.py`:

```

"""
Exercise 2: Real-model quantization comparison.

Methods:
    bnb_int8   : bitsandbytes LLM.int8()
    gptq_int4  : pretrained GPTQ INT4 loaded with GPTQModel

Model:
    Qwen2.5-1.5B-Instruct

Outputs:
    results/bnb_int8.json
    results/gptq_int4.json

Metrics:
    - Local checkpoint size
    - CUDA allocated memory after loading
    - Peak CUDA allocated memory during generation
    - Generation latency
    - Generated tokens per second
    - Responses to identical evaluation prompts

Execute each method in its own Python environment.
"""

import argparse
import gc
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


LOGGER = logging.getLogger("quantization_benchmark")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

GPTQ_MODEL = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

# All methods receive exactly the same prompts.
PROMPTS = [
    {
        "id": "factual",
        "prompt": (
            "What is the capital of France? "
            "Answer in one sentence."
        ),
    },
    {
        "id": "reasoning",
        "prompt": (
            "A farmer has 17 sheep. All but 9 run away. "
            "How many sheep remain? Explain briefly."
        ),
    },
    {
        "id": "arithmetic",
        "prompt": (
            "A product costs $80 and has a 25% discount. "
            "What is the final price? Explain."
        ),
    },
    {
        "id": "python",
        "prompt": (
            "Write a Python function that returns True if a "
            "string is a palindrome. Ignore case and spaces. "
            "Include a short example."
        ),
    },
    {
        "id": "inference",
        "prompt": (
            "Explain how the KV cache improves autoregressive "
            "transformer inference in about 100 words."
        ),
    },
    {
        "id": "structured_json",
        "prompt": (
            "Return ONLY valid JSON with fields 'model', "
            "'quantization', and 'bits'. Use values "
            "'transformer', 'weight-only', and 4."
        ),
    },
]


def configure_logging() -> None:
    """Initialize readable console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_cuda() -> None:
    """Require CUDA to ensure meaningful GPU benchmark results."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This benchmark requires an "
            "NVIDIA GPU with compatible PyTorch and quantization "
            "kernels. Do not treat CPU-offloaded results as GPU VRAM "
            "measurements."
        )


def directory_bytes(path: Path) -> int:
    """Return the physical sizes of all regular files in a directory."""
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    )


def download_checkpoint(
    method: str,
) -> Path:
    """
    Download and cache the appropriate Hugging Face checkpoint.

    For bitsandbytes, this downloads the original high-precision
    source. Quantization then occurs during model loading.

    For GPTQ, this downloads an already quantized checkpoint.
    """
    model_id = BASE_MODEL if method == "bnb_int8" else GPTQ_MODEL
    destination = MODEL_DIR / method

    destination.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading checkpoint: %s", model_id)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(destination),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "*.model",
            "*.txt",
            "*.tiktoken",
            "*.py",
        ],
    )

    return destination


def load_quantized_model(
    method: str,
    checkpoint: Path,
):
    """
    Load a quantized causal language model.

    Returns:
        (model, tokenizer)

    Notes:
        GPTQModel supplies its own quantized model loader.
        bitsandbytes quantizes the original model on GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
    )

    if method == "bnb_int8":
        from transformers import (
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            quantization_config=config,
            dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=False,
        )

    elif method == "gptq_int4":
        from gptqmodel import GPTQModel

        model = GPTQModel.from_quantized(
            str(checkpoint),
            device="cuda:0",
            trust_remote_code=False,
        )

    else:
        raise ValueError(f"Unsupported method: {method}")

    model.eval()
    return model, tokenizer


def create_input(
    tokenizer,
    prompt: str,
) -> dict:
    """
    Format a single user prompt using the model's chat template.

    Returns:
        Tokenized PyTorch inputs placed on CUDA device zero.
    """
    messages = [{"role": "user", "content": prompt}]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    return {
        key: value.to("cuda:0")
        for key, value in encoded.items()
    }


def generate_once(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """
    Generate one deterministic response and measure CUDA memory.

    The timing includes prompt processing and autoregressive
    decoding. Therefore tokens/sec is end-to-end generated-token
    throughput, not pure decode-only speed.
    """
    inputs = create_input(tokenizer, prompt)

    # Synchronize asynchronous GPU work before timing.
    torch.cuda.synchronize(0)

    # Reset peak counters while preserving existing allocations.
    torch.cuda.reset_peak_memory_stats(0)

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - start

    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, input_length:]
    generated_tokens = int(generated_ids.numel())

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)

    return {
        "prompt": prompt,
        "response": response,
        "input_tokens": int(input_length),
        "generated_tokens": generated_tokens,
        "latency_seconds": round(elapsed, 4),
        "end_to_end_generated_tokens_per_second": round(
            generated_tokens / elapsed,
            3,
        ) if elapsed > 0 else None,
        "peak_allocated_gib": round(
            peak_allocated / (1024 ** 3),
            4,
        ),
        "peak_reserved_gib": round(
            peak_reserved / (1024 ** 3),
            4,
        ),
    }


def runtime_information() -> dict:
    """Collect hardware and framework details for reproducibility."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(
            torch.cuda.get_device_capability(0)
        ),
        "gpu_total_gib": round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            3,
        ),
    }


def main() -> None:
    """Execute a complete single-method quantization benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark bitsandbytes INT8 or GPTQ INT4."
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=["bnb_int8", "gptq_int4"],
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")

    configure_logging()
    ensure_cuda()

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    RESULT_DIR.mkdir(exist_ok=True, parents=True)

    LOGGER.info("Method: %s", args.method)
    LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))

    checkpoint = download_checkpoint(args.method)
    checkpoint_bytes = directory_bytes(checkpoint)

    # Measure allocations after the model has been loaded.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    model, tokenizer = load_quantized_model(
        args.method,
        checkpoint,
    )

    torch.cuda.synchronize(0)

    loaded_allocated_gib = (
        torch.cuda.memory_allocated(0) / (1024 ** 3)
    )
    loaded_reserved_gib = (
        torch.cuda.memory_reserved(0) / (1024 ** 3)
    )

    LOGGER.info(
        "Loaded model memory: %.3f GiB",
        loaded_allocated_gib,
    )

    # Warm-up reduces kernel initialization and caching effects.
    LOGGER.info("Executing warm-up generation.")

    generate_once(
        model,
        tokenizer,
        "Write one sentence about artificial intelligence.",
        max_new_tokens=16,
    )

    records = []

    for index, item in enumerate(PROMPTS, start=1):
        LOGGER.info(
            "Evaluating prompt %s/%s: %s",
            index,
            len(PROMPTS),
            item["id"],
        )

        measurement = generate_once(
            model,
            tokenizer,
            item["prompt"],
            args.max_new_tokens,
        )

        measurement["id"] = item["id"]
        records.append(measurement)

    average_latency = sum(
        record["latency_seconds"]
        for record in records
    ) / len(records)

    peak_vram = max(
        record["peak_allocated_gib"]
        for record in records
    )

    total_tokens = sum(
        record["generated_tokens"]
        for record in records
    )

    total_seconds = sum(
        record["latency_seconds"]
        for record in records
    )

    report = {
        "method": args.method,
        "source_model": BASE_MODEL,
        "checkpoint_model": (
            BASE_MODEL
            if args.method == "bnb_int8"
            else GPTQ_MODEL
        ),
        "hardware": runtime_information(),
        "checkpoint_directory_gib": round(
            checkpoint_bytes / (1024 ** 3),
            4,
        ),
        "checkpoint_directory_bytes": checkpoint_bytes,
        "checkpoint_size_kind": (
            "unquantized_source"
            if args.method == "bnb_int8"
            else "quantized_checkpoint"
        ),
        "loaded_allocated_gib": round(
            loaded_allocated_gib,
            4,
        ),
        "loaded_reserved_gib": round(
            loaded_reserved_gib,
            4,
        ),
        "peak_allocated_gib": round(peak_vram, 4),
        "average_latency_seconds": round(
            average_latency,
            4,
        ),
        "aggregate_generated_tokens_per_second": round(
            total_tokens / total_seconds,
            3,
        ) if total_seconds > 0 else None,
        "max_new_tokens": args.max_new_tokens,
        "prompts": records,
    }

    output = RESULT_DIR / f"{args.method}.json"

    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print(f"QUANTIZATION BENCHMARK: {args.method}")
    print("=" * 65)

    print(
        "Checkpoint directory (GiB):",
        report["checkpoint_directory_gib"],
    )
    print(
        "Loaded CUDA allocations (GiB):",
        report["loaded_allocated_gib"],
    )
    print(
        "Peak CUDA allocations (GiB):",
        report["peak_allocated_gib"],
    )
    print(
        "Average generation latency (s):",
        report["average_latency_seconds"],
    )
    print(
        "Aggregate generated tokens/s:",
        report["aggregate_generated_tokens_per_second"],
    )

    for record in records:
        print(f"\n[{record['id']}]")
        print(record["response"])

    print(f"\nSaved report: {output.resolve()}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as exc:
        LOGGER.exception("Benchmark failed.")
        raise SystemExit(str(exc)) from exc
```

### Important measurement distinctions

For bitsandbytes INT8, `checkpoint_directory_gib` measures the original downloaded source checkpoint, not the quantized INT8 artifact. This is intentional so that the report does not mislabel a 3 GB source checkpoint as a 3 GB INT8 model.

For the GPTQ variant, it measures the actual downloaded prequantized checkpoint directory.

To compare quantized file sizes directly, we must additionally serialize the bitsandbytes INT8 model. We will do this next.

The CUDA allocated-memory metric measures GPU tensor allocations tracked by PyTorch. It is not identical to the total memory shown by `nvidia-smi`, which also includes CUDA context and other driver-level allocations. The benchmark records both allocated and reserved memory to make that distinction visible.

## 5. Measure the actual quantized INT8 checkpoint size

Use the following separate script to save the bitsandbytes INT8 model, then measure its serialized file size.

Create `save_bnb_int8.py`:

```

"""Serialize and measure an INT8 bitsandbytes checkpoint."""

import gc
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


SOURCE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT = Path("models/bnb_int8_saved")


def main() -> None:
    """Quantize, save, and measure the INT8 model artifact."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL,
        quantization_config=config,
        dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_MODEL,
        trust_remote_code=False,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Requires a compatible transformers/bitsandbytes version
    # that supports serialization of quantized parameters.
    model.save_pretrained(
        OUTPUT,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(OUTPUT)

    total_bytes = sum(
        file.stat().st_size
        for file in OUTPUT.rglob("*")
        if file.is_file()
    )

    print(f"Saved checkpoint: {OUTPUT.resolve()}")
    print(f"Serialized size: {total_bytes / (1024**3):.3f} GiB")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(
            f"INT8 checkpoint serialization failed: {exc}"
        ) from exc
```

Hugging Face documents `save_pretrained()` support for recent compatible versions of bitsandbytes and Transformers. If your installed combination cannot serialize the model, report that as a compatibility limitation rather than substituting the FP16 source size.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



## 6. Execute both benchmarks

Run bitsandbytes first:

```
.\.venv-bnb\Scripts\python.exe benchmark_quantization.py `
  --method bnb_int8 `
  --max-new-tokens 128
```

Serialize the INT8 artifact:

```
.\.venv-bnb\Scripts\python.exe save_bnb_int8.py
```

Run GPTQ INT4 separately:

```
.\.venv-gptq\Scripts\python.exe benchmark_quantization.py `
  --method gptq_int4 `
  --max-new-tokens 128
```

If GPTQModel rejects Qwen's historical GPTQ checkpoint layout, that is a loader/checkpoint compatibility issue, not proof that GPTQ cannot run on the hardware. The official GPTQModel loader supports quantized models, but older model repositories can use different packing conventions and metadata. A GPTQModel-native checkpoint produced from the same Qwen source is the cleaner fallback. Do not substitute a different base model, because it would invalidate the comparison.

For a stricter latency comparison, run the benchmark multiple times, randomize method order between runs, and report medians. The current script reports end-to-end generation throughput; it does not independently isolate prefill from decode.

## 7. Summarize the results automatically

Create `compare_results.py` to combine both JSON reports into one readable table and inspect their outputs side by side.

```
    int4_disk = disk_gib(MODELS / "gptq_int4")    metrics = [        ("Quantized disk GiB", int8_disk, int4_disk),        (            "Loaded VRAM GiB",            int8["loaded_allocated_gib"],            int4["loaded_allocated_gib"],        ),        (            "Peak VRAM GiB",            int8["peak_allocated_gib"],            int4["peak_allocated_gib"],        ),        (            "Average latency seconds",            int8["average_latency_seconds"],            int4["average_latency_seconds"],        ),        (            "Aggregate generated tokens/s",            int8["aggregate_generated_tokens_per_second"],            int4["aggregate_generated_tokens_per_second"],        ),    ]    print(f"{'Metric':<35} {'INT8':>14} {'GPTQ INT4':>14}")    print("-" * 65)    for name, a, b in metrics:        a_text = f"{a:.4f}" if a is not None else "Unavailable"        b_text = f"{b:.4f}" if b is not None else "Unavailable"        print(f"{name:<35} {a_text:>14} {b_text:>14}")    int4_prompts = {        item["id"]: item        for item in int4["prompts"]    }    for item in int8["prompts"]:        other = int4_prompts.get(item["id"])        if other is None:            raise ValueError(f"Missing GPTQ prompt: {item['id']}")        print("\n" + "=" * 70)        print(f"PROMPT: {item['id']}")        print("=" * 70)        print("\nINT8 RESPONSE:\n")        print(item["response"])        print("\nGPTQ INT4 RESPONSE:\n")        print(other["response"])if __name__ == "__main__":    try:        main()    except (        OSError,        ValueError,        KeyError,        TypeError,    ) as exc:        raise SystemExit(f"Comparison failed: {exc}") from exc
```

Execute it using either environment:

```
.\.venv-bnb\Scripts\python.exe compare_results.py
```

## 8. Evaluate output quality

For this small exercise, we can use a consistent qualitative rubric rather than building a complete benchmark suite.

Inspect factual correctness, reasoning, instruction following, code correctness, and coherence.

Interactive qualitative evaluation sheet

After running both models, score each dimension from 1 (poor) to 5 (excellent). This is a manual assessment, not an automated correctness metric.

| Criterion             | INT8 | GPTQ INT4 |
| --------------------- | ---- | --------- |
| Factual accuracy      | —    | —         |
| Reasoning correctness | —    | —         |
| Code correctness      | —    | —         |
| Instruction following | —    | —         |
| Coherence             | —    | —         |
| Average / 5           | —    | —         |

&#x20;Copy evaluation as CSV

The JSON-formatting prompt is particularly useful because you can automatically parse the answer and verify whether it contains exactly the required fields and values. The Python prompt can be evaluated by executing the generated function in a restricted test environment and checking known examples. These objective checks are stronger than subjective coherence scores.

For a production comparison, five or six prompts are not sufficient to establish a reliable quality difference. They are appropriate for learning the workflow, but deployment selection should use many representative, held-out prompts and deterministic correctness checks.

## 9. What I would trade off in a real deployment

Choose bitsandbytes INT8 when quality preservation is the priority

I would favor INT8 if the application involves precise reasoning, code generation, structured outputs, or other tasks where a small reduction in accuracy matters more than memory savings. However, this choice still requires actual quality measurements. INT8 is not guaranteed to outperform a well-calibrated GPTQ INT4 model on every prompt.

Choose GPTQ INT4 when GPU memory is the primary constraint

If INT4 meets the application's quality threshold, its lower weight-memory requirement may free capacity for larger batches, longer contexts, or additional concurrent requests. This can improve overall serving economics, provided efficient kernels are available.

Choose based on measured latency, not bit width alone

Four-bit weights do not automatically produce faster inference. GPTQ may require specialized dequantization kernels, while bitsandbytes INT8 has its own handling of activation outliers. The faster choice depends on the GPU, sequence length, batch size, and implementation.

For this particular 1.5B experiment, my starting hypothesis would be that GPTQ INT4 offers lower weight storage and may provide sufficient quality for lightweight interactive workloads. I would retain INT8 if the INT4 model shows consistent mistakes in reasoning, coding, or structured outputs. The benchmark must determine whether that hypothesis holds.

## 10. Final experiment report

After execution, your completed report should contain the following measurements.

| Metric                     | bitsandbytes INT8                   | GPTQ INT4          |
| -------------------------- | ----------------------------------- | ------------------ |
| Base model                 | Qwen2.5-1.5B-Instruct               | Same base model    |
| Weight precision           | 8-bit with mixed-precision handling | 4-bit GPTQ         |
| Actual quantized disk size | Measured by script                  | Measured by script |
| Loaded GPU allocations     | Measured by script                  | Measured by script |
| Peak GPU allocations       | Measured by script                  | Measured by script |
| Average latency            | Measured by script                  | Measured by script |
| Generated tokens/s         | Measured by script                  | Measured by script |
| Output quality             | Manual and deterministic checks     | Same checks        |

The key lesson from Exercise 2 is that quantization should be evaluated as an end-to-end inference-system decision, not merely as a model-compression ratio.

Disk size tells us how economically we can store and distribute the model. VRAM usage tells us whether it fits and how much room remains for KV-cache capacity and concurrency. Throughput and latency measure whether the quantized kernels improve actual inference performance. Output quality determines whether the savings are acceptable.

For a production deployment, I would choose the smallest quantized model that consistently satisfies the application's quality requirements and latency SLOs, with checkpoint versioning, integrity verification, GPU telemetry, and repeatable regression evaluations.

This exercise provides the code and evaluation structure to reach that decision with measured evidence rather than assumptions.
