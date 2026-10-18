# LLM quantization benchmark

Independent reproduction scaffold for comparing one base model as FP16 GGUF and as 4-bit GGUF with llama.cpp. This directory is not part of the Graph-Tree RAG result table, and it contains no lab data.

No run has been recorded. Every measurement below is Pending.

File size, runtime memory, and throughput are different measurements. Do not substitute one for another, and do not collapse them into a single reduction claim.

| Measurement | What it proves | FP16 | 4-bit |
| --- | --- | --- | --- |
| Model file size | Storage on disk | Pending | Pending |
| RAM | Process memory during an equivalent inference session | Pending | Pending |
| GPU VRAM | GPU memory during that same session | Pending | Pending |
| Tokens per second | Inference throughput from `llama-bench` | Pending | Pending |

## Rules

- Use the same base model for both files.
- Use the same hardware, context length, and offloading configuration.
- Repeat the throughput run. `configs/benchmark_settings.json` sets prompt length 512, generation length 128, and 5 repeats, matching `llama-bench -p 512 -n 128 -r 5`.
- `llama-bench` reports throughput. It does not measure peak RAM or peak GPU VRAM. Record those with the process sampler and `nvidia-smi` during a separate equivalent session.
- Leave a field null in the settings file until you know it. The script exits without writing a result when the model or hardware is unset.

## Layout

```text
llm-quantization-benchmarks/
├── README.md
├── benchmarks/
│   ├── fp16_results.csv
│   ├── q4_results.csv
│   └── comparison.md
├── scripts/
│   └── run_benchmark.sh
└── configs/
    └── benchmark_settings.json
```

`benchmarks/fp16_results.csv` and `benchmarks/q4_results.csv` are schemas only. They do not contain measured rows.

## Run

```bash
# Fill base_model, hardware, context_length, and offload_layers first.
./llm-quantization-benchmarks/scripts/run_benchmark.sh
```

File-size listing, when both GGUF files exist:

```bash
ls -lh models/model-f16.gguf models/model-q4.gguf
```

Throughput, when `llama-bench` and both model files exist:

```bash
./build/bin/llama-bench \
  -m models/model-f16.gguf \
  -m models/model-q4.gguf \
  -p 512 \
  -n 128 \
  -r 5 \
  -o csv > llm-quantization-benchmarks/benchmarks/throughput.csv
```

The script writes throughput only to `benchmarks/throughput.csv`. It does not copy that number into the RAM or VRAM columns.
