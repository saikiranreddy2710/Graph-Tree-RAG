# FP16 vs 4-bit comparison

Status: not run.

Both columns must name the same base model, the same hardware, the same context length, and the same offloading configuration. Repeats are required. Empty cells stay Pending.

| Measurement | FP16 | 4-bit | Unit | What it proves |
| --- | --- | --- | --- | --- |
| Model file size | Pending | Pending | bytes | Storage on disk |
| RAM | Pending | Pending | bytes | Process memory during inference |
| GPU VRAM | Pending | Pending | bytes | GPU memory during inference |
| Tokens per second | Pending | Pending | tokens/s | llama-bench throughput |

| Field | Value |
| --- | --- |
| Base model | Not yet run |
| Hardware | Not yet run |
| Context length | Not yet run |
| Offload layers | Not yet run |
| Prompt tokens | 512 |
| Generation tokens | 128 |
| Repeats | 5 |
| Experiment date | Not yet run |
| Code commit | Not yet run |

File size is not RAM. RAM is not GPU VRAM. Tokens per second is not a memory measurement.
