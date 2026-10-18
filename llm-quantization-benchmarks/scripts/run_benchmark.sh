#!/usr/bin/env bash
# Run an FP16 vs 4-bit llama.cpp comparison when the locked settings and
# local model files exist. Incomplete settings leave every result Pending.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SETTINGS="$ROOT/configs/benchmark_settings.json"
BENCH="$ROOT/benchmarks"

python3 - "$SETTINGS" <<'PY'
import json
import sys
from pathlib import Path

settings = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = ("base_model", "hardware", "context_length", "offload_layers")
missing = [key for key in required if settings.get(key) in (None, "", [])]
if missing:
    print("Quantization benchmark is not configured:", ", ".join(missing))
    print("File size, RAM, GPU VRAM, and tokens per second stay Pending.")
    print("Those four measurements are recorded separately.")
    sys.exit(2)
if settings.get("prompt_tokens") != 512 or settings.get("generation_tokens") != 128:
    print("Prompt and generation lengths must stay 512 and 128 for this protocol.")
    sys.exit(2)
if int(settings.get("repeats") or 0) < 2:
    print("Repeats must be at least 2.")
    sys.exit(2)
print("Settings name a model and a machine. No measurement has been written by this check.")
PY

F16="${F16_MODEL:-models/model-f16.gguf}"
Q4="${Q4_MODEL:-models/model-q4.gguf}"

if [[ -f "$F16" && -f "$Q4" ]]; then
  echo "File sizes (disk only; not RAM and not VRAM):"
  ls -lh "$F16" "$Q4"
else
  echo "Model files are absent. File size stays Pending."
fi

LLAMA_BENCH="${LLAMA_BENCH:-build/bin/llama-bench}"
if [[ -x "$LLAMA_BENCH" && -f "$F16" && -f "$Q4" ]]; then
  "$LLAMA_BENCH" -m "$F16" -m "$Q4" -p 512 -n 128 -r 5 -o csv > "$BENCH/throughput.csv"
  echo "Wrote throughput to $BENCH/throughput.csv"
  echo "Throughput is not a memory measurement. Sample RAM and GPU VRAM separately."
else
  echo "llama-bench or model files are absent. Tokens per second stay Pending."
fi

echo "RAM: sample the process RSS during an equivalent inference session."
echo "GPU VRAM: sample nvidia-smi during that same session, if a GPU is in use."
echo "Do not copy file size, RSS, VRAM, or tokens per second into each other's columns."
