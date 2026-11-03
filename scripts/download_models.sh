#!/bin/bash
# Downloads models into models/ via HF resolve URLs. Idempotent (skips finished files).
set -u
cd "$(dirname "$0")/.."
BASE=https://huggingface.co
mkdir -p models/embeddinggemma-2

function fetch() { # repo rel out
  local url="$BASE/$1/resolve/main/$2" out="$3"
  local remote_len local_len
  remote_len=$(curl -sIL -o /dev/null -w '%{http_code} %{size_download}' --max-time 15 "$url" 2>/dev/null)
  # resolve redirect: use content-length from a fresh HEAD with -L compiled redirects
  remote_len=$(curl -sIL -L "$url" 2>/dev/null | tr -d '\r' | awk 'BEGIN{IGNORECASE=1} /^content-length:/{last=$2} END{print last}')
  local_len=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ -n "$remote_len" ] && [ "$local_len" = "$remote_len" ]; then echo "[skip] $out ($local_len bytes)"; return 0; fi
  echo "[dl] $out (have $local_len, need $remote_len)"
  rm -f "$out"
  curl -fSsL --retry 3 "$url" -o "$out" || { echo "[FAIL] $out"; return 1; }
  echo "[done] $out"
}

( fetch google/embeddinggemma-2 model.safetensors models/embeddinggemma-2/model.safetensors ) &
for f in config.json tokenizer.json tokenizer.model tokenizer_config.json preprocessor_config.json processor_config.json chat_template.jinja modules.json sentence_bert_config.json config_sentence_transformers.json; do
  ( fetch google/embeddinggemma-2 "$f" "models/embeddinggemma-2/$f" ) &
done
wait
echo "ALL DONE"
ls -la models/embeddinggemma-2 | grep -v '^d' | grep -v total
