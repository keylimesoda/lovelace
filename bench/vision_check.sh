#!/usr/bin/env bash
# Vision-profile check (one GPU-lock hold): largest context that starts with the mmproj loaded, then one image request.
set -uo pipefail
BIN=$HOME/Work/lovelace/engine/build-sm89/bin
IMG=$HOME/Work/ninfer-4090-lin-hybrid/bench/fixtures/ttft/media/load_00.png
OUT=$HOME/Work/lovelace/bench/results/vision; mkdir -p "$OUT"
for C in 327680 294912 262144 229376; do
  docker rm -f vis >/dev/null 2>&1
  docker run -d --name vis --gpus all --network host -v "$BIN:/e" -v "$HOME/models:/models:ro" lovelace-engine-runtime:cuda13.1 \
    /e/llama-server -m /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf --mmproj /models/ref/mmproj-Qwen3.8-27B-Q8_0.gguf \
    -ngl 99 -np 6 --kv-unified -c "$C" -ctk q5_0 -ctv q5_0 --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q5_0 -ctvd q5_0 \
    -fa on -ub 1024 -b 2048 --jinja --reasoning-preserve --cache-ram 4096 -ctxcp 8 --host 127.0.0.1 --port 18200 >/dev/null
  ok=0
  for i in $(seq 90); do
    [[ "$(docker inspect -f '{{.State.Running}}' vis 2>/dev/null)" == true ]] || break
    curl -sf 127.0.0.1:18200/health >/dev/null && { ok=1; break; }; sleep 2
  done
  if (( ok )); then
    echo "fits: -c $C, $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" | tee "$OUT/fit.txt"
    b64=$(base64 -w0 "$IMG")
    curl -s 127.0.0.1:18200/v1/chat/completions -H 'Content-Type: application/json' -d @- > "$OUT/response.json" <<EOF
{"model":"x","temperature":0,"max_tokens":300,"chat_template_kwargs":{"enable_thinking":false},
 "messages":[{"role":"user","content":[{"type":"text","text":"Describe this image in two sentences, including any numbers or labels you can read."},
 {"type":"image_url","image_url":{"url":"data:image/png;base64,$b64"}}]}]}
EOF
    python3 -c "import json;d=json.load(open('$OUT/response.json'));print(d['choices'][0]['message']['content']);print(d.get('timings',{}))" | tee "$OUT/answer.txt"
    docker logs vis > "$OUT/server.log" 2>&1; docker rm -f vis >/dev/null; exit 0
  fi
  echo "no fit at -c $C"
  docker rm -f vis >/dev/null 2>&1
done
exit 1
