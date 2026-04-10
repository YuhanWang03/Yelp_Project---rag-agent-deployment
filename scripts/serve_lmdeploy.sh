#!/bin/bash
# LMDeploy server startup script
#
# Usage (run on Colab A100):
#   bash scripts/serve_lmdeploy.sh fp16   # Qwen2.5-7B fp16, pytorch backend
#   bash scripts/serve_lmdeploy.sh awq    # Qwen2.5-7B AWQ, turbomind backend
#
# The server exposes an OpenAI-compatible API at http://localhost:23333
# Wait ~60s after starting before sending requests.

MODE=${1:-"awq"}
PORT=23333

case "$MODE" in
  fp16)
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    BACKEND="pytorch"
    EXTRA_ARGS=""
    ;;
  awq)
    MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
    BACKEND="turbomind"
    EXTRA_ARGS="--model-format awq"
    ;;
  *)
    echo "Unknown mode: $MODE. Use 'fp16' or 'awq'."
    exit 1
    ;;
esac

echo "========================================"
echo "Starting LMDeploy API server"
echo "  Mode    : $MODE"
echo "  Model   : $MODEL"
echo "  Backend : $BACKEND"
echo "  Port    : $PORT"
echo "========================================"

lmdeploy serve api_server "$MODEL" \
    --server-port "$PORT" \
    --backend "$BACKEND" \
    $EXTRA_ARGS
