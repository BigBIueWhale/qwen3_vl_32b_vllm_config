#!/usr/bin/env bash
set -Eeuo pipefail

# Create a persistent vLLM container for Qwen3-VL-32B-Thinking (AWQ-INT4).
# Not a system service; just a normal docker container you can start/stop.

IMAGE="nvcr.io/nvidia/vllm:25.10-py3"
NAME="qwen3_vllm"
PORT="8000"             # container port (OpenAI-compatible API)
# Bind to docker port, so that it's accessible by OpenWebUI docker instance
# instead of being exposed to entire internet.
BIND="172.17.0.1"
CONFIG="$(pwd)/config.yaml"
HF_CACHE="${HOME}/.cache/huggingface"

echo "Pulling image ${IMAGE} ..."
docker pull "${IMAGE}"

echo "Ensuring cache and config exist ..."
mkdir -p "${HF_CACHE}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config.yaml in $(pwd)"
  exit 1
fi

# Remove any stopped container with same name (idempotent install)
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Container ${NAME} already exists; removing to recreate ..."
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
fi

echo "Creating container ${NAME} ..."
docker create \
  --name "${NAME}" \
  --gpus all \
  --ipc=host \
  --restart unless-stopped \
  -p "${BIND}:${PORT}:8000" \
  -v "${CONFIG}:/workspace/config.yaml:ro" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
  "${IMAGE}" \
  vllm serve --config /workspace/config.yaml \
    --limit-mm-per-prompt.image 0 \
    --limit-mm-per-prompt.video 0

echo "Starting ${NAME} ..."
docker start -a "${NAME}"
