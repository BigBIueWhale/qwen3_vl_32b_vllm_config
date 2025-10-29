#!/usr/bin/env bash
set -Eeuo pipefail
# Build a local image that behaves like nvcr.io/nvidia/vllm but with newer bits.
# Tag matches the style you use and is referenced by run_vllm.sh below.
docker build --pull -f Dockerfile.vllm-local -t local/vllm:25.09-py3 .
echo "Built local/vllm:25.09-py3"
