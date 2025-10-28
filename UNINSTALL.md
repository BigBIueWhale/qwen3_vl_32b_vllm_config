# Uninstall / Cleanup

Stop and remove the persistent container:

```bash
docker stop qwen3_vllm || true
docker rm qwen3_vllm || true
```

Optionally delete the image (re-downloads next install):

```bash
docker rmi nvcr.io/nvidia/vllm:25.10-py3
```

Optional cache cleanup (removes downloaded weights):

```bash
rm -rf ~/.cache/huggingface/hub/*
```
