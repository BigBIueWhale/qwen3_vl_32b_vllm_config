# Uninstall / Cleanup

Stop and remove the persistent container:

```bash
docker stop qwen3_vllm
docker rm qwen3_vllm
```

Delete the local image (rebuilds on next run):

```bash
docker rmi local/vllm:25.09-py3
```

Optional Docker build cache cleanup:

```bash
docker builder prune -f
```

Optional Hugging Face cache cleanup (removes downloaded weights):

```bash
rm -rf ~/.cache/huggingface/hub/*
```
