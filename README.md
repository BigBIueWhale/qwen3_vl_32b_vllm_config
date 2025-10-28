# qwen3_vl_32b_vllm_config

> Production-ready vLLM “just works” setup for [Qwen3-VL-32B-Thinking (AWQ-INT4)](https://huggingface.co/cpatonn/Qwen3-VL-32B-Thinking-AWQ-4bit) on **Ubuntu 24.04 LTS + RTX 5090** — tuned for my personal server and exact hardware.  
> Repo link: https://github.com/BigBIueWhale/personal_server

---

# ⚠️ **MODEL QUALITY DISCLAIMER**

🔴🔴🔴

**Qwen3-VL-32B-Thinking is configured here in text-only mode (vision modules never load) to fit within 32GB VRAM on RTX 5090.** This means the VL checkpoint behaves like a plain 32B text model in memory and runtime. Any request containing images will be rejected (by design), keeping your runs deterministic and VRAM-predictable.

Thinking budget supported (default 8,192-token budget) inspired by a request-side [workaround approach](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html#thinking-budget) due to the complete lack of support of vLLM (and llama.cpp, and OpenWebUI) in thinking budget implementation.

Thinking budget is crucial for us to ensure that the model has enough context length to actually respond.

---

## What this is

This project pins **server‑side defaults** for vLLM so that **the CLI** (or any OpenAI‑compatible client) doesn’t have to pass a dozen sampling knobs. It’s built for **my specific machine**:
- **GPU:** RTX 5090 (Blackwell, sm_120)
- **Driver:** 580.xx (open)  
- **CUDA:** 13.0
- **OS:** Ubuntu 24.04 LTS

With this exact combo, **Qwen3-VL-32B-Thinking (AWQ-INT4)** runs **in 4-bit quantized precision** and stays stable. We intentionally use **vLLM** (not Ollama here) because llama.cpp doesn't support Qwen3-VL yet (as of 28 October 2025). And AWQ quantizations although they require a calibration dataset, the result seems to be OK.

> ⚠️ **Why custom config is needed (philosophy):** Pretty much all models **require very specific runtime flags** to behave correctly. For Qwen3-VL-32B-Thinking, using the wrong defaults can make the model **quietly underperform (“silently stupid”)** without obvious errors. This repo bakes in the correct settings — notably text-only mode with `--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0` — plus conservative decoding defaults. Clients can still tweak a couple of knobs (e.g., `temperature`), but the server remains the single source of truth for everything else.

---

## Contents

- **[`config.yaml`](./config.yaml)** — vLLM server config. Sets the model, networking, Qwen3‑specific flags, and **complete** default decoding parameters (temperature/top‑p/top‑k/penalties/max_tokens, etc.). Clients inherit these unless they explicitly override a field.
- **[`run_vllm.sh`](./run_vllm.sh)** — one‑shot “install” script that pulls the NVIDIA vLLM container and creates a persistent Docker container named `qwen3_vllm` with `--restart unless-stopped`. Run it once; use `docker start qwen3_vllm` on reboots.
- **[`chat_cli.py`](./chat_cli.py)** — a simple Python CLI for multi-turn chatting with the model, supporting streaming and the thinking budget workaround (default 8192 tokens).
- **[`UNINSTALL.md`](./UNINSTALL.md)** — how to stop/remove the container and (optionally) delete the Docker image and HF cache.

---

## Quick start

1. **Optional:** export your HF token (for first‑time download from Hugging Face):
   ```bash
   export HF_TOKEN=hf_************************
   ```

2. **Install / create the server container** (one time):
   ```bash
   chmod +x ./run_vllm.sh
   ./run_vllm.sh
   ```

3. **Test the endpoint:**
   ```bash
   # From the host:
   curl -s http://172.17.0.1:8000/v1/models | jq .
   ```

4. **Chat with the CLI:** 
   Install openai: `pip install openai`
   Then run `python chat_cli.py` for thinking mode, or `python chat_cli.py --thinking-budget 0` for no thinking (although in that case I would recommend `Qwen3-VL-32B-Instruct`).

---

## Why these settings?

* **Qwen3‑specific requirement:** Text-only mode (enforced in `run_vllm.sh` with `--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0`) prevents loading the vision tower/projector entirely, reducing static VRAM and making your attention/KV use identical to a pure 32B text model at the same context.
* **Large context with realistic concurrency:** The config sets `max-model-len: 18000` and **`max-num-seqs: 1`**. With 32B params (AWQ-INT4) on an RTX 5090, this leaves safe headroom for KV cache at startup, which practically limits safe concurrency to ~1 request at 18k context.
* **Decoding defaults:** We pin **temperature 1.0 / top_p 0.95 / top_k 20**, with repetition_penalty 1.0 / presence_penalty 1.5 / frequency_penalty 0.0 and a safe `max_new_tokens` default (2048). These mirror the Text preset from the model card. Clients can override per‑request; the server supplies sensible fallback behavior.
* The containerized vLLM server listens on `0.0.0.0:8000` **inside the container**, and the Docker publish in `run_vllm.sh` maps it to **`172.17.0.1:8000` on the host** (the Docker bridge gateway). This keeps the API reachable to the host while not exposing it on your host’s primary interfaces. If you ever need LAN exposure, change `BIND` to `0.0.0.0`.

### Using the CLI

The `chat_cli.py` provides an interactive chat interface with support for the thinking budget workaround as described in the Qwen documentation.

- Run with default thinking budget (8192 tokens): `python chat_cli.py`
- Disable thinking: `python chat_cli.py --thinking-budget 0`
- Custom budget: `python chat_cli.py --thinking-budget 4096`

The CLI supports streaming responses and maintains conversation history for multi-turn interactions.

### Notes about the NVIDIA vLLM image

* **Flash‑Attention backend** is used automatically on Blackwell; no extra flags required.
* **Chunked prefill:** vLLM auto‑enables it for contexts larger than 32K and will warn that it “might not work with some features/models.” Leave it on unless you observe issues; then relaunch with `--enable-chunked-prefill=False`.

---

## Compatibility notes

- Uses **NVIDIA vLLM container** (`nvcr.io/nvidia/vllm:25.10-py3`) built for **CUDA 13** and **Blackwell** (RTX 50‑series). No wheel rebuilds, no mismatched compute capability issues.
- Assumes you already installed the **NVIDIA Container Toolkit** and your driver exposes the GPU inside containers via `--gpus all`.

---

## Operating the server (start / stop / pause)

### Check status

```bash
docker ps
curl -s http://172.17.0.1:8000/v1/models | jq .
docker logs -f qwen3_vllm
```

### Stop (frees VRAM)

> Use **stop** to fully release GPU memory.

```bash
docker stop qwen3_vllm
```

Verify with:

```bash
nvidia-smi
```

### Start (loads model and uses VRAM again)

```bash
docker start -a qwen3_vllm   # attach logs
# or
docker start qwen3_vllm
```

### Restart

```bash
docker restart qwen3_vllm
```

### Pause vs Stop

* `docker pause` **does not free VRAM** (the process is frozen but GPU memory stays allocated).
* `docker stop` **does free VRAM** (the process exits and releases the GPU).

```bash
docker pause qwen3_vllm
docker unpause qwen3_vllm
```

### Auto-restart policy

The container is created with `--restart unless-stopped`.

* Disable auto-restart:

```bash
docker update --restart=no qwen3_vllm
```

* Re-enable:

```bash
docker update --restart=unless-stopped qwen3_vllm
```

### Remove and recreate (if you want a clean slate)

```bash
docker stop qwen3_vllm || true
docker rm qwen3_vllm || true
./run_vllm.sh
```

## License

This repo contains only configuration and scripts I wrote for my own server layout. Model weights are **not** distributed here; they are fetched from Hugging Face under their license terms.
