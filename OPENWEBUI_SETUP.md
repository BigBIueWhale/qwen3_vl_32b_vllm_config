# OpenWebUI Setup (vLLM backend)

These steps match my deployment style (loopback bind + Docker bridge access).

## 1) Run OpenWebUI (if not already)
I run OpenWebUI isolated and let it reach the host via the Docker host‑gateway mapping:

```bash
docker run -d   -p 127.0.0.1:3000:8080   --add-host=host.docker.internal:host-gateway   -v open-webui:/app/backend/data   --name open-webui --restart always   ghcr.io/open-webui/open-webui:v0.6.25
```

## 2) Point OpenWebUI to vLLM

In OpenWebUI admin settings: http://127.0.0.1:3000/admin/settings/connections go to: `OpenAI API` -> `Edit Connection` and set:

- Base URL: `http://host.docker.internal:8000/v1` (instead of default value: `https://api.openai.com/v1`)
- Connection type: Internal
- API Key: Empty
- Model name: `nvidia/NVIDIA-Nemotron-Nano-12B-v2`

This keeps both containers on loopback while still allowing OpenWebUI to reach vLLM via the Docker host gateway.

## 3) Suggested app settings (optional)

- **LLM Providers → Default**: set to OpenAI Compatible provider above.
- **Task model**: a small non‑reasoning model is fine, but Nemotron works too.
- **Disable “Code Interpreter”** (I keep it off) and keep **“Enable Code Execution”** on `pyodide` if you like running code blocks inline.
