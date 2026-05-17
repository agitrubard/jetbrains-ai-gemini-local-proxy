# jetbrains-ai-gemini-local-proxy

A tiny, zero-buffer **FastAPI + httpx** proxy that makes **Google Gemini** work
reliably as a **JetBrains AI Assistant** "OpenAI-compatible" (BYOK) provider.

If you tried connecting IntelliJ IDEA / PyCharm / WebStorm / GoLand AI Assistant
directly to
`https://generativelanguage.googleapis.com/v1beta/openai` and got:

- **"Something went wrong" / "Try again"** when generating a commit message, or
- a crash / empty model list when picking a model, or
- requests that simply time out,

…this proxy fixes all of it. It sits between the IDE and Gemini's own
OpenAI-compatibility endpoint, cleans up the request, and streams the response
back instantly.

> **TL;DR:** point JetBrains AI Assistant at `http://localhost:5003`, paste your
> Gemini API key into `docker-compose.yml`, run `docker compose up -d --build`.

---

## Why this exists (the four JetBrains quirks)

JetBrains' "OpenAI-compatible" provider is not strictly OpenAI-compatible. Four
concrete issues make a direct Gemini connection fail:

**1. Model-prefix bug.** IntelliJ prepends `OpenAIAPI/models/` (or `OpenAIAPI/`)
to the model id, so it sends `"model": "OpenAIAPI/models/gemini-2.5-flash"`.
Google rejects that. The proxy strips these prefixes down to the bare id
(`gemini-2.5-flash`), case-insensitively and repeatedly.

**2. Strict `/v1/models` parser.** The IDE is very picky about the JSON shape of
the model list and chokes on Google's raw response. The proxy fully intercepts
`/models` and `/v1/models` and returns a clean, static, OpenAI-standard list.

**3. The commit-message timeout — the real one.** For commit messages the IDE
sends `stream: true`. Gemini 2.5 models keep **"thinking" ON by default**, which
adds **~10–15 s** before the first token. JetBrains' *own* HTTP client times out
in that window and shows "Something went wrong" — and a proxy cannot extend the
IDE's internal timeout. The fix is to **disable Gemini's thinking** by injecting
`reasoning_effort: "none"` (top-level field on Google's OpenAI-compat endpoint).
Time-to-first-byte drops from ~14 s to ~1–2 s and the timeout never fires.
On top of that, the proxy uses a fully async `httpx.AsyncClient` with **no read
timeout** and forwards SSE chunks the instant they arrive (no buffering).

**4. Unsupported parameters.** The IDE sends extras like `logit_bias` or empty
`tools` arrays that Gemini rejects. The proxy whitelist-filters the body and
forwards only: `model`, `messages`, `stream`, `temperature`, `top_p`,
`max_tokens`, `reasoning_effort`.

> Note: thinking can be fully disabled on **Gemini 2.5 Flash / Flash-Lite**
> only. **Gemini 2.5 Pro cannot turn thinking off**, so the proxy uses
> `reasoning_effort: "low"` for Pro to minimise (not eliminate) latency. For
> latency-critical work like commit messages, prefer Flash.

---

## Quick start

### 1. Get a Gemini API key

Create one at <https://aistudio.google.com/apikey>.

### 2. Configure

Open `docker-compose.yml` and paste your key:

```yaml
environment:
  GEMINI_API_KEY: "AIza...your-key..."
```

### 3. Run

```bash
docker compose up -d --build
```

Verify it is up (should report `"key_loaded": true`):

```bash
curl http://localhost:5003/health
```

### 4. Point JetBrains AI Assistant at it

In your JetBrains IDE: **Settings → Tools → AI Assistant → Models →
Third-party AI providers** → enable the **OpenAI-compatible** option and set:

| Field   | Value                                            |
| ------- | ------------------------------------------------ |
| URL     | `http://localhost:5003`                          |
| API key | any non-empty value (e.g. `x`) — it is ignored\* |
| Model   | `gemini-2.5-flash` or `gemini-2.5-pro`           |

\* The real key is read from the proxy's environment, not from the IDE. The
incoming `Authorization` header is intentionally ignored.

That's it. Code completion, chat, and commit-message generation all work.

---

## Running without Docker

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="AIza...your-key..."
python main.py        # serves on http://0.0.0.0:5003
```

---

## Configuration

All settings are environment variables (set them in `docker-compose.yml`):

| Variable            | Default                                                      | Description                                              |
| ------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| `GEMINI_API_KEY`    | *(required)*                                                 | Your Google AI Studio API key.                           |
| `PROXY_PORT`        | `5003`                                                        | Port the proxy listens on.                               |
| `GEMINI_BASE_URL`   | `https://generativelanguage.googleapis.com/v1beta/openai`    | Upstream OpenAI-compat base URL.                         |
| `DEFAULT_MODEL`     | `gemini-2.5-flash`                                            | Used when the IDE sends no / an empty model id.          |
| `FLASH_REASONING`   | `none`                                                        | `reasoning_effort` for 2.5 Flash / Flash-Lite.           |
| `PRO_REASONING`     | `low`                                                         | `reasoning_effort` for 2.5 Pro (cannot be `none`).       |
| `DEFAULT_REASONING` | `none`                                                        | `reasoning_effort` for any other model.                  |

Valid `reasoning_effort` values: `none`, `low`, `medium`, `high`. Use a higher
value to trade speed for answer quality. If the IDE ever sends its own
`reasoning_effort`, the proxy respects it instead of injecting one.

---

## Endpoints

| Method | Path                                          | Behaviour                                  |
| ------ | --------------------------------------------- | ------------------------------------------ |
| `GET`  | `/models`, `/v1/models`                       | Static OpenAI-standard model list.         |
| `POST` | `/chat/completions`, `/v1/chat/completions`   | Cleans + forwards; streams SSE unbuffered. |
| `GET`  | `/`, `/health`                                | Health check.                              |

Both `/x` and `/v1/x` variants exist so the proxy works no matter which path
JetBrains appends to the base URL.

---

## Verifying the latency fix

Watch the container logs while generating a commit message:

```bash
docker compose logs -f
```

You should see something like:

```
chat model=gemini-2.5-flash stream=True reasoning=none msgs=5
first byte in 1.34s
```

If `first byte` is back up around 10+ seconds, thinking is not being disabled —
check that you are using `gemini-2.5-flash` (not Pro) and that `FLASH_REASONING`
is `none`.

---

## How it works

```
JetBrains AI Assistant
        │  POST /v1/chat/completions
        │  { "model": "OpenAIAPI/models/gemini-2.5-flash", logit_bias, tools:[], ... }
        ▼
┌─────────────────────────────────────────────┐
│  this proxy (FastAPI + httpx, async)        │
│  • strip "OpenAIAPI/models/" prefix         │
│  • drop unsupported keys                    │
│  • inject reasoning_effort: "none"          │
│  • stream SSE chunk-by-chunk, no buffer     │
└─────────────────────────────────────────────┘
        │  POST /v1beta/openai/chat/completions
        ▼
Google Gemini (OpenAI-compatibility layer)
```

The HTTP client is a single shared `httpx.AsyncClient` (created on startup via
lifespan, reused for every request) with `read=None` so a slow first token can
never trip a *proxy-side* timeout. Streaming uses `client.stream(...)` +
`aiter_bytes()`, requesting `Accept-Encoding: identity` so there is nothing to
de-buffer, and sets `X-Accel-Buffering: no` / `Cache-Control: no-cache`.

---

## Tech stack

- **FastAPI** + **Uvicorn** (`uvicorn[standard]` → uvloop + httptools)
- **httpx** `AsyncClient` for streaming upstream calls
- **Docker** + **Docker Compose**

---

## Troubleshooting

**Still "Something went wrong" on commit messages.** Confirm the selected model
is `gemini-2.5-flash`. Pro cannot disable thinking, so its first token can still
arrive after the IDE's timeout. Check the logs for the `first byte in …s` line.

**`/health` shows `key_loaded: false`.** The key isn't reaching the container.
Re-check `GEMINI_API_KEY` in `docker-compose.yml` and rebuild
(`docker compose up -d --build`).

**Model picker empty or IDE error on connect.** Make sure the base URL is
exactly `http://localhost:5003` with no trailing path, and that the port matches
`PROXY_PORT` / the compose port mapping.

**429 / quota errors.** That's Google rate-limiting your key, not the proxy. The
upstream error is passed through unchanged.

---

## Related projects

This is not the only proxy in this space — others (e.g. `xrip/ollama-api-proxy`,
`Stream29/ProxyAsLocalModel`) emulate **Ollama / LM Studio** for JetBrains AI
Assistant. This project is deliberately different: it targets the
**OpenAI-compatible BYOK** path against **Gemini's own** compatibility layer, and
its main value is documenting and fixing the **commit-message timeout** caused by
Gemini 2.5 thinking.

---

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

Not affiliated with or endorsed by JetBrains or Google. "JetBrains", "IntelliJ
IDEA", and "Gemini" are trademarks of their respective owners. Use of the Gemini
API is subject to Google's terms.
