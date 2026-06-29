"""
JetBrains AI Assistant  ->  Google Gemini (OpenAI-compatible) proxy.

Fixes four JetBrains BYOK quirks:

  1. Model prefix bug   : strips "OpenAIAPI/models/" / "OpenAIAPI/" / "models/".
  2. Strict model parser: serves a clean, static OpenAI-standard /models list.
  3. Streaming timeout  : true zero-buffer async SSE pass-through (no read
                          timeout, chunks forwarded the instant they arrive).
  4. Unsupported params : forwards only model/messages/stream/temperature/
                          top_p/max_tokens; drops logit_bias, empty tools, etc.

Framework : FastAPI + Uvicorn (uvloop/httptools via uvicorn[standard]).
HTTP      : a single shared httpx.AsyncClient + client.stream(...).
"""

import os
import json
import time
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


# Configure the root logger from LOG_LEVEL (default INFO) so the app's own
# log.info / log.debug calls are actually emitted. Running under uvicorn does
# NOT configure this named logger, so we must do it here at import time.
#   LOG_LEVEL=DEBUG  -> raw bodies, sanitised payloads, per-chunk stream dumps.
#   LOG_LEVEL=INFO   -> one line per request + timing (default).
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("jetbrains-ai-gemini-local-proxy")

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
).rstrip("/")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "5003"))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gemini-2.5-flash")

# Only these keys survive sanitisation and reach Google.
ALLOWED_KEYS = {
    "model", "messages", "stream", "temperature", "top_p", "max_tokens",
    "reasoning_effort",
}

# Per-model thinking control. Gemini 2.5 Flash keeps "thinking" ON by default,
# which adds ~10-15s to time-to-first-byte and trips JetBrains' OWN internal
# timeout (the proxy's read=None cannot override the IDE's client timeout).
# reasoning_effort is a top-level field on Google's OpenAI-compat endpoint.
#   "none" -> thinking fully OFF (only works on 2.5 Flash / Flash-Lite).
#   "low"  -> minimal thinking (2.5 Pro CANNOT be turned off, so use "low").
REASONING_BY_MODEL = {
    "gemini-2.5-flash": os.environ.get("FLASH_REASONING", "none"),
    "gemini-2.5-flash-lite": os.environ.get("FLASH_REASONING", "none"),
    "gemini-2.5-pro": os.environ.get("PRO_REASONING", "low"),
}
DEFAULT_REASONING = os.environ.get("DEFAULT_REASONING", "none")

# Static list handed back to IntelliJ's strict parser.
STATIC_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

# Prefixes IntelliJ may glue onto the model id (checked case-insensitively).
_PREFIXES = ("openaiapi/models/", "openaiapi/", "openai/", "models/")


def clean_model_id(model: str | None) -> str:
    """Aggressively strip JetBrains prefixes -> bare 'gemini-2.5-flash'."""
    if not model:
        return DEFAULT_MODEL
    m = model.strip()
    changed = True
    while changed:
        changed = False
        low = m.lower()
        for p in _PREFIXES:
            if low.startswith(p):
                m = m[len(p):]
                changed = True
                break
    if "/" in m:                       # anything still path-like -> last segment
        m = m.split("/")[-1]
    return m or DEFAULT_MODEL


def sanitize_payload(raw: dict) -> dict:
    """Whitelist-filter the request body, normalise the model id, and inject
    thinking control so Gemini 2.5 Flash answers fast enough for the IDE."""
    out = {k: v for k, v in raw.items() if k in ALLOWED_KEYS}
    model = clean_model_id(out.get("model"))
    out["model"] = model
    if not isinstance(out.get("messages"), list):
        out["messages"] = raw.get("messages", [])
    out["stream"] = bool(raw.get("stream", False))
    # Only inject if the client did not explicitly request a level.
    if not out.get("reasoning_effort"):
        out["reasoning_effort"] = REASONING_BY_MODEL.get(model, DEFAULT_REASONING)
    return out


# --------------------------------------------------------------------------- #
# HTTP client lifecycle                                                        #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    # read=None  -> NO read timeout: a slow first token can never trip a
    # "Something went wrong" timeout in the IDE.
    timeout = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=15.0)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    app.state.client = httpx.AsyncClient(timeout=timeout, limits=limits)
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY is empty - set it in docker-compose.yml")
    log.info(
        "Proxy ready -> %s (port=%s log_level=%s key_loaded=%s)",
        GEMINI_BASE_URL, PROXY_PORT, LOG_LEVEL, bool(GEMINI_API_KEY),
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="JetBrains -> Gemini Proxy", lifespan=lifespan)


def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
        # Disable compression so there is literally nothing to debuffer.
        "Accept-Encoding": "identity",
    }


# --------------------------------------------------------------------------- #
# /models  (and /v1/models)  -- fully intercepted, static, OpenAI-standard     #
# --------------------------------------------------------------------------- #

def _models_response() -> dict:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "google"}
            for mid in STATIC_MODELS
        ],
    }


@app.get("/models")
@app.get("/v1/models")
async def list_models():
    return JSONResponse(_models_response())


# --------------------------------------------------------------------------- #
# /chat/completions  (and /v1/chat/completions)                                #
# --------------------------------------------------------------------------- #

# Content-Type is set via media_type on StreamingResponse, so it is omitted
# here to avoid a duplicate header.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _safe_json(r: httpx.Response):
    try:
        return r.json()
    except Exception:
        return {"error": {"message": r.text}}


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.body()
    # Identify the caller (Authorization redacted) so empty/probe requests can
    # be told apart from real prompts.
    hdrs = {
        k: v for k, v in request.headers.items()
        if k.lower() != "authorization"
    }
    log.debug("incoming %s %s headers=%s", request.method, request.url.path, hdrs)
    log.debug("raw body: %s", body)
    try:
        if not body:
            # content-length > 0 with an empty body means the server dropped it
            # (historically: uvicorn's httptools parser swallowing the body on a
            # ktor 'Upgrade: h2c' request -- fixed by running uvicorn --http h11).
            log.error(
                "empty request body (ua=%r content-length=%r upgrade=%r) -- if "
                "content-length>0 the HTTP layer dropped the body; check the "
                "uvicorn --http parser / Upgrade handling",
                request.headers.get("user-agent"),
                request.headers.get("content-length"),
                request.headers.get("upgrade"),
            )
            return JSONResponse(
                {"error": {"message": "empty request body"}}, status_code=400
            )
        raw = json.loads(body)
    except Exception:
        log.error("invalid JSON body: %s", body.decode("utf-8", "replace"))
        return JSONResponse(
            {"error": {"message": "invalid JSON body"}}, status_code=400
        )

    payload = sanitize_payload(raw if isinstance(raw, dict) else {})
    log.debug("payload: %s", payload)
    url = f"{GEMINI_BASE_URL}/chat/completions"
    client: httpx.AsyncClient = app.state.client

    log.info(
        "chat model=%s stream=%s reasoning=%s msgs=%d",
        payload["model"],
        payload["stream"],
        payload.get("reasoning_effort"),
        len(payload.get("messages", [])),
    )

    # ---- Non-streaming -------------------------------------------------- #
    if not payload["stream"]:
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=payload, headers=_auth_headers())
        except httpx.HTTPError as e:
            log.error("upstream error: %s", e)
            return JSONResponse(
                {"error": {"message": f"upstream error: {e}"}}, status_code=502
            )
        response_json = _safe_json(r)
        log.info(
            "non-stream done status=%s in %.2fs", r.status_code,
            time.perf_counter() - t0,
        )
        if r.status_code >= 400:
            log.error("upstream %s body: %s", r.status_code, r.text)
        log.debug("non-stream response: %s", response_json)
        return JSONResponse(content=response_json, status_code=r.status_code)

    # ---- Streaming: zero-buffer SSE pass-through ------------------------ #
    async def event_stream():
        t0 = time.perf_counter()
        first = True
        total_bytes = 0
        chunks = 0
        try:
            async with client.stream(
                "POST", url, json=payload, headers=_auth_headers()
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    text = body.decode("utf-8", "replace")
                    log.error("stream upstream %s body: %s", resp.status_code, text)
                    err = {
                        "error": {
                            "message": text,
                            "code": resp.status_code,
                        }
                    }
                    yield f"data: {json.dumps(err)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
                # aiter_bytes() yields each network chunk the moment it
                # arrives (content-decoded, no re-chunking) -> instant relay.
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        if first:
                            log.info(
                                "first byte in %.2fs", time.perf_counter() - t0
                            )
                            first = False
                        total_bytes += len(chunk)
                        chunks += 1
                        log.debug("stream chunk: %s", chunk)
                        yield chunk
            log.info(
                "stream done: %d chunks, %d bytes in %.2fs",
                chunks, total_bytes, time.perf_counter() - t0,
            )
        except httpx.HTTPError as e:
            log.error("stream error after %d chunks: %s", chunks, e)
            err = {"error": {"message": str(e)}}
            yield f"data: {json.dumps(err)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# --------------------------------------------------------------------------- #
# Health                                                                       #
# --------------------------------------------------------------------------- #

@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "target": GEMINI_BASE_URL,
        "key_loaded": bool(GEMINI_API_KEY),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="0.0.0.0", port=PROXY_PORT, log_level=LOG_LEVEL.lower()
    )

