from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI, BadRequestError
import os, json, traceback, re

app = FastAPI()

# === Version banner ===
CODE_VERSION = "v1.6.0"
print(f"🔁 Starting GPT signal evaluation server — code version: {CODE_VERSION}")

# === OpenAI client ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set. The /evaluate route will fail until it is provided.")
client = OpenAI(api_key=OPENAI_API_KEY)

# Allow overriding model via env; default to a cheap model.
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-nano").strip()


def is_nano_or_mini(model_name: str) -> bool:
    """
    Heuristic: nano/mini/small tiers use max_completion_tokens and often disallow temperature control.
    Adjust this if your org uses different naming.
    """
    m = model_name.lower()
    return any(k in m for k in ["nano", "mini", "small"])


def parse_json_strict_but_safe(body_bytes: bytes) -> dict:
    """
    MQL5 WebRequest bodies can include trailing nulls or junk.
    Steps:
      1) drop nulls
      2) find first '{' and last '}'
      3) decode and json.loads
    """
    tail = body_bytes[-16:] if len(body_bytes) >= 16 else body_bytes
    print(f"📦 Incoming bytes: len={len(body_bytes)} tail={repr(tail)}")

    cleaned = body_bytes.replace(b"\x00", b"")
    start = cleaned.find(b"{")
    end = cleaned.rfind(b"}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON object delimiters found in body")

    s = cleaned[start:end + 1].decode("utf-8", errors="ignore").strip()
    return json.loads(s)


@app.get("/health")
async def health():
    return {"status": "ok", "version": CODE_VERSION, "model": MODEL_NAME}


@app.post("/evaluate")
async def evaluate(request: Request):
    print(f"\n📥 New request — version: {CODE_VERSION}")
    try:
        body_bytes = await request.body()
        preview = body_bytes[:400]
        print("Raw request (first 400 bytes):", preview.decode("utf-8", errors="ignore"))

        # Robust parsing
        try:
            payload = parse_json_strict_but_safe(body_bytes)
        except Exception as pe:
            print("❌ JSON parse error:", str(pe))
            traceback.print_exc()
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON", "details": str(pe), "version": CODE_VERSION},
            )

        print("Parsed JSON OK. Keys:", list(payload.keys()))

        # --- Token-lean prompts ---
        # Keep system prompt ultra-short; user content is just compact JSON
        system_prompt = "Return ONLY one number in [0,1] (no text)."
        compact_json = json.dumps(payload, separators=(",", ":"))

        # Build model-specific args
        args = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": compact_json},
            ],
            "stop": ["\n"],
            "n": 1,
        }

        if is_nano_or_mini(MODEL_NAME):
            # nano/mini: no temperature; use max_completion_tokens
            args["max_completion_tokens"] = int(os.getenv("MAX_COMPLETION_TOKENS", "3"))
        else:
            # larger chat models
            args["max_tokens"] = int(os.getenv("MAX_TOKENS", "3"))
            args["temperature"] = float(os.getenv("TEMPERATURE", "0"))

        # Call OpenAI
        try:
            resp = client.chat.completions.create(**args)
        except BadRequestError as e:
            # Graceful fallback if org/model changes parameter support
            msg = str(e)
            print("⚠️ BadRequestError, attempting auto-fix:", msg)

            # Strip temperature if it's rejected
            if "temperature" in msg and "Unsupported" in msg:
                args.pop("temperature", None)

            # Swap token cap if needed
            if "max_tokens" in msg and "Unsupported parameter" in msg:
                args.pop("max_tokens", None)
                args["max_completion_tokens"] = int(os.getenv("MAX_COMPLETION_TOKENS", "3"))
            if "max_completion_tokens" in msg and "Unsupported parameter" in msg:
                args.pop("max_completion_tokens", None)
                args["max_tokens"] = int(os.getenv("MAX_TOKENS", "3"))

            # Retry once
            resp = client.chat.completions.create(**args)

        reply = (resp.choices[0].message.content or "").strip()
        print("🧠 GPT raw reply:", repr(reply))

        # Parse a single float in [0,1]
        try:
            prob = float(reply)
        except ValueError:
            m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", reply)
            if not m:
                raise ValueError(f"Model did not return a numeric probability: {reply}")
            prob = float(m.group(1))

        # Clamp
        prob = min(1.0, max(0.0, prob))
        print(f"✅ Final probability: {prob:.4f}")
        return {"probability": prob, "version": CODE_VERSION, "model": MODEL_NAME}

    except BadRequestError as e:
        # Surface OpenAI 4xx nicely
        print("❌ OpenAI BadRequestError:", str(e))
        return JSONResponse(status_code=400, content={"error": str(e), "version": CODE_VERSION})
    except Exception as e:
        print("❌ Unhandled ERROR:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "version": CODE_VERSION})


# Optional: simple root to avoid 404 noise on /
@app.get("/", response_class=PlainTextResponse)
async def root():
    return f"OK: {CODE_VERSION}\n"
