from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI, BadRequestError
import os, json, traceback, re
from typing import Optional

app = FastAPI()

CODE_VERSION = "v1.6.2"
print(f"🔁 Starting GPT signal evaluation server — code version: {CODE_VERSION}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set. The /evaluate route will fail until it is provided.")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()
STRICT_FAIL_ON_UNPARSABLE = os.getenv("STRICT_FAIL_ON_UNPARSABLE", "0").strip() == "1"

def is_nano_or_mini(model_name: str) -> bool:
    m = model_name.lower()
    return any(k in m for k in ["nano", "mini", "small"])

def parse_json_strict_but_safe(body_bytes: bytes) -> dict:
    tail = body_bytes[-16:] if len(body_bytes) >= 16 else body_bytes
    print(f"📦 Incoming bytes: len={len(body_bytes)} tail={repr(tail)}")
    cleaned = body_bytes.replace(b"\x00", b"")
    start = cleaned.find(b"{")
    end = cleaned.rfind(b"}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON object delimiters found in body")
    s = cleaned[start:end + 1].decode("utf-8", errors="ignore").strip()
    return json.loads(s)

def extract_probability(text: str) -> Optional[float]:
    """
    Accepts:
    - "0.73", "0.7\n", "Probability: 0.73", " 1 " etc.
    Returns float in [0,1] or None.
    """
    if not text:
        return None
    text = text.strip()
    # Fast path: bare number
    try:
        val = float(text)
        if 0.0 <= val <= 1.0:
            return val
    except Exception:
        pass
    # Fallback regex: first 0.x or 1 or 1.0
    m = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)", text)
    if m:
        try:
            val = float(m.group(0))
            if 0.0 <= val <= 1.0:
                return val
        except Exception:
            return None
    return None

def build_args(system_prompt: str, compact_json: str, *, max_tok_primary=3, max_tok_retry=6, retry=False):
    """
    Build model-specific args, with a slightly higher token cap on retry.
    """
    args = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": compact_json},
        ],
        "n": 1,
    }
    if is_nano_or_mini(MODEL_NAME):
        args["max_completion_tokens"] = max_tok_retry if retry else max_tok_primary
        # nano/mini: no temperature, no stop
    else:
        args["max_tokens"] = max_tok_retry if retry else max_tok_primary
        args["temperature"] = 0
        args["stop"] = ["\n"]
    return args

def auto_heal_and_call(args):
    """
    Calls chat.completions.create(**args), and if a 400 occurs due to unsupported params,
    strips them and retries once.
    """
    try:
        return client.chat.completions.create(**args)
    except BadRequestError as e:
        msg = str(e)
        print("⚠️ BadRequestError, attempting auto-fix:", msg)
        # Strip explicitly named unsupported params
        for p in re.findall(r"Unsupported parameter: '([^']+)'", msg):
            args.pop(p, None)
        # Token-cap swap
        if "max_tokens" in msg and "Unsupported" in msg:
            args.pop("max_tokens", None)
            args["max_completion_tokens"] = args.get("max_completion_tokens", 3)
        if "max_completion_tokens" in msg and "Unsupported" in msg:
            args.pop("max_completion_tokens", None)
            args["max_tokens"] = args.get("max_tokens", 3)
        # Temperature / stop sometimes rejected on smaller models
        if "temperature" in msg and "Unsupported" in msg:
            args.pop("temperature", None)
        if "stop" in msg and "Unsupported" in msg:
            args.pop("stop", None)
        return client.chat.completions.create(**args)

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

        try:
            payload = parse_json_strict_but_safe(body_bytes)
        except Exception as pe:
            print("❌ JSON parse error:", str(pe))
            traceback.print_exc()
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON", "details": str(pe), "version": CODE_VERSION},
            )
    
        #print("Parsed JSON OK. Keys:", list(payload.keys()))
        compact_json = json.dumps(payload, separators=(",", ":"))

        # Ultra-lean, explicit instruction
        system_prompt_primary = "Output a single number in [0,1]. No text."
        args = build_args(system_prompt_primary, compact_json, max_tok_primary=3, max_tok_retry=6, retry=False)
        resp = auto_heal_and_call(args)
        reply = (resp.choices[0].message.content or "").strip()
        #print("🧠 GPT raw reply (try1):", repr(reply))

        prob = extract_probability(reply)
        if prob is None:
            # Retry once with even stricter instruction and a touch more tokens
            system_prompt_retry = "ONLY digits for a number in [0,1]. Example: 0.73"
            args_retry = build_args(system_prompt_retry, compact_json, max_tok_primary=3, max_tok_retry=6, retry=True)
            resp2 = auto_heal_and_call(args_retry)
            reply2 = (resp2.choices[0].message.content or "").strip()
            #print("🧠 GPT raw reply (try2):", repr(reply2))
            prob = extract_probability(reply2)

        if prob is None:
            msg = "Model did not return a numeric probability"
            print(f"⚠️ {msg}. Using fallback.")
            if STRICT_FAIL_ON_UNPARSABLE:
                return JSONResponse(
                    status_code=502,
                    content={"error": msg, "version": CODE_VERSION, "model": MODEL_NAME},
                )
            # Fallback: 0.5 (deterministic neutral)
            prob = 0.5

        # Clamp
        prob = min(1.0, max(0.0, prob))
        print(f"✅ Final probability: {prob:.4f}")
        return {"probability": prob, "version": CODE_VERSION, "model": MODEL_NAME}

    except BadRequestError as e:
        print("❌ OpenAI BadRequestError:", str(e))
        return JSONResponse(status_code=400, content={"error": str(e), "version": CODE_VERSION})
    except Exception as e:
        print("❌ Unhandled ERROR:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "version": CODE_VERSION})

@app.get("/", response_class=PlainTextResponse)
async def root():
    return f"OK: {CODE_VERSION}\n"
