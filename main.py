from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI, BadRequestError
import os, json, traceback, re
from typing import Optional

app = FastAPI()

CODE_VERSION = "v1.7.1"
print(f"🔁 Starting GPT signal evaluation server — code version: {CODE_VERSION}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set. The /evaluate route will fail until it is provided.")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-4o-mini"
STRICT_FAIL_ON_UNPARSABLE = os.getenv("STRICT_FAIL_ON_UNPARSABLE", "0").strip() == "1"
def extract_binary(text: str) -> Optional[int]:
    """Return 0 or 1 if present; else None."""
    if not text:
        return None
    t = text.strip()
    if t == "0":
        return 0
    if t == "1":
        return 1
    m = re.search(r"[01]", t)
    return int(m.group(0)) if m else None

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

def build_args(system_prompt: str, compact_json: str, *, max_tok_primary=1, max_tok_retry=1, retry=False):
    """
    Build model-specific args for a single-character output.
    """
    args = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": compact_json},
        ],
        "n": 1,
        "max_tokens": max_tok_retry if retry else max_tok_primary,
        "temperature": 0,
        "stop": ["\n"],
    }
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
        # Remove commonly unrecognized params for chat.completions
        for p in ["max_completion_tokens", "stop", "temperature"]:
            if p in args:
                args.pop(p, None)
        # Ensure we have max_tokens
        if "max_tokens" not in args:
            args["max_tokens"] = 3
        return client.chat.completions.create(**args)

@app.get("/health")
async def health():
    return {"status": "ok", "version": CODE_VERSION, "model": MODEL_NAME}




@app.post("/evaluate", response_class=PlainTextResponse)
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
            return PlainTextResponse("0", status_code=400)
    
        compact_json = json.dumps(payload, separators=(",", ":"))

        system_prompt_primary = (
            "You are a strict trading rules engine. Analyze the provided JSON of indicators.\n"
            "- Signal timeframe: check SMA cross, SMA trend, RSI, and ADX.\n"
            "- Higher timeframe confirmation: check SMA alignment, RSI, and ADX.\n"
            "- Structural filters: InsideBarCluster, ATRCompression, SwingProximityATR, HTF_RSIZone.\n"
            "Decision rules:\n"
            "1) If any structural filter indicates sideways/ranging, output 0.\n"
            "2) Otherwise weigh higher timeframe 70% and signal timeframe 30%.\n"
            "3) If both timeframes align (bullish or bearish) and no sideways condition, output 1; else 0.\n"
            "Output exactly one character: 0 or 1. No words, no spaces, no punctuation."
        )
        args = build_args(system_prompt_primary, compact_json, max_tok_primary=1, max_tok_retry=1, retry=False)
        resp = auto_heal_and_call(args)
        reply = (resp.choices[0].message.content or "").strip()

        val = extract_binary(reply)
        if val is None:
            args_retry = build_args(system_prompt_primary, compact_json, max_tok_primary=1, max_tok_retry=1, retry=True)
            resp2 = auto_heal_and_call(args_retry)
            reply2 = (resp2.choices[0].message.content or "").strip()
            val = extract_binary(reply2)

        if val is None:
            print("⚠️ Model did not return 0/1. Using fallback 0.")
            if STRICT_FAIL_ON_UNPARSABLE:
                return PlainTextResponse("0", status_code=502)
            val = 0

        decision = int(val)
        print(f"✅ Final decision: {decision}")
        return PlainTextResponse(str(decision))

    except BadRequestError as e:
        print("❌ OpenAI BadRequestError:", str(e))
        return PlainTextResponse("0", status_code=400)
    except Exception as e:
        print("❌ Unhandled ERROR:", str(e))
        traceback.print_exc()
        return PlainTextResponse("0", status_code=500)
@app.get("/", response_class=PlainTextResponse)
async def root():
    return f"OK: {CODE_VERSION}\n"

