from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI
import os, json, traceback, re

app = FastAPI()

CODE_VERSION = "v1.5.1"
print(f"🔁 Starting GPT signal evaluation server — code version: {CODE_VERSION}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set. The /evaluate route will fail until it is provided.")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-nano")

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
            return JSONResponse(status_code=400, content={"error": "Invalid JSON", "details": str(pe)})

        print("Parsed JSON OK. Keys:", list(payload.keys()))

        # --- Token-lean prompts ---
        system_prompt = "Return ONLY one number in [0,1] (no text)."
        # compact JSON to save tokens
        compact_json = json.dumps(payload, separators=(',', ':'))
        user_prompt = compact_json  # no extra words

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=3,          # the correct param for chat.completions
            stop=["\n"],           # help cut off extra tokens
            n=1,
            seed=0                 # optional: reproducibility
        )

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

        prob = min(1.0, max(0.0, prob))
        print(f"✅ Final probability: {prob:.4f}")
        return {"probability": prob, "version": CODE_VERSION, "model": MODEL_NAME}

    except Exception as e:
        print("❌ Unhandled ERROR:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "version": CODE_VERSION})

@app.get("/", response_class=PlainTextResponse)
async def root():
    return f"OK: {CODE_VERSION}\n"
