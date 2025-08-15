from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI
import os, json, traceback

app = FastAPI()

# === Version banner ===
CODE_VERSION = "v1.4.0"
print(f"🔁 Starting GPT signal evaluation server — code version: {CODE_VERSION}")

# === OpenAI client ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set. The /evaluate route will fail until it is provided.")
client = OpenAI(api_key=OPENAI_API_KEY)

# Allow overriding model via env, default to a strong general model.
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

def parse_json_strict_but_safe(body_bytes: bytes) -> dict:
    """
    Many MQL5 WebRequest bodies include trailing nulls or junk.
    This parser:
      1) logs diagnostics
      2) removes nulls
      3) trims to the first '{' ... last '}' window
      4) json.loads the cleaned slice
    """
    # Diagnostics
    tail = body_bytes[-16:] if len(body_bytes) >= 16 else body_bytes
    print(f"📦 Incoming bytes: len={len(body_bytes)} tail={repr(tail)}")

    # 1) Drop nulls early
    cleaned = body_bytes.replace(b"\x00", b"")

    # 2) Find JSON object window
    start = cleaned.find(b"{")
    end = cleaned.rfind(b"}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON object delimiters found in body")

    slice_bytes = cleaned[start:end + 1]

    # 3) Decode and strip whitespace
    s = slice_bytes.decode("utf-8", errors="ignore").strip()

    # 4) Load JSON
    return json.loads(s)

@app.get("/health")
async def health():
    return {"status": "ok", "version": CODE_VERSION, "model": MODEL_NAME}

@app.post("/evaluate")
async def evaluate(request: Request):
    print(f"\n📥 New request — version: {CODE_VERSION}")
    try:
        body_bytes = await request.body()
        # Log the raw body once (shortened if huge)
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
                content={"error": "Invalid JSON", "details": str(pe)}
            )

        print("Parsed JSON OK. Keys:", list(payload.keys()))

        # Build prompt
        system_prompt = (
            "You are an expert trading assistant. "
            "Analyze the provided trade signal and output ONLY one number between 0 and 1: "
            "the probability of success for the pending order."
        )
        user_prompt = (
            "Evaluate this trade signal and respond with a single numeric probability "
            "between 0 and 1 (no text, just the number):\n\n" +
            json.dumps(payload, indent=2)
        )

        # Call OpenAI (v1.x API)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=10,
        )

        reply = (resp.choices[0].message.content or "").strip()
        print("🧠 GPT raw reply:", repr(reply))

        # Try to parse float from the reply (and clamp to [0,1])
        try:
            prob = float(reply)
        except ValueError:
            # If the model returned something like "0.78\n", or "Probability: 0.78"
            import re
            m = re.search(r"([0-1](?:\.\d+)?)", reply)
            if not m:
                raise ValueError(f"Model did not return a numeric probability: {reply}")
            prob = float(m.group(1))

        # Clamp
        if prob < 0.0: prob = 0.0
        if prob > 1.0: prob = 1.0

        print(f"✅ Final probability: {prob:.4f}")
        return {"probability": prob, "version": CODE_VERSION, "model": MODEL_NAME}

    except Exception as e:
        print("❌ Unhandled ERROR:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "version": CODE_VERSION})

# Optional: simple root to avoid 404 noise on /
@app.get("/", response_class=PlainTextResponse)
async def root():
    return f"OK: {CODE_VERSION}\n"
