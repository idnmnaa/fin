from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import openai
import json
import os

# === Initialize FastAPI app ===
app = FastAPI()

# === Enable CORS for all origins (useful for local testing) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Set OpenAI API key (preferably from environment variable) ===
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...")  # <-- Replace with your real key if needed

# === Request model for validation (optional, you can skip it if not needed) ===
class GPTRequest(BaseModel):
    symbol: str
    timeframe: str
    order_type: str
    candles: list
    pattern: dict
    sma50: dict
    rsi: dict
    atr5: dict

@app.post("/evaluate")
async def evaluate(request: Request):
    json_data = await request.json()

    # === Logging incoming request ===
    print("\n========== NEW REQUEST ==========")
    print(f"[{datetime.now()}] Incoming JSON:")
    print(json.dumps(json_data, indent=2))

    # === Compose GPT prompt ===
    prompt = f"""
Based on the following trading signal parameters, estimate the probability of trade success from 0 to 1. Respond with only the number (e.g. 0.82), no text.

{json.dumps(json_data)}
"""

    # === Call GPT-4 API ===
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional trading analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.2
        )
        reply = response.choices[0].message.content.strip()
        probability = float(reply)
    except Exception as e:
        print(f"[{datetime.now()}] ⚠️ Error from GPT:", str(e))
        probability = -1.0

    # === Return result to MQL5 ===
    print(f"[{datetime.now()}] Probability returned: {probability}")
    return str(probability)
