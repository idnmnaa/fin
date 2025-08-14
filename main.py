from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai
import os
import json
import traceback

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Укажи явную версию кода (меняй при каждом обновлении)
CODE_VERSION = "v1.2.0"

print(f"🔁 Starting GPT signal evaluation server — code version: {CODE_VERSION}")

@app.post("/evaluate")
async def evaluate(request: Request):
    try:
        print(f"\n📥 New request received — version: {CODE_VERSION}")

        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8")
        print("Raw request body:\n", body_str)

        json_data = json.loads(body_str)
        print("Parsed JSON:\n", json.dumps(json_data, indent=2))

        system_prompt = "You are an expert trading assistant. You analyze signals and return a probability of success between 0 and 1."
        user_prompt = f"""Evaluate this trade signal and return a number from 0 to 1:

{json.dumps(json_data, indent=2)}

Only respond with the numeric probability."""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=10,
        )

        reply = response['choices'][0]['message']['content'].strip()
        print("GPT reply:", reply)

        probability = float(reply)
        return {"probability": probability}

    except Exception as e:
        print("❌ ERROR:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
