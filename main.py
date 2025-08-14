from fastapi import FastAPI, Request
import openai
import os
import json
from fastapi.responses import JSONResponse

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/evaluate")
async def evaluate(request: Request):
    try:
        # Read raw body (for debugging)
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8")
        print("Raw request body:\n", body_str)

        # Parse JSON directly from request
        json_data = await request.json()
        print("Parsed JSON:\n", json.dumps(json_data, indent=2))

        # Prompt creation
        system_prompt = "You are an expert trading assistant. You analyze signals and return a probability of success between 0 and 1."
        user_prompt = f"""Evaluate this trade signal:

{json.dumps(json_data, indent=2)}

Respond with only the numeric probability, between 0 and 1."""

        # GPT-4 call
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-4o"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=10,
        )

        reply = response['choices'][0]['message']['content'].strip()
        print("GPT reply:", reply)

        # Return float if possible
        probability = float(reply)
        return {"probability": probability}

    except Exception as e:
        # Full error logging
        print("❌ Exception caught in /evaluate:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
