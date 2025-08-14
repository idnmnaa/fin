from fastapi import FastAPI, Request
import openai
import os
import json

app = FastAPI()

# Set your API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set in Render dashboard or .env

# Or set directly (not recommended for prod)
# openai.api_key = "sk-..."

@app.post("/evaluate")
async def evaluate(request: Request):
    try:
        # Read raw body and decode
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8")

        print("Raw request body:\n", body_str)

        # Load as JSON
        json_data = json.loads(body_str)

        # Optional: Log parsed data
        print("Parsed JSON:\n", json.dumps(json_data, indent=2))

        # Compose prompt
        system_prompt = "You are an expert trading assistant. You analyze signals and return a probability of success between 0 and 1."
        user_prompt = f"""Evaluate this trade signal:

{json.dumps(json_data, indent=2)}

Respond with only the numeric probability, between 0 and 1."""

        # GPT call
        response = openai.ChatCompletion.create(
            model="gpt-5",  # or "gpt-4o", or "gpt-5" if available
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=10,
        )

        reply = response['choices'][0]['message']['content'].strip()
        print("GPT reply:", reply)

        # Parse reply as float
        probability = float(reply)
        return probability

    except Exception as e:
        print("❌ Error:", str(e))
        return {"error": str(e)}
