# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

class Signal(BaseModel):
    json: str

@app.post("/evaluate")
async def evaluate(signal: Signal):
    prompt = f"""
    Ниже приведён JSON с параметрами сделки в трейдинге:
    {signal.json}

    Определи вероятность успешной сделки (в процентах от 0 до 100), учитывая технические и рыночные параметры.
    Ответ должен быть только числом.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.2,
    )

    probability_str = response.choices[0].message.content.strip()
    try:
        return float(probability_str)
    except:
        return 0.0
