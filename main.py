
import os
import json
import re
from datetime import date
from decimal import Decimal
from typing import Annotated

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class InvoiceRequest(BaseModel):
    text: Annotated[str, Field(min_length=1)]


class InvoiceResponse(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str

    @field_validator("vendor")
    @classmethod
    def vendor_not_empty(cls, value):
        if not value.strip():
            raise ValueError("vendor must not be empty")
        return value.strip()

    @field_validator("amount")
    @classmethod
    def finite_amount(cls, value):
        if not (value == value and abs(value) != float("inf")):
            raise ValueError("amount must be finite")
        return value

    @field_validator("currency")
    @classmethod
    def valid_currency(cls, value):
        value = value.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError("currency must be a 3-letter code")
        return value

    @field_validator("date")
    @classmethod
    def valid_date(cls, value):
        value = value.strip()
        date.fromisoformat(value)
        return value


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/extract", response_model=InvoiceResponse)
def extract_invoice(body: InvoiceRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Invoice text is required")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="LLM API key is not configured")

    prompt = f"""
Extract these four fields from the invoice text:
- vendor: exact vendor/company name
- amount: total amount due as a number
- currency: three-letter uppercase currency code
- date: payment due date in YYYY-MM-DD format

Return ONLY one JSON object with exactly these keys:
vendor, amount, currency, date.
Do not guess. Use the invoice's total due, currency, and due date,
not the invoice date or subtotal.

Invoice text:
{text}
"""

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You extract invoice data accurately. Return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON object in model response")

        data = json.loads(match.group(0))
        return InvoiceResponse.model_validate(data)

    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Invoice extraction service failed")
    except (ValueError, KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail="Could not extract valid invoice fields")
