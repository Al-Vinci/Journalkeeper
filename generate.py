from openai import OpenAI
from config import OPENAI_API_KEY_JOURNAL

client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)

def generate_journal(text):
    prompt = f"""
Du är en veterinär assistent.

Skapa en journal i SOAP-format.

Regler:
- Skriv ENDAST det som explicit nämns
- Hitta inte på information
- Om något saknas, lämna tomt

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content