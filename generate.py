from openai import OpenAI
from config import OPENAI_API_KEY_JOURNAL


# Skapar en OpenAI-klient som anvands for att omvandla transkriberad text till ett journalutkast.
client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)


# Tar den transkriberade texten och ber modellen skapa en journal i SOAP-format.
# Prompten ar restriktiv for att minska risken att modellen hittar pa information.
def generate_journal(text):
    prompt = f"""
Du ar en veterinär assistent.

Skapa en journal i SOAP-format.

Regler:
- Skriv ENDAST det som explicit namns
- Hitta inte pa information
- Om nagot saknas, lamna tomt

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
