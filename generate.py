from openai import OpenAI
from config import OPENAI_API_KEY_JOURNAL


# Skapar en OpenAI-klient som anvands for att omvandla transkriberad text till ett journalutkast.
client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)


# Tar den transkriberade texten och ber modellen skapa en journal.
# Prompten ar restriktiv for att minska risken att modellen hittar på information.
def generate_journal(text):
    response = client.responses.create(
        model="gpt-5.4",
        temperature=0,
        input=[
            {
                "role": "system",
                "content": (
                    "Du är en svensk veterinär assistent. "
                    "Skriv kortfattat, strukturerat och kliniskt. "
                    "Hitta inte på information. "
                    "Om något saknas, lämna rubriken tom utan extra text."
                )
            },
            {
                "role": "user",
                "content": f"""
Skapa en veterinärmedicinsk journal med exakt följande struktur:

Anamnes/Historik:
Klinisk undersökning:
Problemlista:
Differentialdiagnoser:
Behandling:
Behandlingsplan:

Regler:
- Använd exakt rubrikerna ovan
- Skriv endast det som explicit nämns
- Ingen extra text före eller efter

Text:
{text}
"""
            }
        ]
    )

    return response.output[0].content[0].text
