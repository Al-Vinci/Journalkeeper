from openai_client import create_openai_client


# Skapar en OpenAI-klient som används för att omvandla transkriberad text till ett journalutkast.
client = create_openai_client()


# Tar den transkriberade texten och ber modellen skapa en journal.
# Prompten är restriktiv för att minska risken att modellen hittar på information.
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

Baserat på behandlingen i den transkriberade texten, skriv ett hemgångsråd efter journalen. 

Text:
{text}
"""
            }
        ]
    )

    return response.output[0].content[0].text
