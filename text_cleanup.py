from openai import OpenAI

from config import OPENAI_API_KEY_JOURNAL


# Skapar en OpenAI-klient for latt efterbearbetning av transkriberad text.
client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)


# Den har funktionen stadar transkriptet utan att skriva om innehallet for mycket.
# Syftet ar att ta bort tydligt brus och hallucinationer samt gora texten mer lasbar.
def cleanup_transcript_text(text):
    cleaned_input = text.strip()
    if not cleaned_input:
        return ""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du stadar latt svensk veterinardiktering fran tal-till-text. "
                        "Ta bort uppenbart hallucinerat nonsens fran tystnad eller brus, "
                        "fixa enkel interpunktion och sma fel, men lagg inte till fakta. "
                        "Behall ordvalet sa nara originalet som mojligt. "
                        "Om texten bara ar brus eller nonsens, returnera en tom strang. "
                        "Returnera endast den stadade transkriberingen."
                    ),
                },
                {"role": "user", "content": cleaned_input},
            ],
        )
        cleaned_output = response.choices[0].message.content or ""
        return cleaned_output.strip()
    except Exception:
        # Om cleanup-steget misslyckas returneras originaltexten hellre an att hela flodet kraschar.
        return cleaned_input
