from openai import OpenAI

from config import OPENAI_API_KEY_JOURNAL


# Skapar en OpenAI-klient för lätt efterbearbetning av transkriberad text.
client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)


# Den här funktionen städar transkriptet utan att skriva om innehallet för mycket.
# Syftet är att ta bort tydligt brus och hallucinationer samt göra texten mer läsbar.
def cleanup_transcript_text(text):
    cleaned_input = text.strip()
    if not cleaned_input:
        return ""

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du städar lätt en svensk veterinardiktering från tal-till-text. "
                        "Ta bort uppenbart hallucinerat nonsens från tystnad eller brus, "
                        "fixa enkel interpunktion och små fel, men lägg inte till fakta. "
                        "Behåll ordvalet så nära originalet som möjligt. "
                        "Om texten bara är brus eller nonsens, returnera en tom sträng. "
                        "Returnera endast den städade transkriberingen."
                    ),
                },
                {"role": "user", "content": cleaned_input},
            ],
        )
        cleaned_output = response.choices[0].message.content or ""
        return cleaned_output.strip()
    except Exception:
        # Om cleanup-steget misslyckas returneras originaltexten hellre än att hela flödet kraschar.
        return cleaned_input
