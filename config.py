import os
from dotenv import load_dotenv


# Laddar variabler från .env-filen så att API-nyckeln kan läsas in utan att hårdkodas i koden.
load_dotenv(override=True)

# Hämtar projektets OpenAI-nyckel från miljövariabler.
OPENAI_API_KEY_JOURNAL = os.getenv("OPENAI_API_KEY_JOUR")

# Stoppar programmet tidigt om nyckeln saknas, så att felet blir tydligt direkt.
if not OPENAI_API_KEY_JOURNAL:
    raise ValueError("API-nyckel saknas")
