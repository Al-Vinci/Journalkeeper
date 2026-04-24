import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY_JOURNAL = os.getenv("OPENAI_API_KEY_JOUR")

if not OPENAI_API_KEY_JOURNAL:
    raise ValueError("API-nyckel saknas")