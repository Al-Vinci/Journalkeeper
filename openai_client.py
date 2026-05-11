import httpx
from openai import OpenAI

from config import OPENAI_API_KEY_JOURNAL


def create_openai_client():
    return OpenAI(
        api_key=OPENAI_API_KEY_JOURNAL,
        http_client=httpx.Client(trust_env=False),
    )
