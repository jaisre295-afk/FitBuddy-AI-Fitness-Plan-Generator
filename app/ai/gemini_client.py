from functools import lru_cache

from google import genai

from ..config import settings


@lru_cache
def get_gemini_client():
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Please add your Gemini API key to the .env file."
        )

    return genai.Client(api_key=settings.GEMINI_API_KEY)


def generate_content(prompt: str) -> str:
    client = get_gemini_client()

    model = "gemini-3.5-flash-lite"

    print(f"Trying Gemini model: {model}")

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        text = getattr(response, "text", None)

        if text and text.strip():
            print(f"Gemini success: {model}")
            return text.strip()

        raise RuntimeError(
            f"Gemini returned an empty response from {model}."
        )

    except Exception as error:
        print(f"Gemini error from {model}: {error}")

        raise RuntimeError(
            f"Gemini API error: {error}"
        ) from error