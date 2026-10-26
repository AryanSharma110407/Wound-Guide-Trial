import requests

from app.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from app.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    """Groq Cloud provider using Groq's OpenAI-compatible API."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError(
                'GROQ_API_KEY is required when LLM_PROVIDER=groq'
            )

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f'{GROQ_BASE_URL}/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': GROQ_MODEL,
                'messages': [
                    {
                        'role': 'system',
                        'content': (
                            'You are the language-generation layer of a '
                            'first-aid education prototype. Follow the '
                            'supplied knowledge base exactly and do not '
                            'invent medical procedures, treatments, '
                            'dosages, or materials.'
                        ),
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    },
                ],
                'temperature': 0.2,
                'max_tokens': 900,
            },
            timeout=90,
        )

        response.raise_for_status()
        data = response.json()

        try:
            content = data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f'Unexpected Groq API response: {data}'
            ) from exc

        if not content:
            return 'No guidance was generated.'

        return content.strip()