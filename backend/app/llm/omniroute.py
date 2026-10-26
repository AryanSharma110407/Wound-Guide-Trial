import requests
from app.config import OMNIROUTE_API_KEY, OMNIROUTE_BASE_URL, OMNIROUTE_MODEL
from app.llm.base import LLMProvider

class OmniRouteProvider(LLMProvider):
    def __init__(self):
        if not OMNIROUTE_API_KEY:
            raise RuntimeError('OMNIROUTE_API_KEY is required when LLM_PROVIDER=omniroute')

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f'{OMNIROUTE_BASE_URL}/chat/completions',
            headers={'Authorization': f'Bearer {OMNIROUTE_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': OMNIROUTE_MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.2},
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
