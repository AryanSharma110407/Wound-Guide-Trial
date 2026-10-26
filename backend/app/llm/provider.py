from app.config import LLM_PROVIDER
from app.llm.groq import GroqProvider
from app.llm.omniroute import OmniRouteProvider


def get_provider():
    return (
        GroqProvider()
        if LLM_PROVIDER == 'groq'
        else OmniRouteProvider()
    )