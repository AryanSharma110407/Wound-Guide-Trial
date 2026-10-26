import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

MODEL_PATH = Path(
    os.getenv(
        'MODEL_PATH',
        str(ROOT / 'models' / 'best_wound_model_finetuned.pth')
    )
)

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq').strip().lower()

# Groq
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '').strip()
GROQ_BASE_URL = os.getenv(
    'GROQ_BASE_URL',
    'https://api.groq.com/openai/v1'
).rstrip('/')
GROQ_MODEL = os.getenv(
    'GROQ_MODEL',
    'openai/gpt-oss-120b'
).strip()

# OmniRoute
OMNIROUTE_API_KEY = os.getenv('OMNIROUTE_API_KEY', '').strip()
OMNIROUTE_BASE_URL = os.getenv(
    'OMNIROUTE_BASE_URL',
    'http://localhost:20128/v1'
).rstrip('/')
OMNIROUTE_MODEL = os.getenv(
    'OMNIROUTE_MODEL',
    'auto'
).strip()

MAX_IMAGE_MB = int(os.getenv('MAX_IMAGE_MB', '10'))

# IMPORTANT:
# ImageFolder sorted these alphabetically in your trained model.
CLASS_NAMES = ['bruise', 'burn', 'cut']

IMAGE_SIZE = 224

if LLM_PROVIDER not in {'groq', 'omniroute'}:
    raise ValueError(
        "LLM_PROVIDER must be 'groq' or 'omniroute'"
    )