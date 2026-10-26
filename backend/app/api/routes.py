import json
import re
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.config import MAX_IMAGE_MB, LLM_PROVIDER
from app.ml.predictor import WoundPredictor
from app.rag.retriever import KnowledgeRetriever
from app.llm.provider import get_provider

router = APIRouter(prefix='/api', tags=['analysis'])
predictor = WoundPredictor()
retriever = KnowledgeRetriever()

LOW_CONFIDENCE = 60.0  # below this %, the UI warns the user

SYSTEM_RULES = '''You are the guidance layer of a first-aid education prototype. The image classifier is not a diagnosis. Use ONLY the supplied knowledge-base facts; do not invent treatments, dosages, materials, or procedures. Keep language simple. If information is insufficient, say so. For severe or uncertain situations, recommend professional/emergency care.'''

JSON_FORMAT = '''Reply with ONLY one valid JSON object (no markdown, no text outside it) with exactly these keys:
{
  "summary": "2 short sentences: what this category means and the first thing to do",
  "do": ["short action step", "..."],
  "dont": ["short thing to avoid", "..."],
  "items": [{"item": "name of a user's item", "how_to_use": "how the knowledge base says to use it"}],
  "good_signs": ["signs that healing is going normally"],
  "warning_signs": ["signs that need a doctor / emergency care"],
  "see_doctor_now": false
}
"items" must cover every matched material (safe use, or what to avoid if it is a harmful item). For items listed under unmatched_items, add an entry saying there is no verified guidance for it. Set "see_doctor_now" to true only if urgent care is likely.'''


def parse_guidance(text: str):
    """Read the LLM reply as JSON (tolerates ```json fences and trailing commas). None if unusable."""
    text = re.sub(r'```(?:json)?', '', text or '')
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        return None
    body = re.sub(r',\s*([}\]])', r'\1', match.group(0))
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    clean = {'summary': str(data.get('summary', '')), 'see_doctor_now': bool(data.get('see_doctor_now', False))}
    for key in ['do', 'dont', 'good_signs', 'warning_signs']:
        value = data.get(key, [])
        clean[key] = [str(v) for v in value] if isinstance(value, list) else []
    items = data.get('items', [])
    clean['items'] = [i for i in items if isinstance(i, dict) and 'item' in i] if isinstance(items, list) else []
    return clean if (clean['do'] or clean['dont']) else None


def kb_fallback(context: dict) -> dict:
    """Build the same structured result straight from the knowledge base (no LLM needed)."""
    g = context['injury_guidance']
    items = []
    for m in context['matched_materials']:
        text = ' '.join(m.get('safe_uses', []) + m.get('avoid', []))
        items.append({'item': m['item'], 'how_to_use': text})
    for name in context.get('unmatched_items', []):
        items.append({'item': name, 'how_to_use': 'No verified guidance for this item in the knowledge base.'})
    first = g.get('immediate_steps', [''])[0]
    return {
        'summary': f"{g.get('what_it_is', '')} First: {first}".strip(),
        'do': g.get('immediate_steps', []), 'dont': g.get('avoid', []), 'items': items,
        'good_signs': g.get('good_signs', []), 'warning_signs': g.get('warning_signs', []),
        'see_doctor_now': False,
    }


@router.get('/health')
def health():
    return {'status': 'ok', 'provider': LLM_PROVIDER, 'model_loaded': True}


@router.post('/analyze')
async def analyze(image: UploadFile = File(...), items: str = Form('')):
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(400, 'Please upload a valid image.')
    raw = await image.read()
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f'Image must be smaller than {MAX_IMAGE_MB} MB.')
    try:
        pil = Image.open(BytesIO(raw)).convert('RGB')
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, 'The uploaded file is not a readable image.')

    prediction = await run_in_threadpool(predictor.predict, pil)
    item_list = [x.strip() for x in items.split(',') if x.strip()]
    context = retriever.retrieve(prediction['injury'], item_list)

    prompt = (
        f"{SYSTEM_RULES}\n\nDetected injury category: {prediction['injury']}\n"
        f"Classifier confidence: {prediction['confidence']}%\n\n"
        f"Knowledge base:\n{json.dumps(context, indent=2)}\n\n"
        f"User's available items: {item_list or 'Not provided'}\n\n{JSON_FORMAT}"
    )

    guidance, structured = '', None
    try:
        guidance = await run_in_threadpool(get_provider().generate, prompt)
        structured = parse_guidance(guidance)
        if structured is None:  # one retry if the reply was not valid JSON
            guidance = await run_in_threadpool(get_provider().generate, prompt)
            structured = parse_guidance(guidance)
    except Exception as exc:
        print(f'LLM failed, using knowledge base fallback: {exc}')

    source = 'llm'
    if structured is None:  # LLM failed or returned unreadable text -> never show raw JSON
        structured, source = kb_fallback(context), 'knowledge_base'

    return {
        'prediction': prediction,
        'low_confidence': prediction['confidence'] < LOW_CONFIDENCE,
        'guidance': guidance,
        'structured': structured,
        'guidance_source': source,
        'matched_materials': context['matched_materials'],
        'sources': context['sources'],
        'provider': LLM_PROVIDER,
    }
