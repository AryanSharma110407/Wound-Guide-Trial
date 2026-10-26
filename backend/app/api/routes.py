from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from app.config import MAX_IMAGE_MB, LLM_PROVIDER
from app.ml.predictor import WoundPredictor
from app.rag.retriever import KnowledgeRetriever
from app.llm.provider import get_provider

router = APIRouter(prefix='/api', tags=['analysis'])
predictor = WoundPredictor()
retriever = KnowledgeRetriever()

SYSTEM_RULES = '''You are the guidance layer of a first-aid education prototype. The image classifier is not a diagnosis. Use ONLY the supplied knowledge-base facts; do not invent treatments, dosages, materials, or procedures. Give simple, practical steps for temporary first aid while the user seeks professional care. Clearly state what not to do. If the supplied information is insufficient, say so. For severe or uncertain situations, recommend prompt professional/emergency medical care.'''

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

    prediction = predictor.predict(pil)
    item_list = [x.strip() for x in items.split(',') if x.strip()]
    context = retriever.retrieve(prediction['injury'], item_list)

    prompt = f'''{SYSTEM_RULES}\n\nDetected injury category: {prediction['injury']}\nClassifier confidence: {prediction['confidence']}%\n\nKnowledge base:\n{context}\n\nUser's available items: {item_list or 'Not provided'}\n\nReturn: 1) what the category means in this prototype, 2) immediate temporary first-aid actions supported by the knowledge base, 3) which supplied items are appropriate and exactly how the knowledge base says to use them, 4) items/actions to avoid, 5) warning signs and when to seek professional care. Do not claim that the classifier confirms a diagnosis.'''

    try:
        guidance = get_provider().generate(prompt)
    except Exception as exc:
        raise HTTPException(502, f'Configured LLM provider failed: {exc}')

    return {
        'prediction': prediction,
        'guidance': guidance,
        'matched_materials': context['matched_materials'],
        'sources': context['sources'],
        'provider': LLM_PROVIDER,
    }
