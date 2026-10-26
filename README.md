# WoundGuide – AI First-Aid Assistant

A college machine-learning project. The user uploads a photo of a **cut, burn or bruise** and lists the items they have. The system classifies the injury with a fine-tuned **EfficientNet-B0**, retrieves matching facts from a **JSON knowledge base (RAG)**, and an **LLM** (Groq or OmniRoute) turns those facts into clear first-aid guidance: what to do, what not to do, how to use the items, good signs and warning signs.

> **Safety note.** This is an educational prototype, not a medical device. It cannot diagnose injuries. Seek professional care for severe, worsening or uncertain injuries.

## Features
- Image classification: bruise, burn, cut (optional 4th class `not_wound`, see section 7)
- Rejects unclear or non-injury photos and asks for a better one
- Knowledge-base retrieval with item aliases (e.g. "ice" finds "ice pack", "haldi" finds "turmeric")
- Structured guidance cards: Do, Don't, Your items, Good signs, Warning signs
- Falls back to the knowledge base directly if the LLM fails, so the page never breaks
- Switchable LLM provider (Groq or OmniRoute) through `.env`, no automatic fallback between providers
- Responsive React interface with camera capture on phones

## Tech stack
| Layer | Tools |
|---|---|
| ML | PyTorch, torchvision EfficientNet-B0 (transfer learning + fine-tuning) |
| Backend | FastAPI, Uvicorn, Pillow, requests |
| Retrieval | JSON knowledge base, whole-word keyword matching |
| LLM | Groq (`openai/gpt-oss-120b`) or OmniRoute (OpenAI-compatible) |
| Frontend | React, Vite, plain CSS |

## Project structure
```text
backend/
  app/
    main.py              FastAPI app and CORS
    config.py            settings from .env, CLASS_NAMES
    api/routes.py        /api/health and /api/analyze (gate, RAG prompt, JSON parsing, fallback)
    ml/predictor.py      loads the model and predicts
    rag/retriever.py     loads knowledge_base.json + data/extra/*.json
    llm/                 base.py, groq.py, omniroute.py, provider.py
  data/
    knowledge_base.json  injuries, materials, sources
    extra/               drop extra knowledge files here (auto-merged)
  models/                put best_wound_model_finetuned.pth here
  verify_model.py        checks the model file loads
frontend/
  index.html, src/main.jsx, src/style.css
```

## How it works
```text
Photo + items -> React -> FastAPI /api/analyze
   1. EfficientNet-B0 predicts class + probabilities
   2. Gate: unclear / not_wound photo -> "please upload a proper photo"
   3. Retriever finds the injury entry and matching materials
   4. LLM receives ONLY these facts and returns JSON
   5. JSON is parsed (or built from the knowledge base if parsing fails)
   6. React shows the result cards
```
The LLM is told to use only the supplied facts, so it does not invent medical advice.

## 1. Requirements
- Python 3.11 or 3.12
- Node.js 20.19+ or 22.12+ (check with `node -v`)
- A Groq API key (free at console.groq.com) or a running OmniRoute server
- The trained model file `best_wound_model_finetuned.pth`

## 2. Backend setup (Windows PowerShell)
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```
1. Copy `best_wound_model_finetuned.pth` into `backend/models/`.
2. Open `backend/.env` in Notepad and paste your key: `GROQ_API_KEY=your_key_here`.
3. Start the server:
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
4. Open `http://127.0.0.1:8000/api/health`. You should see `"status":"ok"`.

macOS/Linux: use `python3 -m venv .venv`, `source .venv/bin/activate` and `cp .env.example .env`.

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process Bypass` once and try again.

## 3. Frontend setup
Open a second terminal:
```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```
Open `http://localhost:5173`.

## 4. Configuration (`backend/.env`)
```env
LLM_PROVIDER=groq              # groq or omniroute
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
OMNIROUTE_API_KEY=
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=auto
MODEL_PATH=models/best_wound_model_finetuned.pth
MAX_IMAGE_MB=10
MIN_CONFIDENCE=55              # optional: reject photos below this top-class %
MIN_MARGIN=12                  # optional: reject if top class leads 2nd by less than this
```
Restart the backend after changing `.env`. To switch to OmniRoute set `LLM_PROVIDER=omniroute` and fill its three values.

## 5. Knowledge base
`backend/data/knowledge_base.json` has three parts: `injuries` (what it is, steps, things to avoid, good signs, warning signs), `materials` (safe uses, things to avoid, aliases) and `sources`.

To add knowledge, create a new file such as `backend/data/extra/more.json` in the same format. It is merged when the backend starts. The content should be checked against trusted sources (Red Cross, NHS, Mayo Clinic, MedlinePlus) before real use.

## 6. Photos that are not injuries
A softmax classifier must always pick one of its classes, so an unrelated photo would still be labelled "cut". Two protections are used:
1. **Confidence gate (works now).** If the top class is below `MIN_CONFIDENCE`, or leads the second class by less than `MIN_MARGIN` points, the API returns `rejected: true` and the page asks for a better photo. Limitation: a random image can still get a high score by chance.
2. **Fourth class `not_wound` (better).** Train the model on non-injury images too. See section 7.

## 7. Adding the `not_wound` class (optional, recommended)
1. In Colab, run the cells from `colab/colab_add_not_wound_class.py`. Upload `own_negatives.zip` with 60+ of your own photos (healthy skin, hands, legs, rooms, objects) first. Own photos matter more than the CIFAR images.
2. Download `best_wound_model_v2.pth`.
3. Rename your old model to `best_wound_model_old.pth` as a backup, then save the new file as `backend/models/best_wound_model_finetuned.pth`.
4. In `backend/app/config.py` set `CLASS_NAMES = ['bruise', 'burn', 'cut', 'not_wound']` (alphabetical order is required because `ImageFolder` sorts class names).
5. In `backend/verify_model.py` change `classes` to the same list, run `python verify_model.py`, then restart the backend.

## 8. Model results (from the training notebook)
- Data: Kaggle "wound-dataset" (yasinpratomo), 231 images: 122 bruise, 59 burn, 50 cut. Split 70/15/15.
- Stage 1 trained only the new classifier layer; stage 2 fine-tuned the last three feature blocks at a lower learning rate.
- Best validation accuracy 90.91%, test accuracy 78.38% (37 test images).
- Per-class test F1: bruise 0.87, burn 0.74, cut 0.62.

The dataset is small, so results vary between photos. More images and more classes would improve it.

## 9. API
`POST /api/analyze` (multipart form: `image`, `items` as comma-separated text)

Accepted photo returns: `prediction`, `low_confidence`, `structured` (summary, do, dont, items, good_signs, warning_signs, see_doctor_now), `guidance_source` (`llm` or `knowledge_base`), `matched_materials`, `sources`, `provider`.

Rejected photo returns: `rejected: true`, `reason` (`unsure` or `not_wound`), `message`, `prediction`.

`GET /api/health` returns status and the active provider.

## 10. Troubleshooting
| Problem | Fix |
|---|---|
| "Cannot reach the backend" | Start the backend; check `VITE_API_URL` in `frontend/.env` |
| 502 "LLM provider failed" | Check the API key and `LLM_PROVIDER` in `backend/.env` (the knowledge-base fallback also covers this) |
| `Model not found` | Put the `.pth` file in `backend/models/` |
| `size mismatch` when loading | `CLASS_NAMES` length differs from the model's output classes |
| Good photo gets rejected | Lower `MIN_CONFIDENCE` / `MIN_MARGIN` in `.env` |

## 11. Limitations and future work
- Only three injury types, small dataset, small test set.
- Keyword retrieval; could be replaced with embeddings and a vector store (for example using MedlinePlus health-topic XML as the corpus).
- Possible additions: more classes (abrasion, laceration), severity estimation, multilingual output, user feedback logging.

## References
American Red Cross first aid, Mayo Clinic first aid, NHS, MedlinePlus (NIH), Kaggle wound dataset by yasinpratomo, Tan and Le (2019) EfficientNet.
