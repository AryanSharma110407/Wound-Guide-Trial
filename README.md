# WoundGuide — AI First-Aid Prototype

A lightweight college-project prototype that combines a fine-tuned EfficientNet-B0 wound classifier, a small retrieval-based knowledge base, and a selectable LLM provider (Gemini or OmniRoute).

## Current scope
- Image classification: **bruise, burn, cut**
- Existing model: `best_wound_model_finetuned.pth`
- Knowledge retrieval: injury guidance + available-material guidance
- LLM provider: **Gemini by default OR OmniRoute by explicit configuration**
- No automatic Gemini fallback
- React frontend with mobile-friendly image capture support
- FastAPI backend

## Important model detail
Your Colab uses `torchvision.datasets.ImageFolder`. `ImageFolder` sorts class-directory names alphabetically, so the actual model mapping is:

```text
0 -> bruise
1 -> burn
2 -> cut
```

The order of the earlier `classes` dictionary (`cut`, `burn`, `bruise`) does **not** determine the labels used by `ImageFolder`.

Your PDF also prints `train_data.classes` as `['bruise', 'burn', 'cut']` and your inference cell currently uses the same order. Therefore **you do not need to retrain the model because of the class-order issue**.

## 1. Put the trained model in place
Copy:

`best_wound_model_finetuned.pth`

to:

`backend/models/best_wound_model_finetuned.pth`

Do not rename the file unless you also change `MODEL_PATH` in `.env`.

## 2. Backend setup
Use Python 3.11 or 3.12.

Windows PowerShell:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and put your Gemini key in `GEMINI_API_KEY`.

Default configuration:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=YOUR_KEY
```

Start the backend:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check:

`http://127.0.0.1:8000/api/health`

## 3. Frontend setup
Open another terminal:

```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open the Vite URL shown in the terminal, normally:

`http://localhost:5173`

## 4. Switch to OmniRoute
There is **no automatic fallback**.

Edit `backend/.env`:

```env
LLM_PROVIDER=omniroute
OMNIROUTE_API_KEY=YOUR_OMNIROUTE_KEY
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=auto
```

Make sure your OmniRoute server is running and the configured model is available. Restart FastAPI after changing `.env`.

To switch back:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=YOUR_GEMINI_KEY
```

The frontend and ML pipeline do not change.

## 5. Colab class-order decision
No retraining is required.

In your PDF, the relevant code is the inference cell labelled `CELL 1: INITIALIZE THE TRAINED MODEL`. Near `Step 6: Define the class names`, you currently have:

```python
class_names = ["bruise", "burn", "cut"]
```

That is **correct** for your current trained model because `ImageFolder` reports:

```python
train_data.classes
# ['bruise', 'burn', 'cut']
```

If you want to make the inference notebook less dependent on a manually repeated list, you can change it to:

```python
class_names = ['bruise', 'burn', 'cut']
```

and keep the same order in the backend's central configuration. This is a code-organization improvement, not a model-training fix.

## 6. What is intentionally lightweight
The RAG layer uses a small JSON knowledge base and keyword matching rather than a vector database. This keeps the first prototype fast and easy to run. It can later be replaced by embeddings/vector retrieval without changing the frontend contract.

The LLM receives retrieved facts rather than being asked to invent medical instructions. The current knowledge base is a prototype and should be expanded/verified with authoritative first-aid sources before treating it as a serious medical tool.

## 7. Architecture

```text
Image + available items
        |
        v
React frontend
        |
        v
FastAPI
  |         \
  v          v
EfficientNet  Knowledge Base
  |             |
  +-------> LLM Provider
              |       |
           Gemini  OmniRoute
              |
              v
        First-aid guidance
```

## 8. Replacing the classifier later
Keep the API response contract the same. If you later train more classes or replace EfficientNet, update the model loader and the single central `CLASS_NAMES` configuration. The RAG and frontend architecture can remain unchanged.

## Safety note
This is a college-project prototype for first-aid education. Image classification is not a medical diagnosis. The app should direct users to professional/emergency care when symptoms are severe, worsening, uncertain, or outside the supported categories.
