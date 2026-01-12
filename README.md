# ChemCheck

Initial skeleton for a FastAPI backend and Streamlit frontend that accepts front/back label images.

## Structure
- `backend/`: FastAPI app (routers + services + models)
- `frontend/`: Streamlit UI

## Setup
```bash
uv venv .venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

## Run backend
```bash
uvicorn app.main:app --reload --app-dir backend
```

## Run frontend
```bash
streamlit run frontend/streamlit_app.py
```

## Environment
- `API_BASE_URL` (optional, default: `http://localhost:8000`)

## Notes
- The API accepts `front_image` and/or `back_image`. At least one image is required.
