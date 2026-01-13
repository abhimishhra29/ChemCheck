# ChemCheck

FastAPI backend + Streamlit frontend for processing label images, with an MCP server used for search.

## Structure
- `backend/`: FastAPI app (routers + services + models)
- `backend/app/mcp_server/`: MCP server + Tavily client utilities
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

## Run MCP server
```bash
cd backend
python -m app.mcp_server.mcp_server
```

## Run frontend
```bash
streamlit run frontend/streamlit_app.py
```

## Environment
- `API_BASE_URL` (optional, default: `http://localhost:8000`)
- `TAVILY_API_KEY` (required for Tavily search)

## Notes
- The API accepts `front_image` and/or `back_image`. At least one image is required.
