# Synaptech Starter

This repo contains a minimal full-stack starter:
- `backend/` -> FastAPI API
- `frontend/` -> Vite + TypeScript UI

## 1) Run the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend runs at: `http://127.0.0.1:8000`

### Optional: run backend via npm or yarn (from repo root)

```powershell
# one-time setup
npm run backend:bootstrap
# or: yarn backend:bootstrap

# start FastAPI
npm run backend:dev
# or: yarn backend:dev
```

## 2) Run the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://127.0.0.1:5173`

You can also run frontend from repo root:

```powershell
npm run frontend:dev
# or: yarn frontend:dev
```

The frontend calls `/api/health` and `/api/message` through Vite's proxy to the backend.

## API endpoints

- `GET /api/health`
- `GET /api/message`
