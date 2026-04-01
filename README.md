# SynapTech

SynapTech is a full-stack connectome simulation app built around a C. elegans neural graph.

## Repo layout

- `backend/` — FastAPI API, simulation engine integration, behavioral metrics, intervention workflows
- `frontend/` — Vite + TypeScript visualization UI
- `brian2/` — vendored Brian2 source used by the simulation stack
- `c302/` — vendored OpenWorm/c302 assets and models

## Backend setup

From the repo root:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
```

Run the API:

```bash
backend/.venv/bin/python -m uvicorn app.main:app --reload --port 8001 --app-dir backend
```

Backend runs at: `http://127.0.0.1:8001`

## Frontend setup

In another terminal:

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

Frontend runs at: `http://127.0.0.1:3000`

The Vite dev server proxies `/api/*` requests to `http://127.0.0.1:8001`.

## Root npm scripts

Useful shortcuts from the repo root:

```bash
npm run backend:bootstrap
npm run backend:dev
npm run frontend:dev
npm run frontend:build
npm test
```

## Tests

Run backend tests:

```bash
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests -q
```

Current validated status in this workspace:
- backend test suite passes
- frontend production build succeeds

## Selected API endpoints

- `GET /api/health`
- `GET /api/message`
- `GET /api/connectome/graph`
- `GET /api/connectome/summary`
- `GET /api/simulation/live`
- `GET /api/simulation/spikes`
- `POST /api/simulation/session/reset`
- `POST /api/simulation/behavior/performance`
- `POST /api/replacement/start`
- `POST /api/replacement/step`
- `GET /api/simulation/replacement-sweep/stream`

## Notes

Large directories such as `brian2/` and `c302/` appear to be intentionally vendored dependencies or research assets. They should be trimmed only with explicit confirmation that they are no longer required by the app.
