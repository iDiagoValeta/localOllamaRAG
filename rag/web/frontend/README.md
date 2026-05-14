# MonkeyGrab web frontend

React + Vite single-page app for the MonkeyGrab web interface.

The backend lives in [`rag/web/app.py`](../app.py) (Flask, port 5000) and serves
the built assets in production. During development this dev server runs on
port 3000 and proxies API calls to Flask.

## Run locally

**Prerequisites:** Node.js 20+, the Flask backend running (`python rag/web/app.py`).

```bash
npm install
npm run dev      # Vite on http://localhost:3000 — proxies /api → :5000
```

## Build for production

```bash
npm run build    # emits to rag/web/frontend/dist (gitignored)
```

Flask serves the built assets automatically when you open `http://localhost:5000`.
