# MonkeyGrab web frontend

<p align="center">
  <img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white" alt="Vite">
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/Tailwind%20CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white" alt="Tailwind CSS">
</p>

React + Vite single-page app for the MonkeyGrab web interface.

The backend lives in [`rag/web/app.py`](../app.py) (Flask, port 5000) and serves
the built assets in production. During development this dev server runs on
port 3000 and proxies API calls to Flask.

## Run locally

> [!IMPORTANT]
> **Prerequisites:** Node.js 22.12+ (vitest's floor), and the Flask backend running (`python rag/web/app.py`).

```bash
pnpm install
pnpm run dev      # Vite on http://localhost:3000 — proxies /api → :5000
```

## Tests

```bash
pnpm run test     # vitest + Testing Library, jsdom, no network
```

`tests/` renders the real `App` against a fake backend (`tests/backend.ts`
answers every `/api` route; a route nobody declared fails the test). They
pin behaviour a refactor must not change: store selection, PDF upload, chat
stream handling, settings toggles and the page language. They live outside
`src/` on purpose: the structural gates in `tests/unit` scan `src/` for the
`i18n` keys read and the `api` members called, and a test must not count as
a consumer.

## Build for production

```bash
pnpm run build    # emits to rag/web/frontend/dist (gitignored)
```

Flask serves the built assets automatically when you open `http://localhost:5000`.
