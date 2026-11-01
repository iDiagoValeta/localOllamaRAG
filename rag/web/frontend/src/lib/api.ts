import type { Strings } from './i18n';
import type { Mode, ModelRoles, PipelineSettings, StudyKind } from './types';

const API_BASE = '/api';

// Failure codes the UI says in the user's language. A route's `code` (or, for
// the older routes, an `error` that is already a code) is the contract; the
// `error` prose is Spanish for clients without i18n, so it is only the
// fallback for a code not listed here (issue #555). A Map, so a code named
// like an Object.prototype member cannot resolve to something inherited.
// upload_failed and no_file are deliberately absent: their only text is the
// server's own, which handleAddDocs puts inside addPdfsError (a template with
// {error}; mapping the code here would fill it twice).
const ERROR_CODE_KEYS = new Map<string, keyof Strings>([
  ['invalid_filename', 'errInvalidFilename'],
  ['indexing_in_progress', 'corpusConflict'],
  ['invalid_flag_value', 'settingsSaveError'],
  ['indexing_failed', 'indexingFailed'],
]);

export function apiErrorText(
  strings: Strings,
  body: { code?: unknown; error?: unknown; detail?: unknown } | null | undefined,
  fallback: string,
): string {
  // `detail` is raw exception text: untranslatable, but it says what went
  // wrong, which the generic line for the same code does not.
  if (typeof body?.detail === 'string' && body.detail) return body.detail;
  const code = typeof body?.code === 'string' ? body.code : body?.error;
  const key = typeof code === 'string' ? ERROR_CODE_KEYS.get(code) : undefined;
  if (key) return strings[key];
  return typeof body?.error === 'string' && body.error ? body.error : fallback;
}

export const api = {
  init: () =>
    fetch(`${API_BASE}/init`).then(r => r.json()),

  setMode: (mode: Mode) =>
    fetch(`${API_BASE}/mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    }).then(r => r.json()),

  chat: (message: string, signal?: AbortSignal) =>
    fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, stream: true }),
      signal,
    }),

  rag: (message: string, signal?: AbortSignal) =>
    fetch(`${API_BASE}/rag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, stream: true }),
      signal,
    }),

  clear: () =>
    fetch(`${API_BASE}/clear`, { method: 'POST' }).then(r => r.json()),

  listStores: () =>
    fetch(`${API_BASE}/stores`).then(r => r.json()),

  selectStore: (name: string) =>
    fetch(`${API_BASE}/stores/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).then(r => r.json()),

  ollamaStatus: () =>
    fetch(`${API_BASE}/ollama`).then(r => r.json()),

  startOllama: () =>
    fetch(`${API_BASE}/ollama/start`, { method: 'POST' }).then(r => r.json()),

  ollamaModels: () =>
    fetch(`${API_BASE}/ollama/models`).then(r => r.json()),

  getModels: () =>
    fetch(`${API_BASE}/models`).then(r => r.json()),

  setModels: (roles: Partial<ModelRoles>) =>
    fetch(`${API_BASE}/models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(roles),
    }).then(r => r.json()),

  reindex: () =>
    fetch(`${API_BASE}/reindex`, { method: 'POST' }).then(r => r.json()),

  // Adds documents without touching what is already indexed; the call returns
  // once they are indexed, so no polling follows it.
  addDocs: (files: File[]) => {
    const form = new FormData();
    files.forEach(f => form.append('file', f));
    return fetch(`${API_BASE}/docs`, { method: 'POST', body: form }).then(r => r.json());
  },

  getSettings: () =>
    fetch(`${API_BASE}/settings`).then(r => r.json()),

  updateSettings: (settings: Partial<PipelineSettings>) =>
    fetch(`${API_BASE}/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    }).then(r => r.json()),

  deleteDoc: (filename: string) =>
    fetch(`${API_BASE}/docs/${encodeURIComponent(filename)}`, { method: 'DELETE' }).then(r => r.json()),

  // One call for the three artifacts, mirroring the single route. The status
  // code is kept: 422 means the model ignored the format, which the panel
  // reports differently from a server failure because the user's next move
  // differs too (retry or change model, versus check the backend).
  study: (kind: StudyKind, document: string, language: string, questionCount?: number) =>
    fetch(`${API_BASE}/study`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, document, language, question_count: questionCount }),
    }).then(async r => ({ status: r.status, body: await r.json() })),
};
