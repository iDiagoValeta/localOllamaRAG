// A fake Flask backend behind `fetch`. Every route the app touches has a
// default answer; a test overrides only the one it is about. A request to a
// route nobody declared fails loudly, so a test never passes by talking to a
// server that is not there.
import { vi } from 'vitest';

export interface Call {
  method: string;
  path: string;
  body: unknown;
  form: FormData | null;
  signal: AbortSignal | null;
}

type Reply = unknown | Response;
type Handler = (call: Call) => Reply | Promise<Reply>;

const STORES = [
  { name: 'en', label: 'English', docs_folder: 'rag/docs/en', pdf_count: 2, indexed: true, active: true, fragments: 120 },
  { name: 'es', label: 'Castellano', docs_folder: 'rag/docs/es', pdf_count: 1, indexed: true, active: false, fragments: 40 },
  { name: 'ca', label: 'Valencià', docs_folder: 'rag/docs/ca', pdf_count: 0, indexed: false, active: false, fragments: null },
];

const DOCS = ['attention.pdf', 'planck.pdf'];

const defaults: Record<string, Handler> = {
  'GET /init': () => ({
    ok: true, mode: 'rag', documents: DOCS, fingerprint_stale: false,
    active_store: 'en', user: 'Ana', history: [],
  }),
  'POST /mode': () => ({ ok: true }),
  'POST /clear': () => ({ ok: true }),
  'GET /stores': () => ({ ok: true, stores: STORES, active: 'en' }),
  'GET /ollama': () => ({ ok: true, running: true, version: '0.9.0' }),
  'GET /ollama/models': () => ({ ok: true, models: [{ name: 'gemma' }] }),
  'GET /models': () => ({ ok: true, roles: { rag: 'gemma', chat: 'gemma', contextual: 'gemma', recomp: 'gemma' } }),
  'GET /settings': () => ({ ok: true, settings: {} }),
};

const calls: Call[] = [];

/** Requests made so far to one route, oldest first. */
export function callsTo(method: string, path: string): Call[] {
  return calls.filter(c => c.method === method && c.path === path);
}

export function installBackend(overrides: Record<string, Handler> = {}): void {
  calls.length = 0;
  const routes = { ...defaults, ...overrides };
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = (init.method ?? 'GET').toUpperCase();
    const path = url.replace(/^\/api/, '');
    const form = init.body instanceof FormData ? init.body : null;
    const call: Call = {
      method,
      path,
      body: typeof init.body === 'string' ? JSON.parse(init.body) : null,
      form,
      signal: init.signal ?? null,
    };
    calls.push(call);
    const handler = routes[`${method} ${path}`];
    if (!handler) throw new Error(`unexpected request: ${method} ${path}`);
    const reply = await handler(call);
    return reply instanceof Response
      ? reply
      : new Response(JSON.stringify(reply), { headers: { 'content-type': 'application/json' } });
  });
}

const encoder = new TextEncoder();

/** A finished server-sent-event response made of the given blocks. */
export function sseResponse(blocks: string[]): Response {
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const block of blocks) controller.enqueue(encoder.encode(`${block}\n\n`));
        controller.close();
      },
    }),
    { headers: { 'content-type': 'text/event-stream' } },
  );
}

export const token = (t: string) => `event: token\ndata: ${JSON.stringify({ token: t })}`;
export const done = (sources: unknown[] | null = null) =>
  `event: done\ndata: ${JSON.stringify({ done: true, sources })}`;

/** A stream the test feeds by hand, and that errors when the request is aborted. */
export function openStream(signal: AbortSignal | null) {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
      signal?.addEventListener('abort', () =>
        c.error(new DOMException('The operation was aborted.', 'AbortError')));
    },
  });
  return {
    response: new Response(body, { headers: { 'content-type': 'text/event-stream' } }),
    push: (block: string) => controller.enqueue(encoder.encode(`${block}\n\n`)),
  };
}

/** A promise the test resolves later, to observe what the UI does meanwhile. */
export function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(r => { resolve = r; });
  return { promise, resolve };
}
