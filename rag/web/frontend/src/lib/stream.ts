import type { Citation } from './types';

// The backend sends no keepalive comments while it retrieves, reranks and
// loads the model, so the timeout measures silence rather than total time:
// it restarts after every chunk and only fires when the stream stops
// delivering anything at all (issue #406).
const STREAM_INACTIVITY_TIMEOUT_MS = 180_000;

export async function streamSSE(
  response: Response,
  onToken: (token: string) => void,
  onDone: (sources: Citation[] | null) => void,
  onError: (msg: string) => void,
  fb?: {
    unknown?: string;
    queryError?: string;
    modelError?: string;
    closed?: string;
    no_results?: string;
    question_too_short?: string;
    model_unavailable?: string;
    embedder_unavailable?: string;
    generation_error?: string;
  },
  controller?: AbortController,
  onStall?: () => void,
  onSources?: (sources: Citation[]) => void,
) {
  const contentType = response.headers.get('content-type') || '';

  // The backend's stable codes are the translatable part; `message` is its
  // Spanish fallback for clients that do not know the code (issue #379).
  const localized = (code: unknown, fallback: string) => {
    const mapped = typeof code === 'string' ? fb?.[code as keyof typeof fb] : undefined;
    return typeof mapped === 'string' ? mapped : fallback;
  };

  // If the response is NOT a stream (error responses come as JSON)
  if (!contentType.includes('event-stream')) {
    const data = await response.json().catch(() => ({ message: fb?.unknown ?? 'Error desconocido' }));
    if (!data.ok) {
      const raw = typeof data.message === 'string'
        ? data.message
        : typeof data.error === 'string'
          ? data.error
          : fb?.unknown ?? 'Error desconocido';
      onError(localized(data.error, raw));
    } else {
      onDone(null);
    }
    return;
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finished = false;

  const processBlock = (block: string) => {
    if (!block.trim()) return;

    let event = 'message';
    const dataLines: string[] = [];

    for (const rawLine of block.split(/\r?\n/)) {
      const line = rawLine.trimEnd();
      if (!line || line.startsWith(':')) continue;
      if (line.startsWith('event:')) {
        event = line.slice(6).trim() || 'message';
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart());
      }
    }

    if (!dataLines.length) return;

    try {
      const data = JSON.parse(dataLines.join('\n'));
      if (event === 'error' || data.error) {
        finished = true;
        const raw = data.message || data.error || fb?.modelError || 'Error en la respuesta del modelo';
        onError(localized(data.error, raw));
        return;
      }
      if (event === 'sources') {
        onSources?.(data.sources || []);
        return;
      }
      if (event === 'token' || data.token) onToken(data.token || '');
      if (event === 'done' || data.done) {
        finished = true;
        onDone(data.sources || null);
      }
    } catch {
      // Ignore malformed SSE blocks; the backend always sends JSON.
    }
  };

  // Restarted after every chunk, so a slow first token does not trip it while
  // the backend is still retrieving.
  let timer: ReturnType<typeof setTimeout> | undefined;
  const armTimer = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      onStall?.();
      controller?.abort();
    }, STREAM_INACTIVITY_TIMEOUT_MS);
  };

  try {
    armTimer();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      armTimer();

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || '';

      for (const block of blocks) processBlock(block);
    }

    buffer += decoder.decode();
    if (buffer.trim()) processBlock(buffer);
    if (!finished) onError(fb?.closed ?? 'La conexión se cerró antes de completar la respuesta.');
  } catch (err) {
    // A user stop or the inactivity timeout aborts the read; the caller has
    // already reported that, and the generic "connection closed" would be a
    // second, wrong diagnosis.
    if (!controller?.signal.aborted) throw err;
  } finally {
    clearTimeout(timer);
  }
}
