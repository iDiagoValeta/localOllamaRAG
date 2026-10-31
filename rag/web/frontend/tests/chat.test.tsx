import { describe, expect, it, vi } from 'vitest';
import { act, screen, waitFor } from '@testing-library/react';
import { callsTo, done, installBackend, openStream, sseResponse, token } from './backend';
import { renderApp, T } from './render';

// The app's inactivity timeout (STREAM_INACTIVITY_TIMEOUT_MS, issue #406).
// It is deliberately duplicated: a change to the value should be a decision
// that touches this test, not something it silently follows.
const STALL_MS = 180_000;

const composer = () => screen.getByPlaceholderText(T.placeholderRag) as HTMLTextAreaElement;

describe('chat stream handling', () => {
  it('appends tokens to the answer and posts the question to /api/rag with streaming on', async () => {
    installBackend({
      'POST /rag': () => sseResponse([token('Hel'), token('lo '), token('world'), done([])]),
    });
    const user = await renderApp();

    await user.type(composer(), 'greet me{Enter}');

    await screen.findByText('Hello world');
    expect(callsTo('POST', '/rag')[0].body).toEqual({ message: 'greet me', stream: true });
    expect(screen.getByText('greet me')).toBeTruthy();
    expect(composer().value).toBe('');
  });

  it('sends chat mode to /api/chat', async () => {
    installBackend({ 'POST /chat': () => sseResponse([token('plain'), done()]) });
    const user = await renderApp();
    await user.click(screen.getByRole('button', { name: 'CHAT' }));
    await user.type(screen.getByPlaceholderText(T.placeholderChat), 'hi{Enter}');
    await screen.findByText('plain');
    expect(callsTo('POST', '/mode')[0].body).toEqual({ mode: 'chat' });
    expect(callsTo('POST', '/chat')).toHaveLength(1);
    expect(callsTo('POST', '/rag')).toHaveLength(0);
  });

  it('shows cited sources and the document count when the answer finishes', async () => {
    installBackend({
      'POST /rag': () => sseResponse([
        token('See the paper.'),
        done([{ document: 'planck.pdf', pages: [3, 4], best_page: 4 }]),
      ]),
    });
    const user = await renderApp();
    await user.type(composer(), 'planck?{Enter}');

    const cite = await screen.findByRole('button', { name: /planck\.pdf/ });
    expect(cite.textContent).toContain('p. 4');
    expect(screen.getByText(T.sources.replace('{n}', '1'))).toBeTruthy();
  });

  it('shows sources announced before the first token (#376)', async () => {
    installBackend({
      'POST /rag': () => sseResponse([
        `event: sources\ndata: ${JSON.stringify({ sources: [{ document: 'early.pdf', pages: [1] }] })}`,
        `event: error\ndata: ${JSON.stringify({ error: 'model_unavailable', message: 'modelo no disponible' })}`,
      ]),
    });
    const user = await renderApp();
    await user.type(composer(), 'q?{Enter}');

    await screen.findByText(T.modelUnavailable);
    expect(screen.getByRole('button', { name: /early\.pdf/ })).toBeTruthy();
  });

  it('translates the error code of a stream error event, not its Spanish message', async () => {
    installBackend({
      'POST /rag': () => sseResponse([
        `event: error\ndata: ${JSON.stringify({ error: 'no_results', message: 'Sin resultados' })}`,
      ]),
    });
    const user = await renderApp();
    await user.type(composer(), 'anything{Enter}');
    await screen.findByText(T.noResults);
    expect(screen.queryByText('Sin resultados')).toBeNull();
  });

  it('translates the generation_error default code (#475)', async () => {
    installBackend({
      'POST /rag': () => sseResponse([
        `event: error\ndata: ${JSON.stringify({ error: 'generation_error', message: 'Fallo' })}`,
      ]),
    });
    const user = await renderApp();
    await user.type(composer(), 'anything{Enter}');
    await screen.findByText(T.modelError);
  });

  it('shows an unknown error code with the backend message as the fallback', async () => {
    installBackend({
      'POST /rag': () => sseResponse([
        `event: error\ndata: ${JSON.stringify({ error: 'new_code', message: 'Algo nuevo' })}`,
      ]),
    });
    const user = await renderApp();
    await user.type(composer(), 'anything{Enter}');
    await screen.findByText('Algo nuevo');
  });

  it('translates a JSON error response that is not a stream (question too short)', async () => {
    installBackend({
      'POST /rag': () => new Response(
        JSON.stringify({ ok: false, error: 'question_too_short', message: 'Demasiado corta' }),
        { status: 400, headers: { 'content-type': 'application/json' } },
      ),
    });
    const user = await renderApp();
    await user.type(composer(), 'a{Enter}');
    await screen.findByText(T.questionTooShort);
  });

  it('reports a stream that closes before it finishes', async () => {
    installBackend({ 'POST /rag': () => sseResponse([token('partial')]) });
    const user = await renderApp();
    await user.type(composer(), 'q{Enter}');
    await screen.findByText(T.connClosed);
  });

  it('reports a request that cannot be made at all', async () => {
    installBackend({ 'POST /rag': () => { throw new Error('offline'); } });
    const user = await renderApp();
    await user.type(composer(), 'q{Enter}');
    await screen.findByText(T.connClosed);
    expect(composer().disabled).toBe(false);
  });

  it('does not send while the answer is still being written', async () => {
    const stream = openStreamHolder();
    installBackend({ 'POST /rag': call => stream.open(call.signal) });
    const user = await renderApp();
    await user.type(composer(), 'first{Enter}');
    await waitFor(() => expect(composer().disabled).toBe(true));
    expect(callsTo('POST', '/rag')).toHaveLength(1);
  });

  it('keeps the partial text when the user stops, without reporting an error', async () => {
    const stream = openStreamHolder();
    installBackend({ 'POST /rag': call => stream.open(call.signal) });
    const user = await renderApp();
    await user.type(composer(), 'q{Enter}');
    await waitFor(() => expect(stream.current).toBeTruthy());
    await act(async () => { stream.current!.push(token('half an ans')); });
    await screen.findByText('half an ans');

    await user.click(screen.getByRole('button', { name: T.stopGenerating }));

    expect(screen.getByText('half an ans')).toBeTruthy();
    await waitFor(() => expect(composer().disabled).toBe(false));
    expect(callsTo('POST', '/rag')[0].signal!.aborted).toBe(true);
    expect(screen.queryByText(T.connClosed)).toBeNull();
    expect(screen.queryByText(T.streamStalled)).toBeNull();
  });

  describe('stalled stream (#406)', () => {
    it('unlocks the composer with a "server stopped responding" message after 180 s of silence', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      const stream = openStreamHolder();
      installBackend({ 'POST /rag': call => stream.open(call.signal) });
      const user = await renderApp('en', { advanceTimers: vi.advanceTimersByTime });
      await user.type(composer(), 'q{Enter}');
      await waitFor(() => expect(composer().disabled).toBe(true));

      await act(async () => { await vi.advanceTimersByTimeAsync(STALL_MS - 1_000); });
      expect(composer().disabled).toBe(true);
      expect(screen.queryByText(T.streamStalled)).toBeNull();

      await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });

      await screen.findByText(T.streamStalled);
      expect(composer().disabled).toBe(false);
      expect(screen.queryByText(T.connClosed)).toBeNull();
      expect(callsTo('POST', '/rag')[0].signal!.aborted).toBe(true);
    });

    it('measures silence, not total time: every chunk restarts the clock', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      const stream = openStreamHolder();
      installBackend({ 'POST /rag': call => stream.open(call.signal) });
      const user = await renderApp('en', { advanceTimers: vi.advanceTimersByTime });
      await user.type(composer(), 'q{Enter}');
      await waitFor(() => expect(stream.current).toBeTruthy());

      await act(async () => { await vi.advanceTimersByTimeAsync(120_000); });
      await act(async () => { stream.current!.push(token('still alive')); });
      await screen.findByText('still alive');

      // 240 s since the request, but only 120 s since the last chunk.
      await act(async () => { await vi.advanceTimersByTimeAsync(120_000); });
      expect(screen.queryByText(T.streamStalled)).toBeNull();
      expect(composer().disabled).toBe(true);

      await act(async () => { await vi.advanceTimersByTimeAsync(61_000); });
      await screen.findByText(T.streamStalled);
      expect(composer().disabled).toBe(false);
    });
  });
});

/** Lets a test reach the stream the app opened, since the route handler only returns its response. */
function openStreamHolder() {
  const holder: { current: ReturnType<typeof openStream> | null; open: (s: AbortSignal | null) => Response } = {
    current: null,
    open(signal) {
      holder.current = openStream(signal);
      return holder.current.response;
    },
  };
  return holder;
}
