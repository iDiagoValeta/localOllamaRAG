import { describe, expect, it } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { callsTo, deferred, installBackend } from './backend';
import { renderApp, T } from './render';

async function openPipeline(user: Awaited<ReturnType<typeof renderApp>>) {
  await user.click(screen.getByRole('button', { name: T.tabPipeline }));
  await screen.findByRole('dialog');
}

const reranker = () => screen.getByRole('switch', { name: new RegExp(T.labelReranker) });

describe('pipeline settings toggles', () => {
  it('starts from what the backend reports', async () => {
    installBackend({
      'GET /settings': () => ({ ok: true, settings: { reranker: false, hybridSearch: true } }),
    });
    const user = await renderApp();
    await openPipeline(user);
    expect(reranker().getAttribute('aria-checked')).toBe('false');
    expect(screen.getByRole('switch', { name: new RegExp(T.labelHybrid) }).getAttribute('aria-checked')).toBe('true');
  });

  it('flips the switch at once, posts only that flag, and keeps the server value', async () => {
    const reply = deferred<unknown>();
    installBackend({
      'GET /settings': () => ({ ok: true, settings: { reranker: true } }),
      'POST /settings': () => reply.promise,
    });
    const user = await renderApp();
    await openPipeline(user);

    await user.click(reranker());

    expect(reranker().getAttribute('aria-checked')).toBe('false');
    expect(callsTo('POST', '/settings')[0].body).toEqual({ reranker: false });

    reply.resolve({ ok: true, settings: { reranker: false }, fingerprint_stale: false });
    await waitFor(() => expect(reranker().getAttribute('aria-checked')).toBe('false'));
  });

  it('shows the value the server settled on when it overrides the request', async () => {
    installBackend({
      'GET /settings': () => ({ ok: true, settings: { reranker: false } }),
      'POST /settings': () => ({ ok: true, settings: { reranker: false }, fingerprint_stale: false }),
    });
    const user = await renderApp();
    await openPipeline(user);

    await user.click(reranker());

    await waitFor(() => expect(callsTo('POST', '/settings')).toHaveLength(1));
    await waitFor(() => expect(reranker().getAttribute('aria-checked')).toBe('false'));
  });

  it('rolls the switch back and translates the code when the save is refused', async () => {
    installBackend({
      'GET /settings': () => ({ ok: true, settings: { reranker: true } }),
      'POST /settings': () => ({ ok: false, code: 'invalid_flag_value', error: 'Valor no valido' }),
    });
    const user = await renderApp();
    await openPipeline(user);

    await user.click(reranker());

    await screen.findByText(T.settingsSaveError);
    expect(reranker().getAttribute('aria-checked')).toBe('true');
  });

  it('prefers the raw detail of a failure over the generic line', async () => {
    installBackend({
      'GET /settings': () => ({ ok: true, settings: { reranker: true } }),
      'POST /settings': () => ({
        ok: false, code: 'invalid_flag_value', error: 'Valor no valido', detail: 'reranker unavailable: no CUDA',
      }),
    });
    const user = await renderApp();
    await openPipeline(user);
    await user.click(reranker());
    await screen.findByText('reranker unavailable: no CUDA');
  });

  it('falls back to the error text for a code it does not know', async () => {
    installBackend({
      'GET /settings': () => ({ ok: true, settings: { reranker: true } }),
      'POST /settings': () => ({ ok: false, code: 'something_new', error: 'Algo fallo' }),
    });
    const user = await renderApp();
    await openPipeline(user);
    await user.click(reranker());
    await screen.findByText('Algo fallo');
  });

  it('uses the generic line when the request fails outright', async () => {
    installBackend({
      'GET /settings': () => ({ ok: true, settings: { reranker: true } }),
      'POST /settings': () => { throw new Error('offline'); },
    });
    const user = await renderApp();
    await openPipeline(user);
    await user.click(reranker());
    await screen.findByText(T.settingsSaveError);
    expect(reranker().getAttribute('aria-checked')).toBe('true');
  });

  it('warns that the index is stale when an index-time flag makes it so', async () => {
    installBackend({
      'GET /settings': () => ({ ok: true, settings: { contextualRetrieval: true } }),
      'POST /settings': () => ({
        ok: true, settings: { contextualRetrieval: false }, fingerprint_stale: true,
      }),
    });
    const user = await renderApp();
    expect(screen.queryByText(T.fingerprintStaleWarning)).toBeNull();
    await openPipeline(user);

    await user.click(screen.getByRole('switch', { name: new RegExp(T.labelContextual) }));

    await screen.findByText(T.fingerprintStaleWarning);
  });
});
