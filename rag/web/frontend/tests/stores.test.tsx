import { describe, expect, it } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { callsTo, deferred, installBackend, sseResponse, token, done } from './backend';
import { renderApp, T } from './render';

const row = (label: string) => screen.getByRole('button', { name: new RegExp(label) });

describe('store selection', () => {
  it('lists the three stores with what each holds', async () => {
    installBackend();
    await renderApp();
    expect(row('English').textContent).toContain('120');
    expect(row('Castellano').textContent).toContain('40');
    expect(row('Valenci').textContent).toContain(T.storeNotIndexed);
  });

  it('asks the backend to switch and shows the documents of the new store', async () => {
    installBackend({
      'POST /stores/select': () => ({
        ok: true, indexing: false, documents: ['quijote.pdf'], fingerprint_stale: false,
        stores: [
          { name: 'en', label: 'English', pdf_count: 2, indexed: true, active: false, fragments: 120 },
          { name: 'es', label: 'Castellano', pdf_count: 1, indexed: true, active: true, fragments: 40 },
        ],
      }),
    });
    const user = await renderApp();
    expect(screen.getByText('attention.pdf')).toBeTruthy();

    await user.click(row('Castellano'));

    await screen.findByText('quijote.pdf');
    expect(screen.queryByText('attention.pdf')).toBeNull();
    expect(callsTo('POST', '/stores/select').map(c => c.body)).toEqual([{ name: 'es' }]);
  });

  it('makes no request when the active store is clicked again', async () => {
    installBackend();
    const user = await renderApp();
    await user.click(row('English'));
    expect(callsTo('POST', '/stores/select')).toHaveLength(0);
  });

  it('disables the rows while a switch is in flight', async () => {
    const reply = deferred<unknown>();
    installBackend({ 'POST /stores/select': () => reply.promise });
    const user = await renderApp();

    await user.click(row('Castellano'));
    await waitFor(() => expect((row('English') as HTMLButtonElement).disabled).toBe(true));

    reply.resolve({
      ok: true, indexing: false, documents: [],
      stores: [{ name: 'en', label: 'English', pdf_count: 2, indexed: true, active: false, fragments: 120 }],
    });
    await waitFor(() => expect((row('English') as HTMLButtonElement).disabled).toBe(false));
  });

  it('keeps the old documents and says why when the backend refuses (indexing in progress)', async () => {
    installBackend({
      'POST /stores/select': () => ({ ok: false, error: 'indexing_in_progress' }),
    });
    const user = await renderApp();

    await user.click(row('Castellano'));

    await screen.findByText(T.corpusConflict);
    expect(screen.getByText('attention.pdf')).toBeTruthy();
  });

  it('shows the generic store error for any other refusal', async () => {
    installBackend({ 'POST /stores/select': () => ({ ok: false, error: 'boom' }) });
    const user = await renderApp();
    await user.click(row('Castellano'));
    await screen.findByText(T.corpusError);
  });

  it('shows the connection error when the request itself fails', async () => {
    installBackend({
      'POST /stores/select': () => { throw new Error('network down'); },
    });
    const user = await renderApp();
    await user.click(row('Castellano'));
    await screen.findByText(T.corpusConnError);
    expect(screen.getByText('attention.pdf')).toBeTruthy();
  });

  it('drops the conversation, which belongs to the store it was asked of (#405)', async () => {
    installBackend({
      'POST /rag': () => sseResponse([token('Planck answered'), done([])]),
      'POST /stores/select': () => ({
        ok: true, indexing: false, documents: ['quijote.pdf'], stores: [],
      }),
    });
    const user = await renderApp();
    await user.type(screen.getByPlaceholderText(T.placeholderRag), 'who is planck?{Enter}');
    await screen.findByText('Planck answered');

    await user.click(row('Castellano'));

    await screen.findByText('quijote.pdf');
    expect(screen.queryByText('Planck answered')).toBeNull();
    expect(screen.queryByText('who is planck?')).toBeNull();
  });

  it('shows the indexing screen when the new store has to be indexed first', async () => {
    let inits = 0;
    installBackend({
      'GET /init': () => (++inits === 1
        ? { ok: true, mode: 'rag', documents: [], active_store: 'en', user: '', history: [] }
        : { ok: false, indexing: true, progress: { file: 'es1.pdf', file_index: 1, total_files: 3 } }),
      'POST /stores/select': () => ({ ok: true, indexing: true, stores: [] }),
    });
    const user = await renderApp();
    await user.click(row('Castellano'));
    await screen.findByText(T.indexingTitle);
  });
});
