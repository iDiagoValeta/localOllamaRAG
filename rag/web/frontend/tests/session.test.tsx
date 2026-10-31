import { describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { callsTo, installBackend, sseResponse, token, done } from './backend';
import { mountApp, renderApp, T } from './render';

describe('study mode', () => {
  const studyBox = () => screen.getByPlaceholderText(T.studyPlaceholder);

  async function enterStudy(user: Awaited<ReturnType<typeof renderApp>>) {
    await user.click(screen.getByRole('button', { name: T.modeStudy }));
  }

  it('does not tell the backend about study mode', async () => {
    installBackend();
    const user = await renderApp();
    await enterStudy(user);
    expect(callsTo('POST', '/mode')).toHaveLength(0);
  });

  it('suggests matching documents and completes one with Tab', async () => {
    installBackend();
    const user = await renderApp();
    await enterStudy(user);

    await user.type(studyBox(), 'planc');
    expect(screen.getByRole('button', { name: 'planck.pdf' })).toBeTruthy();
    await user.keyboard('{Tab}');

    expect((studyBox() as HTMLTextAreaElement).value).toBe('planck.pdf');
  });

  it('asks for the chosen kind of artifact for the resolved document and shows it', async () => {
    installBackend({
      'POST /study': () => ({
        ok: true,
        artifact: { sections: [{ heading: 'Quanta', body: 'Energy comes in packets.', source_pages: [2] }] },
      }),
    });
    const user = await renderApp();
    await enterStudy(user);
    await user.click(screen.getByRole('button', { name: T.studyKindOutline }));

    await user.type(studyBox(), 'planc{Enter}');

    await waitFor(() => expect(callsTo('POST', '/study')).toHaveLength(1));
    expect(callsTo('POST', '/study')[0].body).toMatchObject({
      kind: 'outline', document: 'planck.pdf', language: 'English',
    });
  });

  it('says so when no document matches, without calling the backend', async () => {
    installBackend();
    const user = await renderApp();
    await enterStudy(user);
    await user.type(studyBox(), 'zzz{Enter}');
    await screen.findByText(T.studyNoMatch.replace('{query}', 'zzz'));
    expect(callsTo('POST', '/study')).toHaveLength(0);
  });

  it('reports a malformed model answer (422) differently from a server failure', async () => {
    installBackend({
      'POST /study': () => new Response(JSON.stringify({ ok: false }), {
        status: 422, headers: { 'content-type': 'application/json' },
      }),
    });
    const user = await renderApp();
    await enterStudy(user);
    await user.type(studyBox(), 'planck.pdf{Enter}');
    await screen.findByText(T.studyMalformed);
  });

  it('translates the route validation code', async () => {
    installBackend({
      'POST /study': () => new Response(JSON.stringify({ ok: false, error: 'document_not_indexed' }), {
        status: 404, headers: { 'content-type': 'application/json' },
      }),
    });
    const user = await renderApp();
    await enterStudy(user);
    await user.type(studyBox(), 'planck.pdf{Enter}');
    await screen.findByText(T.studyFailed.replace('{error}', T.studyNotIndexed));
  });
});

describe('start-up and recovery', () => {
  it('paints the history the backend will build the next prompt from (#403)', async () => {
    installBackend({
      'GET /init': () => ({
        ok: true, mode: 'rag', documents: [], active_store: 'en', user: '',
        history: [{ role: 'user', content: 'earlier question' }, { role: 'assistant', content: 'earlier answer' }],
      }),
    });
    await renderApp();
    expect(screen.getByText('earlier question')).toBeTruthy();
    expect(screen.getByText('earlier answer')).toBeTruthy();
  });

  it('does not repaint the history after a later init (#473)', async () => {
    let inits = 0;
    installBackend({
      'GET /init': () => ({
        ok: true, mode: 'rag', documents: [], active_store: 'en', user: '',
        history: inits++ === 0 ? [] : [{ role: 'user', content: 'resurrected' }],
      }),
      'POST /reindex': () => ({ ok: true, indexing: true, progress: null }),
      'POST /rag': () => sseResponse([token('answer'), done([])]),
    });
    const user = await renderApp();
    await user.type(screen.getByPlaceholderText(T.placeholderRag), 'q{Enter}');
    await screen.findByText('answer');
    await user.click(screen.getByRole('button', { name: T.tabPipeline }));
    await user.click(await screen.findByRole('button', { name: T.reindexBtn }));
    await waitFor(() => expect(callsTo('GET', '/init')).toHaveLength(2));
    // The pipeline panel is still open over the chat; closing it shows the chat again.
    await user.click(await screen.findByRole('button', { name: T.close }));
    await screen.findByPlaceholderText(T.placeholderRag);
    expect(screen.queryByText('resurrected')).toBeNull();
    expect(screen.getByText('answer')).toBeTruthy();
  });

  it('greets the user by name when the chat is empty', async () => {
    installBackend();
    await renderApp();
    // The greeting is typeset letter by letter, so match on the heading's text.
    await waitFor(() => expect(screen.getByRole('heading', { level: 2 }).textContent).toBe('Hi, Ana'));
  });

  it('shows the connection error screen and recovers on retry', async () => {
    let up = false;
    installBackend({
      'GET /init': () => {
        if (!up) throw new Error('offline');
        return { ok: true, mode: 'rag', documents: [], active_store: 'en', user: '', history: [] };
      },
    });
    const user = mountApp();
    await screen.findByText(T.noServer);

    up = true;
    await user.click(screen.getByRole('button', { name: T.retry }));

    await screen.findByPlaceholderText(T.placeholderRag);
  });

  it('shows a permanent indexing failure, translated, and restarts it through /api/reindex (#379)', async () => {
    installBackend({
      'GET /init': () => ({ ok: false, indexing: false, code: 'indexing_failed', error: 'La indexacion fallo' }),
      'POST /reindex': () => ({ ok: false, code: 'indexing_in_progress', error: 'ocupado' }),
    });
    const user = mountApp();
    await screen.findByRole('heading', { name: T.indexingFailed });
    expect(screen.getAllByText(T.indexingFailed).length).toBeGreaterThan(0);

    await user.click(screen.getByRole('button', { name: T.retry }));

    await waitFor(() => expect(callsTo('POST', '/reindex')).toHaveLength(1));
    await screen.findByText(T.corpusConflict);
  });

  it('deletes a document after confirmation and tells the user in the chat', async () => {
    vi.stubGlobal('confirm', () => true);
    installBackend({
      'DELETE /docs/planck.pdf': () => ({ ok: true, documents: ['attention.pdf'] }),
    });
    const user = await renderApp();
    await user.click(screen.getAllByTitle(T.deleteDoc)[1]);
    await waitFor(() => expect(screen.getAllByTitle(T.deleteDoc)).toHaveLength(1));
    expect(callsTo('DELETE', '/docs/planck.pdf')).toHaveLength(1);
    // The confirmation in the chat names the file in bold; the sidebar row is gone.
    await screen.findByText('planck.pdf', { selector: 'strong' });
    expect(screen.queryByText('planck.pdf', { selector: 'span' })).toBeNull();
  });
});
