import { describe, expect, it } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { callsTo, deferred, installBackend } from './backend';
import { renderApp, T } from './render';

const pdf = (name: string) => new File(['%PDF-1.4'], name, { type: 'application/pdf' });

/** The picker behind the "Add PDFs" button: a hidden input, so tests reach it directly. */
const picker = () => document.querySelector<HTMLInputElement>('input[type="file"]')!;

async function openPipeline(user: Awaited<ReturnType<typeof renderApp>>) {
  await user.click(screen.getByRole('button', { name: T.tabPipeline }));
  await screen.findByRole('dialog');
}

describe('adding PDFs', () => {
  it('posts every chosen file as multipart "file" parts to /api/docs, and shows the new list', async () => {
    installBackend({
      'POST /docs': () => ({ ok: true, documents: ['attention.pdf', 'planck.pdf', 'new.pdf', 'newer.pdf'] }),
    });
    const user = await renderApp();
    await openPipeline(user);

    await user.upload(picker(), [pdf('new.pdf'), pdf('newer.pdf')]);

    await waitFor(() => expect(callsTo('POST', '/docs')).toHaveLength(1));
    const form = callsTo('POST', '/docs')[0].form!;
    expect(form.getAll('file').map(f => (f as File).name)).toEqual(['new.pdf', 'newer.pdf']);
    // Nothing is reindexed: the call is add-only and no polling follows it.
    expect(callsTo('POST', '/reindex')).toHaveLength(0);
    await screen.findByText('newer.pdf');
  });

  it('only accepts PDFs in the picker and allows several', async () => {
    installBackend();
    await renderApp();
    expect(picker().accept).toBe('.pdf');
    expect(picker().multiple).toBe(true);
  });

  it('shows the busy status while the upload runs and blocks a second one', async () => {
    const reply = deferred<unknown>();
    installBackend({ 'POST /docs': () => reply.promise });
    const user = await renderApp();
    await openPipeline(user);

    await user.upload(picker(), pdf('slow.pdf'));

    await screen.findByText(T.addingPdfsStatus);
    expect((screen.getByRole('button', { name: T.reindexBtn }) as HTMLButtonElement).disabled).toBe(true);

    reply.resolve({ ok: true, documents: ['attention.pdf', 'planck.pdf', 'slow.pdf'] });
    await screen.findByText('slow.pdf');
    expect(screen.queryByText(T.addingPdfsStatus)).toBeNull();
  });

  it('names the files that failed to index and still refreshes the list (partial failure)', async () => {
    installBackend({
      'POST /docs': () => ({
        ok: false, failed_files: ['bad.pdf'], documents: ['attention.pdf', 'planck.pdf', 'good.pdf'],
      }),
    });
    const user = await renderApp();
    await openPipeline(user);

    await user.upload(picker(), [pdf('good.pdf'), pdf('bad.pdf')]);

    await screen.findByText(T.addPdfsFailed.replace('{files}', 'bad.pdf'));
    expect(screen.getByText('good.pdf')).toBeTruthy();
  });

  it('translates a backend error code instead of showing the Spanish prose', async () => {
    installBackend({
      'POST /docs': () => ({ ok: false, code: 'invalid_filename', error: 'Nombre no valido' }),
    });
    const user = await renderApp();
    await openPipeline(user);

    await user.upload(picker(), pdf('x.pdf'));

    await screen.findByText(T.addPdfsError.replace('{error}', T.errInvalidFilename));
  });

  it('puts a failure with no code in the error line as the server wrote it', async () => {
    installBackend({ 'POST /docs': () => ({ ok: false, error: 'disk full' }) });
    const user = await renderApp();
    await openPipeline(user);
    await user.upload(picker(), pdf('x.pdf'));
    await screen.findByText(T.addPdfsError.replace('{error}', 'disk full'));
  });

  it('reports a connection failure', async () => {
    installBackend({ 'POST /docs': () => { throw new Error('offline'); } });
    const user = await renderApp();
    await openPipeline(user);
    await user.upload(picker(), pdf('x.pdf'));
    await screen.findByText(T.addPdfsConnError);
  });
});
