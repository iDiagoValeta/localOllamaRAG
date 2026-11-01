import { describe, expect, it } from 'vitest';
import { screen } from '@testing-library/react';
import { installBackend } from './backend';
import { STRINGS } from '../src/lib/i18n';
import { mountApp, renderApp, T } from './render';

describe('page language', () => {
  it('defaults to Spanish when nothing was chosen, and sets html lang to it', async () => {
    installBackend();
    mountApp(null);
    // The static index.html says "en"; the page language has to follow the UI instead.
    await screen.findByPlaceholderText(STRINGS.es.placeholderRag);
    expect(document.documentElement.lang).toBe('es');
  });

  it('makes html lang follow the language toggle and remembers the choice', async () => {
    installBackend();
    const user = await renderApp('en');
    expect(document.documentElement.lang).toBe('en');

    await user.click(screen.getByRole('button', { name: 'VAL' }));
    expect(document.documentElement.lang).toBe('ca');
    expect(localStorage.getItem('monkeygrab_lang')).toBe('ca');

    await user.click(screen.getByRole('button', { name: 'ES' }));
    expect(document.documentElement.lang).toBe('es');
  });

  it('retranslates the visible text with the toggle', async () => {
    installBackend();
    const user = await renderApp('en');
    expect(screen.getByTitle(T.clearChat)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'ES' }));
    expect(screen.getByTitle('Limpiar chat')).toBeTruthy();
  });
});
