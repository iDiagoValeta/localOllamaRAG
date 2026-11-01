import { render, screen } from '@testing-library/react';
import userEvent, { type UserEvent } from '@testing-library/user-event';
import App from '../src/App';
import { STRINGS } from '../src/lib/i18n';
import type { Lang } from '../src/lib/types';

/** The English table: every test runs the UI in English unless it says otherwise. */
export const T = STRINGS.en;

/** Mounts the app without waiting for anything: for the screens before the main one. */
export function mountApp(
  lang: Lang | null = 'en',
  options: Parameters<typeof userEvent.setup>[0] = {},
): UserEvent {
  if (lang) localStorage.setItem('monkeygrab_lang', lang);
  const user = userEvent.setup(options);
  render(<App />);
  return user;
}

/** Mounts the app and waits until the main screen (not the loading one) is up. */
export async function renderApp(
  lang: Lang = 'en',
  options: Parameters<typeof userEvent.setup>[0] = {},
): Promise<UserEvent> {
  const user = mountApp(lang, options);
  await screen.findByPlaceholderText(STRINGS[lang].placeholderRag);
  return user;
}
