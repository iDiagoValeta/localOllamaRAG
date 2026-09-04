// Theme helper: dark by default, light optional, persisted in localStorage.
// Components read semantic CSS tokens (--color-surface, --color-ink, …)
// defined in index.css; toggling `data-theme` flips the whole palette.

const STORAGE_KEY = 'monkeygrab-theme';

export type Theme = 'dark' | 'light';

export function getStoredTheme(): Theme {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'light' ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute('data-theme', theme);
}

export function setTheme(theme: Theme): void {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* ignore quota/availability errors */
  }
  applyTheme(theme);
}

/** Apply the stored theme on boot (call before first paint) and return it. */
export function initTheme(): Theme {
  const theme = getStoredTheme();
  applyTheme(theme);
  return theme;
}

