import { useCallback, useState } from 'react';
import { getStoredTheme, setTheme, type Theme } from '../lib/theme';

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getStoredTheme);
  const toggleTheme = useCallback(() => {
    setThemeState(prev => {
      const next: Theme = prev === 'dark' ? 'light' : 'dark';
      setTheme(next);
      return next;
    });
  }, []);
  return { theme, toggleTheme };
}
