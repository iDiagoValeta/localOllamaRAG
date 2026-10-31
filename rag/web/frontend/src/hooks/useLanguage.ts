import { useCallback, useEffect, useState } from 'react';
import { normalizeLang, STRINGS } from '../lib/i18n';
import type { Lang } from '../lib/types';

/** The UI language, its string table, and the page language that follows it. */
export function useLanguage() {
  const [lang, setLangState] = useState<Lang>(() =>
    normalizeLang(localStorage.getItem('monkeygrab_lang'))
  );
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    localStorage.setItem('monkeygrab_lang', l);
  }, []);

  // The page language drives how a screen reader pronounces the text; the
  // static index.html says "en" whatever the UI shows (issue #555).
  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  return { lang, setLang, T: STRINGS[lang] };
}
