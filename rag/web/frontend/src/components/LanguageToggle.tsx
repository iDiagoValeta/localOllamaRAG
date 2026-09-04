import { Languages } from '../lib/icons';
import { LANG_OPTIONS } from '../lib/i18n';
import type { Lang } from '../lib/types';

export function LanguageToggle({ lang, setLang }: { lang: Lang; setLang: (lang: Lang) => void }) {
  return (
    <div className="flex items-center gap-1 border border-edge bg-field rounded-lg p-1">
      <Languages className="ml-1.5 h-3.5 w-3.5 text-ink-faint" />
      {LANG_OPTIONS.map(option => (
        <button
          key={option.code}
          type="button"
          className={`w-9 text-center px-1 py-1 text-[10px] font-bold tracking-wide rounded transition-all ${
            lang === option.code
              ? 'bg-surface-raised text-ink'
              : 'text-ink-muted hover:text-ink'
          }`}
          onClick={() => setLang(option.code)}
          aria-pressed={lang === option.code}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export default LanguageToggle;
