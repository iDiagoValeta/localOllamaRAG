import { Database, Menu, Moon, Ollama, Sun, Trash2, BookOpen } from '../lib/icons';
import type { Strings } from '../lib/i18n';
import type { Lang, Mode } from '../lib/types';
import type { Theme } from '../lib/theme';
import { LanguageToggle } from './LanguageToggle';

interface ChatHeaderProps {
  isSidebarOpen: boolean;
  setIsSidebarOpen: (open: boolean) => void;
  mode: Mode;
  handleModeChange: (mode: Mode) => void;
  theme: Theme;
  toggleTheme: () => void;
  lang: Lang;
  setLang: (lang: Lang) => void;
  handleClear: () => void;
  strings: Strings;
}

export function ChatHeader({
  isSidebarOpen,
  setIsSidebarOpen,
  mode,
  handleModeChange,
  theme,
  toggleTheme,
  lang,
  setLang,
  handleClear,
  strings: T,
}: ChatHeaderProps) {
  return (
    <header className="h-20 border-b border-divider flex items-center justify-between gap-3 px-4 bg-surface-raised z-10">
      <div className="flex items-center gap-3 min-w-0">
        <button
          className="p-2.5 text-ink-muted hover:text-ink bg-field hover:bg-surface border border-edge rounded-lg transition-colors flex-shrink-0 focus:outline-none focus-visible:ring-1 focus-visible:ring-divider"
          aria-label={isSidebarOpen ? T.closeSidebar : T.openSidebar}
          aria-expanded={isSidebarOpen}
          aria-controls="app-sidebar"
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex bg-field p-1 border border-edge rounded-xl flex-shrink-0">
          <button
            aria-pressed={mode === 'chat'}
            className={`min-w-[84px] justify-center px-4 py-2 text-xs font-bold tracking-wide rounded-lg transition-all flex items-center gap-2 focus:outline-none focus-visible:ring-1 focus-visible:ring-divider ${mode === 'chat' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
            onClick={() => handleModeChange('chat')}
          >
            <Ollama className="w-4 h-4 text-ink" />
            CHAT
          </button>
          <button
            aria-pressed={mode === 'rag'}
            className={`min-w-[84px] justify-center px-4 py-2 text-xs font-bold tracking-wide rounded-lg transition-all flex items-center gap-2 focus:outline-none focus-visible:ring-1 focus-visible:ring-divider ${mode === 'rag' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
            onClick={() => handleModeChange('rag')}
          >
            <Database className="w-4 h-4" />
            RAG
          </button>
          <button
            aria-pressed={mode === 'study'}
            className={`min-w-[84px] justify-center px-4 py-2 text-xs font-bold tracking-wide rounded-lg transition-all flex items-center gap-2 focus:outline-none focus-visible:ring-1 focus-visible:ring-divider ${mode === 'study' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
            onClick={() => handleModeChange('study')}
          >
            <BookOpen className="w-4 h-4" />
            {T.modeStudy}
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        <button
          type="button"
          onClick={toggleTheme}
          aria-pressed={theme === 'dark'}
          className="p-2 text-ink-muted hover:text-ink bg-field hover:bg-surface border border-edge rounded-lg transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-divider"
          title={theme === 'dark' ? T.themeLight : T.themeDark}
          aria-label={theme === 'dark' ? T.themeLight : T.themeDark}
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
        <LanguageToggle lang={lang} setLang={setLang} />
        <button
          className="p-2 text-ink-muted hover:text-ink bg-field hover:bg-surface border border-edge rounded-lg transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-divider"
          onClick={handleClear}
          title={T.clearChat}
          aria-label={T.clearChat}
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
