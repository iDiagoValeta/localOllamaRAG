import { FileText, Loader2, Send, Square } from '../lib/icons';
import type { Strings } from '../lib/i18n';
import type { Mode, StudyKind } from '../lib/types';

interface ComposerProps {
  mode: Mode;
  input: string;
  setInput: (input: string) => void;
  isLoading: boolean;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  documentSuggestions: string[];
  activeSuggestion: number;
  setSuggestionIndex: React.Dispatch<React.SetStateAction<number>>;
  acceptSuggestion: (name: string) => void;
  studyKind: StudyKind;
  setStudyKind: (kind: StudyKind) => void;
  handleSend: () => void;
  handleStop: () => void;
  strings: Strings;
}

export function Composer({
  mode,
  input,
  setInput,
  isLoading,
  textareaRef,
  documentSuggestions,
  activeSuggestion,
  setSuggestionIndex,
  acceptSuggestion,
  studyKind,
  setStudyKind,
  handleSend,
  handleStop,
  strings: T,
}: ComposerProps) {
  return (
    <div
      className="p-6 absolute bottom-0 left-0 right-0 z-20"
      style={{ background: 'linear-gradient(to top, var(--color-surface), color-mix(in srgb, var(--color-surface) 88%, transparent) 55%, transparent)' }}
    >
      <div className="max-w-3xl mx-auto relative">
        {/* Autocomplete — above the input, so the caret never sits under it */}
        {mode === 'study' && documentSuggestions.length > 0 && (
          <div className="composer-panel absolute bottom-full mb-2 left-0 right-0 py-1.5 max-h-64 overflow-y-auto custom-scrollbar z-30">
            {documentSuggestions.map((doc, i) => (
              <button
                key={doc}
                type="button"
                onMouseDown={(e) => { e.preventDefault(); acceptSuggestion(doc); }}
                onMouseEnter={() => setSuggestionIndex(i)}
                className={`w-full flex items-center gap-2.5 px-4 py-2 text-left text-sm transition-colors ${
                  i === activeSuggestion
                    ? 'bg-surface-raised text-ink'
                    : 'text-ink-muted hover:text-ink'
                }`}
              >
                <FileText className={`w-4 h-4 shrink-0 ${i === activeSuggestion ? 'text-ink' : 'text-ink-faint'}`} />
                <span className="truncate">{doc}</span>
              </button>
            ))}
          </div>
        )}

        {/* The chooser lives INSIDE the input panel. Floating it above sat it
            on the transparent end of the area's gradient, so the chat read
            straight through it. Padding cannot fix that — content passes
            under a floating input by design; what it needed was a floor. */}
        <div className={`composer-panel relative focus-within:border-divider transition-all ${
          mode === 'study' ? 'flex flex-col p-2.5 gap-2' : 'flex items-end gap-3 p-2.5'
        }`}>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => { setInput(e.target.value); setSuggestionIndex(0); }}
            onKeyDown={(e) => {
              const suggesting = mode === 'study' && documentSuggestions.length > 0;
              if (suggesting && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
                e.preventDefault();
                setSuggestionIndex(i => {
                  const current = Math.min(i, documentSuggestions.length - 1);
                  const next = e.key === 'ArrowDown' ? current + 1 : current - 1;
                  return (next + documentSuggestions.length) % documentSuggestions.length;
                });
                return;
              }
              // Tab completes; Enter sends. Keeping them apart means a
              // half-typed name never becomes a request by reflex.
              if (suggesting && e.key === 'Tab') {
                e.preventDefault();
                acceptSuggestion(documentSuggestions[activeSuggestion]);
                return;
              }
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={mode === 'study' ? T.studyPlaceholder : mode === 'rag' ? T.placeholderRag : T.placeholderChat}
            className={`${mode === 'study' ? 'w-full max-h-40 min-h-[44px]' : 'flex-1 max-h-48 min-h-[52px]'} bg-transparent border-none focus:ring-0 focus:outline-none resize-none py-3.5 px-4 text-[15px] text-ink placeholder:text-ink-muted custom-scrollbar font-medium`}
            rows={1}
            disabled={isLoading}
          />

          {mode === 'study' ? (
            <div className="flex items-center justify-between gap-3">
              <div className="flex bg-field border border-edge rounded-lg p-0.5">
                {([
                  ['summary', T.studyKindSummary],
                  ['outline', T.studyKindOutline],
                  ['quiz', T.studyKindQuiz],
                ] as [StudyKind, string][]).map(([kind, label]) => (
                  <button
                    key={kind}
                    type="button"
                    aria-pressed={studyKind === kind}
                    onClick={() => setStudyKind(kind)}
                    className={`px-3 py-2 text-[11px] font-bold tracking-wide uppercase rounded transition-all focus:outline-none focus-visible:ring-1 focus-visible:ring-divider ${
                      studyKind === kind
                        ? 'bg-surface-raised text-ink'
                        : 'text-ink-muted hover:text-ink'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="p-3 bg-ink text-surface hover:opacity-90 active:scale-95 disabled:opacity-40 disabled:bg-surface-raised disabled:text-ink-faint transition-all flex-shrink-0 rounded-xl focus:outline-none focus-visible:ring-1 focus-visible:ring-divider"
              >
                {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5 ml-0.5" />}
              </button>
            </div>
          ) : isLoading ? (
            <button
              type="button"
              onClick={handleStop}
              aria-label={T.stopGenerating}
              title={T.stopGenerating}
              className="p-3.5 bg-ink text-surface hover:opacity-90 active:scale-95 transition-all flex-shrink-0 rounded-2xl focus:outline-none focus-visible:ring-1 focus-visible:ring-divider"
            >
              <Square className="w-5 h-5" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="p-3.5 bg-ink text-surface hover:opacity-90 active:scale-95 disabled:opacity-40 disabled:bg-surface-raised disabled:text-ink-faint transition-all flex-shrink-0 rounded-2xl focus:outline-none focus-visible:ring-1 focus-visible:ring-divider"
            >
              <Send className="w-5 h-5 ml-0.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
