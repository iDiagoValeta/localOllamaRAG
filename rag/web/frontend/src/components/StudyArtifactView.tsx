import { CheckCircle2, FileText } from '../lib/icons';
import type { StudyArtifact, StudyKind } from '../lib/types';
import type { Strings } from '../lib/i18n';
import { MarkdownContent } from './Markdown';
import { OutlineTree } from './OutlineTree';

export function StudyArtifactView({
  artifact, kind, revealed, onReveal, strings,
}: {
  artifact: StudyArtifact;
  kind: StudyKind;
  revealed: boolean;
  onReveal: () => void;
  strings: Strings;
}) {
  const pages = Array.from(new Set<number>([
    ...(artifact.sections ?? []).flatMap(x => x.source_pages ?? []),
    ...(artifact.questions ?? []).flatMap(x => x.source_pages ?? []),
  ])).sort((a, b) => a - b);

  return (
    <div className="w-full space-y-4 text-[15px] leading-relaxed">
      {kind === 'summary' && (artifact.sections ?? []).map((section, i) => (
        <div key={i} className="space-y-1">
          {section.heading && (
            <h3 className="text-[var(--text)] font-semibold">{section.heading}</h3>
          )}
          <div className="text-[var(--text-muted)]">
            <MarkdownContent text={section.body} />
          </div>
        </div>
      ))}

      {kind === 'outline' && <OutlineTree nodes={artifact.nodes ?? []} />}

      {kind === 'quiz' && (
        <div className="space-y-5">
          {(artifact.questions ?? []).map((q, i) => (
            <div key={i} className="space-y-2">
              <p className="text-[var(--text)] font-semibold">
                <span className="text-[var(--accent)] mr-1.5">{i + 1}.</span>{q.prompt}
              </p>
              <ul className="space-y-1.5">
                {q.options.map((option, j) => {
                  const isKey = revealed && j === q.correct_index;
                  return (
                    <li
                      key={j}
                      className={`flex items-start gap-2.5 px-3 py-1.5 border transition-colors ${
                        isKey
                          ? 'border-[var(--accent)]/40 bg-[var(--accent)]/10 text-[var(--text)]'
                          : 'border-transparent text-[var(--text-muted)]'
                      }`}
                    >
                      <span className={`font-mono text-xs mt-0.5 ${isKey ? 'text-[var(--accent)]' : 'text-[var(--text-faint)]'}`}>
                        {String.fromCharCode(97 + j)}
                      </span>
                      <span className="flex-1">{option}</span>
                      {isKey && (
                        <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-[var(--accent)]" />
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
          {/* Hidden by default: a quiz whose key is on screen from the first
              render is a summary with extra steps. */}
          {!revealed && (artifact.questions ?? []).length > 0 && (
            <button
              type="button"
              onClick={onReveal}
              className="px-3 py-1.5 text-[11px] font-bold tracking-wide uppercase text-[var(--text-muted)] hover:text-[var(--text)] bg-[var(--surface)] hover:bg-[var(--surface-2)] border border-[var(--border)] transition-colors"
            >
              {strings.studyShowAnswersInline}
            </button>
          )}
        </div>
      )}

      {artifact.source_document && (
        <div className="flex flex-wrap items-center gap-2 pt-1 text-xs text-[var(--text-faint)]">
          <span className="inline-flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5" />
            {artifact.source_document}
          </span>
          {pages.length > 0 && <span>· {pages.length} {strings.studyPagesShort}</span>}
        </div>
      )}
    </div>
  );
}

export default StudyArtifactView;
