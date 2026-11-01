import { motion, AnimatePresence } from 'motion/react';
import {
  AlertCircle, CheckCircle2, Check, Copy, FileText, Layers, Search,
} from '../lib/icons';
import { fill, type Strings } from '../lib/i18n';
import type { Lang, Message, Mode } from '../lib/types';
import { MarkdownContent } from './Markdown';
import { ShimmerText } from './ShimmerText';
import { StudyArtifactView } from './StudyArtifactView';

const GREETING: Record<Lang, string> = { es: 'Hola', en: 'Hi', ca: 'Hola' };

interface MessageListProps {
  // The scrolling pane itself: the auto-scroll hook owns its ref.
  containerRef: React.RefObject<HTMLDivElement | null>;
  mode: Mode;
  messages: Message[];
  isLoading: boolean;
  userName: string;
  lang: Lang;
  copiedId: string | null;
  handleCopyMessage: (msg: Message) => void;
  revealedQuiz: Record<string, boolean>;
  revealQuiz: (id: string) => void;
  openPdf: (doc: string, page: number, mode: 'split') => void;
  strings: Strings;
}

export function MessageList({
  containerRef,
  mode,
  messages,
  isLoading,
  userName,
  lang,
  copiedId,
  handleCopyMessage,
  revealedQuiz,
  revealQuiz,
  openPdf,
  strings: T,
}: MessageListProps) {
  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto p-6 md:p-8 custom-scrollbar scroll-smooth relative">
      <div className={`max-w-3xl mx-auto space-y-10 relative z-10 ${mode === 'study' ? 'pb-44' : 'pb-32'}`}>
        <AnimatePresence>
          {messages.length === 0 && !isLoading && (
            <motion.div
              key="greeting"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8, transition: { duration: 0.3 } }}
              className="pointer-events-none select-none absolute inset-x-0 top-[30vh] flex flex-col items-center justify-center px-6 text-center"
            >
              <h2 className="flex flex-wrap justify-center text-4xl md:text-5xl font-extrabold tracking-tight">
                <ShimmerText text={userName ? `${GREETING[lang]}, ${userName}` : GREETING[lang]} />
              </h2>
            </motion.div>
          )}
        </AnimatePresence>
        {messages.map((msg) => (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            key={msg.id}
            className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {/* System messages */}
            {msg.role === 'system' ? (
              <div className={`flex max-w-[85%] items-start gap-2 px-4 py-2.5 rounded-2xl text-xs font-medium ${msg.isError ? 'bg-danger/10 text-danger border border-danger/20' : 'bg-field text-ink-muted border border-edge'}`}>
                {msg.isError
                  ? <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-danger" />
                  : <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0 text-ok" />
                }
                <MarkdownContent text={msg.content} compact />
              </div>
            ) : (
              <>
                <div className={`flex flex-col gap-2 max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start w-full'}`}>
                  {/* Meta label + copy */}
                  <div className="flex items-center gap-2 px-2 group/meta">
                    {msg.role === 'assistant' && (
                      <span className="text-[9px] px-2 py-0.5 rounded-full uppercase tracking-widest font-bold bg-field text-ink-muted border border-edge">
                        {msg.mode}
                      </span>
                    )}
                    <button
                      onClick={() => handleCopyMessage(msg)}
                      className="p-2 rounded-full text-ink-muted hover:text-ink hover:bg-surface-raised focus-visible:text-ink transition-all"
                      title={T.copyMsg}
                    >
                      {copiedId === msg.id ? (
                        <Check className="w-3.5 h-3.5 text-ok" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>

                  {/* Message bubble */}
                  <div className={`text-[15px] leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-field text-ink border border-edge rounded-2xl px-4 py-3 font-medium text-left'
                      : msg.isError
                        ? 'text-danger'
                        : 'text-ink-soft'
                  }`}>
                    {msg.artifact && msg.artifactKind ? (
                      <StudyArtifactView
                        artifact={msg.artifact}
                        kind={msg.artifactKind}
                        revealed={!!revealedQuiz[msg.id]}
                        onReveal={() => revealQuiz(msg.id)}
                        strings={T}
                      />
                    ) : msg.content ? (
                      <MarkdownContent text={msg.content} />
                    ) : msg.isStreaming && msg.mode === 'study' ? (
                      // A study artifact is one generation over the whole
                      // document, and a long one legitimately takes minutes
                      // on a small model; say so instead of blinking.
                      <span className="inline-flex items-center gap-2 text-xs text-ink-muted">
                        <span className="inline-block w-2 h-5 bg-ink-muted rounded-sm animate-pulse" />
                        {T.studyWorking}
                      </span>
                    ) : msg.isStreaming ? (
                      <span className="inline-block w-2 h-5 bg-ink-muted rounded-sm animate-pulse" />
                    ) : null}
                    {msg.isStreaming && msg.content && (
                      <span className="inline-block w-2 h-5 bg-ink-muted rounded-sm animate-pulse ml-1 align-text-bottom" />
                    )}
                  </div>

                  {/* Citations */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="mt-3 w-full space-y-3 pl-2">
                      <div className="flex flex-wrap gap-2">
                        {msg.citations.map((cite, i) => (
                          <button
                            key={i}
                            className="inline-flex max-w-full items-center gap-2 px-3 py-1.5 rounded-full bg-field border border-edge text-xs text-ink-muted hover:bg-surface-raised hover:border-divider hover:text-ink transition-all group cursor-pointer"
                            onClick={() => openPdf(cite.document, cite.best_page ?? cite.pages[0] ?? 1, 'split')}
                            title={T.viewPdf}
                          >
                            <FileText className="w-3.5 h-3.5 text-ink-muted group-hover:text-ink" />
                            <span className="font-medium truncate min-w-0">{cite.document}</span>
                            <span className="text-ink-faint">|</span>
                            <span className="text-ink-muted shrink-0">p. {cite.best_page ?? cite.pages[0]}</span>
                          </button>
                        ))}
                      </div>
                      {msg.metrics && (
                        <div className="flex items-center gap-4 text-[11px] text-ink-muted font-mono bg-field inline-flex px-3 py-1.5 rounded-full border border-edge">
                          <span className="flex items-center gap-1.5"><Search className="w-3.5 h-3.5 text-ink-faint" /> {msg.metrics.searchTime}</span>
                          <span className="w-1 h-1 rounded-full bg-divider"></span>
                          <span className="flex items-center gap-1.5"><Layers className="w-3.5 h-3.5 text-ink-faint" /> {fill(T.sources, { n: msg.metrics.chunks })}</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
