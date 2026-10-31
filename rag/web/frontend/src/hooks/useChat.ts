import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';
import { streamSSE } from '../lib/stream';
import type { Strings } from '../lib/i18n';
import type { Lang, Message, Mode } from '../lib/types';
import { useStudy } from './useStudy';

interface Options {
  T: Strings;
  lang: Lang;
  mode: Mode;
  documents: string[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

/** Sending a turn (RAG, chat or study), streaming its answer, and the controls on the conversation. */
export function useChat({ T, lang, mode, documents, setMessages }: Options) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [revealedQuiz, setRevealedQuiz] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // One request at a time: a user stop and the inactivity timeout both end the
  // stream through this controller.
  const requestControllerRef = useRef<AbortController | null>(null);
  const activeAssistantRef = useRef<string | null>(null);

  const study = useStudy({
    T, lang, mode, input, setInput, documents, setMessages, setIsLoading, textareaRef,
  });
  const { sendStudy } = study;

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    if (mode === 'study') {
      await sendStudy(text);
      return;
    }

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      mode,
    };

    const assistantId = (Date.now() + 1).toString();
    const assistantMsg: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      mode,
      isStreaming: true,
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setInput('');
    setIsLoading(true);

    const controller = new AbortController();
    requestControllerRef.current = controller;
    activeAssistantRef.current = assistantId;

    const startTime = performance.now();

    try {
      const response = mode === 'rag'
        ? await api.rag(text, controller.signal)
        : await api.chat(text, controller.signal);

      await streamSSE(
        response,
        // onToken
        (token) => {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId
                ? { ...m, content: m.content + token }
                : m
            )
          );
        },
        // onDone
        (sources) => {
          const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId
                ? {
                    ...m,
                    isStreaming: false,
                    citations: sources || undefined,
                    metrics: mode === 'rag' ? {
                      searchTime: `${elapsed}s`,
                      // One chip per document, so the badge counts documents
                      // too: a single two-page PDF used to read "2 fuentes"
                      // next to its one chip (#377).
                      chunks: sources?.length || 0,
                    } : undefined,
                  }
                : m
            )
          );
        },
        // onError
        (errorMsg) => {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId
                ? { ...m, content: errorMsg, isStreaming: false, isError: true }
                : m
            )
          );
        },
        {
          unknown: T.unknownError, queryError: T.queryError, modelError: T.modelError,
          closed: T.connClosed, no_results: T.noResults, question_too_short: T.questionTooShort,
          model_unavailable: T.modelUnavailable, embedder_unavailable: T.embedderUnavailable,
          // _pipeline_error_payload's default code; unmapped, the UI showed the
          // backend's Spanish fallback in every language (issue #475).
          generation_error: T.modelError,
        },
        controller,
        // Inactivity timeout, not a connection error: the stream went silent
        // while the socket stayed open.
        () => {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId
                ? { ...m, content: T.streamStalled, isStreaming: false, isError: true }
                : m
            )
          );
        },
        // Shown before the first token, so a failed answer still says which
        // documents it was drawing on (issue #376).
        (sources) => {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId ? { ...m, citations: sources } : m
            )
          );
        },
      );
    } catch {
      // A stop or a stall already updated the bubble; overwriting it here
      // would replace the partial text with a connection error.
      if (!controller.signal.aborted) {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, content: T.connClosed, isStreaming: false, isError: true }
              : m
          )
        );
      }
    } finally {
      if (requestControllerRef.current === controller) requestControllerRef.current = null;
      if (activeAssistantRef.current === assistantId) activeAssistantRef.current = null;
      setIsLoading(false);
    }
  }, [input, mode, isLoading, lang, sendStudy]);

  // Keep whatever was already streamed: the user chose to stop, so the text
  // they have is the result, not a failure to report.
  const handleStop = useCallback(() => {
    requestControllerRef.current?.abort();
    setMessages(prev =>
      prev.map(m =>
        m.id === activeAssistantRef.current ? { ...m, isStreaming: false } : m
      )
    );
    setIsLoading(false);
  }, []);


  // ---- Copy message ----
  const handleCopyMessage = useCallback(async (msg: Message) => {
    if (!msg.content) return;
    try {
      await navigator.clipboard.writeText(msg.content);
      setCopiedId(msg.id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      /* fallback no soportado */
    }
  }, []);

  // ---- Clear history ----
  // Reset to an empty thread so the "Hola, <user>" greeting reappears; no
  // "history cleared" system message is shown.
  const handleClear = useCallback(async () => {
    await api.clear().catch(() => {});
    setMessages([]);
  }, []);

  // ---- Textarea auto-resize ----
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${Math.min(el.scrollHeight, 192)}px`;
    }
  }, [input]);


  const revealQuiz = useCallback((id: string) => {
    setRevealedQuiz(prev => ({ ...prev, [id]: true }));
  }, []);

  return {
    input, setInput, isLoading, textareaRef,
    revealedQuiz, revealQuiz, copiedId,
    handleSend, handleStop, handleCopyMessage, handleClear,
    ...study,
  };
}
