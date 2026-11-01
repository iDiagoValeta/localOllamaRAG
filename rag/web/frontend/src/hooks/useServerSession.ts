import { useCallback, useEffect, useRef, useState } from 'react';
import { api, apiErrorText } from '../lib/api';
import type { Strings } from '../lib/i18n';
import type { Message, Mode, PipelineSettings } from '../lib/types';
import type { Corpus } from './useCorpus';
import type { Indexing } from './useIndexing';

interface Options {
  T: Strings;
  corpus: Corpus;
  indexing: Indexing;
  setMode: (mode: Mode) => void;
  setMessages: (messages: Message[]) => void;
  setSettings: React.Dispatch<React.SetStateAction<PipelineSettings>>;
}

/** Talking to the backend on load: the init poller, and the retry buttons of the screens that block the UI. */
export function useServerSession({ T, corpus, indexing, setMode, setMessages, setSettings }: Options) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [userName, setUserName] = useState('');
  const historyHydratedRef = useRef(false);
  const { setDocuments, setFingerprintStale, setActiveStore, loadStores } = corpus;
  const {
    retryTrigger, setRetryTrigger, setIsIndexing,
    setIndexingProgress, setIndexingError,
  } = indexing;

  // ---- Initialize on mount ----
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const initData = await api.init();
        if (cancelled) return;

        if (initData.ok) {
          setInitError(null);
          setIndexingError(null);
          setIsIndexing(false);
          setIndexingProgress(null);
          setMode(initData.mode || 'rag');
          setDocuments(initData.documents || []);
          setFingerprintStale(initData.fingerprint_stale || false);
          setActiveStore(initData.active_store || 'en');
          setUserName(initData.user || '');
          setIsInitialized(true);
          // The control-panel effect below only runs when isInitialized flips
          // false→true. A reindex starts from an initialized app and sets
          // isIndexing instead, so this success branch is the one that has to
          // refresh the store rows out of "not indexed".
          loadStores();

          // The backend reloads historial_chat.json on every init and builds
          // the next prompt from it; paint those turns or the model answers
          // with context the user cannot see (issue #403). Only on the page's
          // first init: init runs again after a reindex or a store switch,
          // and repainting there wiped the visible RAG answers and brought
          // back turns a store switch had cleared (issue #473).
          if (!historyHydratedRef.current) {
            historyHydratedRef.current = true;
            setMessages(
              Array.isArray(initData.history)
                ? initData.history
                    .map((entry: { role: 'user' | 'assistant'; content: string }, i: number) => ({
                      id: `hist-${i}`,
                      role: entry.role,
                      content: entry.content,
                      mode: 'chat' as const,
                    }))
                : []
            );
          }
        } else if (initData.indexing) {
          // A run is still alive: keep the blocking screen and poll.
          setInitError(null);
          setIndexingError(null);
          setIsIndexing(true);
          if (initData.progress) setIndexingProgress(initData.progress);
          setTimeout(() => { if (!cancelled) init(); }, 2000);
        } else {
          // Permanent failure (issue #379). /api/init answers ok: false
          // without `indexing` only when the last run failed on every file,
          // and _ensure_indexed refuses to restart it -- polling from here
          // is the infinite spinner, so report the message and stop.
          setIsIndexing(false);
          setIndexingProgress(null);
          setIndexingError(apiErrorText(T, initData, T.indexingFailed));
        }

        try {
          const settingsData = await api.getSettings();
          if (settingsData.ok) setSettings(prev => ({ ...prev, ...settingsData.settings }));
        } catch {
          /* ignorar fallo de settings */
        }
      } catch (err) {
        if (!cancelled) setInitError(T.noServer);
      }
    }
    init();
    return () => { cancelled = true; };
  }, [retryTrigger]);

  // ---- Connection error screen ----
  const handleRetry = useCallback(async () => {
    setInitError(null);
    setIsIndexing(true);
    setIndexingError(null);
    try {
      const result = await api.init();
      if (result.ok) {
        setMode(result.mode || 'rag');
        setDocuments(result.documents || []);
        setFingerprintStale(result.fingerprint_stale || false);
        setActiveStore(result.active_store || 'en');
        setIndexingError(null);
        setIndexingProgress(null);
        setIsInitialized(true);
        setIsIndexing(false);
      } else if (result.indexing) {
        setIndexingProgress(result.progress || null);
        setRetryTrigger(t => t + 1);
      } else {
        setIsIndexing(false);
        setIndexingError(apiErrorText(T, result, T.indexingFailed));
      }
    } catch {
      setIsIndexing(false);
      setInitError(T.noServer);
    }
  }, [T]);

  // Retrying a failed index must start a new run: /api/init keeps reporting
  // the failure because _ensure_indexed refuses to restart after one, so only
  // /api/reindex clears the state and launches the worker (issue #379).
  const handleRetryIndexing = useCallback(async () => {
    setIsIndexing(true);
    setIndexingError(null);
    try {
      const result = await api.reindex();
      if (result.ok && result.indexing) {
        setIndexingProgress(result.progress || null);
        setRetryTrigger(t => t + 1);
      } else {
        setIsIndexing(false);
        setIndexingError(apiErrorText(T, result, T.indexingFailed));
      }
    } catch {
      setIsIndexing(false);
      setIndexingError(T.noServer);
    }
  }, [T]);

  return { isInitialized, initError, userName, handleRetry, handleRetryIndexing };
}
