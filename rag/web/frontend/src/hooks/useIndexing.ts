import { useCallback, useState } from 'react';
import type { IndexingProgress } from '../lib/types';

/** The state of a running or failed index, which replaces the whole UI with a blocking screen. */
export function useIndexing() {
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexingProgress, setIndexingProgress] = useState<IndexingProgress | null>(null);
  const [indexingError, setIndexingError] = useState<string | null>(null);
  // Bumping it re-runs the init poller, which is what follows a run to its end.
  const [retryTrigger, setRetryTrigger] = useState(0);

  /** A run just started elsewhere (reindex, store switch): show it and start polling. */
  const startIndexing = useCallback((progress: IndexingProgress | null) => {
    setIndexingError(null);
    setIndexingProgress(progress);
    setIsIndexing(true);
    setRetryTrigger(t => t + 1);
  }, []);

  return {
    isIndexing, setIsIndexing,
    indexingProgress, setIndexingProgress,
    indexingError, setIndexingError,
    retryTrigger, setRetryTrigger,
    startIndexing,
  };
}

export type Indexing = ReturnType<typeof useIndexing>;
