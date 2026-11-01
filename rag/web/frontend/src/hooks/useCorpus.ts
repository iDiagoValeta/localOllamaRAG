import { useCallback, useState } from 'react';
import { api } from '../lib/api';
import type { VectorStore } from '../lib/types';

/** What the active store holds: its documents, the store rows and whether the index is stale. */
export function useCorpus() {
  const [documents, setDocuments] = useState<string[]>([]);
  // True when the active store's recorded index fingerprint disagrees with
  // the configuration in force (issue #36). Detection only -- the backend
  // never reindexes automatically on this; the user re-indexes explicitly.
  const [fingerprintStale, setFingerprintStale] = useState(false);
  const [stores, setStores] = useState<VectorStore[]>([]);
  const [activeStore, setActiveStore] = useState<string>('en');

  const loadStores = useCallback(async () => {
    const res = await api.listStores().catch(() => null);
    if (res?.ok) {
      setStores(res.stores || []);
      setActiveStore(res.active || 'en');
    }
  }, []);

  return {
    documents, setDocuments,
    fingerprintStale, setFingerprintStale,
    stores, setStores,
    activeStore, setActiveStore,
    loadStores,
  };
}

export type Corpus = ReturnType<typeof useCorpus>;
