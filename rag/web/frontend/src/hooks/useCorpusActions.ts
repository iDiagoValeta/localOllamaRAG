import { useCallback, useRef, useState } from 'react';
import { api, apiErrorText } from '../lib/api';
import { fill, type Strings } from '../lib/i18n';
import type { Message, Mode } from '../lib/types';
import type { Corpus } from './useCorpus';
import type { Indexing } from './useIndexing';
import type { PdfViewerState } from './usePanes';

interface Options {
  T: Strings;
  mode: Mode;
  // A reindex swaps the chat for the indexing screen, which would discard an
  // answer still streaming (issue #404), so these actions wait for it.
  isLoading: boolean;
  corpus: Corpus;
  indexing: Indexing;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  setPdfViewer: (viewer: PdfViewerState | null) => void;
  setSettingsError: (error: string | null) => void;
}

/** Everything the user does to the corpus: switch store, add, delete, reindex. */
export function useCorpusActions({
  T, mode, isLoading, corpus, indexing, setMessages, setPdfViewer, setSettingsError,
}: Options) {
  const { activeStore, setActiveStore, setStores, setDocuments, setFingerprintStale } = corpus;
  const [storeBusy, setStoreBusy] = useState(false);
  const [storeError, setStoreError] = useState<string | null>(null);
  const [isReindexing, setIsReindexing] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [isAddingDocs, setIsAddingDocs] = useState(false);
  const reindexFileInputRef = useRef<HTMLInputElement>(null);

  // ---- Switch active vector store ----
  const handleStoreSelect = useCallback(async (name: string) => {
    if (name === activeStore || storeBusy) return;
    const prev = activeStore;
    setStoreError(null);
    setActiveStore(name);
    setStoreBusy(true);
    try {
      const res = await api.selectStore(name);
      if (!res.ok) {
        setActiveStore(prev);
        setStoreError(res.error === 'indexing_in_progress' ? T.corpusConflict : T.corpusError);
        return;
      }
      setStores(res.stores || []);
      // Citations carry only a document name and /api/pdf resolves it in the
      // active store, so after a switch they would open the wrong file or 404
      // (issue #405). The conversation belongs to the store it was asked of.
      setMessages([]);
      setPdfViewer(null);
      if (res.indexing) {
        indexing.startIndexing(null);
        return;
      }
      setDocuments(res.documents || []);
      setFingerprintStale(res.fingerprint_stale || false);
    } catch {
      setActiveStore(prev);
      setStoreError(T.corpusConnError);
    } finally {
      setStoreBusy(false);
    }
  }, [activeStore, storeBusy, T]);

  // ---- Reindex (full, with current pipeline settings) ----
  // Hands off to the full-screen blocking indexing UI (no chat system messages);
  // the init poller refreshes the document list + fragment count on completion.
  const handleReindex = useCallback(async () => {
    // The indexing screen replaces the chat and drops the answer in flight
    // (issue #404); the button is disabled too, this guards any other caller.
    if (isReindexing || isLoading) return;
    setIsReindexing(true);
    setSettingsError(null);
    try {
      const result = await api.reindex();
      if (result.ok && result.indexing) {
        indexing.startIndexing(result.progress || null);
      } else if (result.ok) {
        setDocuments(result.documents || []);
      } else {
        setSettingsError(fill(T.reindexError, { error: apiErrorText(T, result, T.unknownError) }));
      }
    } catch {
      setSettingsError(T.reindexConnError);
    } finally {
      setIsReindexing(false);
    }
  }, [isReindexing, isLoading, T]);

  // ---- Add documents (indexes only the new files) ----
  const handleAddDocs = useCallback(async (files: File[]) => {
    if (isAddingDocs || isReindexing || isLoading || !files.length) {
      // The picker keeps the chosen files otherwise, and picking the same ones
      // again would fire no change event.
      if (reindexFileInputRef.current) reindexFileInputRef.current.value = '';
      return;
    }
    setIsAddingDocs(true);
    setSettingsError(null);
    try {
      const result = await api.addDocs(files);
      // A partial failure still changed the store: refresh the list either way.
      if (Array.isArray(result.documents)) setDocuments(result.documents);
      if (!result.ok) {
        const failed: string[] = Array.isArray(result.failed_files) ? result.failed_files : [];
        setSettingsError(
          failed.length
            ? fill(T.addPdfsFailed, { files: failed.join(', ') })
            : fill(T.addPdfsError, { error: apiErrorText(T, result, T.unknownError) }),
        );
      }
    } catch {
      setSettingsError(T.addPdfsConnError);
    } finally {
      setIsAddingDocs(false);
      if (reindexFileInputRef.current) reindexFileInputRef.current.value = '';
    }
  }, [isAddingDocs, isReindexing, isLoading, T]);

  // ---- Delete document ----
  const handleDeleteDoc = useCallback(async (docName: string) => {
    if (deletingDoc) return;
    if (!window.confirm(fill(T.confirmDelete, { name: docName }))) return;
    setDeletingDoc(docName);
    try {
      const result = await api.deleteDoc(docName);
      if (result.ok) {
        setDocuments(result.documents || []);
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: fill(T.docDeleted, { name: docName }),
          mode,
        }]);
      } else {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: fill(T.deleteError, { error: apiErrorText(T, result, T.unknownError) }),
          mode,
          isError: true,
        }]);
      }
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: T.deleteConnError,
        mode,
        isError: true,
      }]);
    } finally {
      setDeletingDoc(null);
    }
  }, [mode, deletingDoc, T]);

  return {
    storeBusy, storeError, isReindexing, deletingDoc, isAddingDocs, reindexFileInputRef,
    handleStoreSelect, handleReindex, handleAddDocs, handleDeleteDoc,
  };
}
