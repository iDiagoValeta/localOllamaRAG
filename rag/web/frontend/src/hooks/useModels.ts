import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { Strings } from '../lib/i18n';
import type { ModelRole, ModelRoles, OllamaModel } from '../lib/types';

interface Options {
  T: Strings;
  isInitialized: boolean;
  loadStores: () => Promise<void>;
  setFingerprintStale: (stale: boolean) => void;
}

/** The local Ollama server and which model plays each role. */
export function useModels({ T, isInitialized, loadStores, setFingerprintStale }: Options) {
  const [ollamaStatus, setOllamaStatus] = useState<{ running: boolean; version: string | null }>({ running: true, version: null });
  const [ollamaStarting, setOllamaStarting] = useState(false);
  // Why a start attempt left the server stopped. Kept apart from modelError
  // because it belongs to the Ollama row, not to the role selectors below it.
  const [ollamaError, setOllamaError] = useState<string | null>(null);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [modelRoles, setModelRoles] = useState<ModelRoles | null>(null);
  const [savingRole, setSavingRole] = useState<ModelRole | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);

  const loadModelRoles = useCallback(async () => {
    const res = await api.getModels().catch(() => null);
    if (res?.ok) setModelRoles(res.roles);
  }, []);

  const refreshOllama = useCallback(async () => {
    const st = await api.ollamaStatus().catch(() => null);
    if (st?.ok) {
      setOllamaStatus({ running: st.running, version: st.version });
      if (st.running) {
        // A status check that finds the server up retires any stale start
        // failure: the condition the message described no longer holds.
        setOllamaError(null);
        const m = await api.ollamaModels().catch(() => null);
        if (m?.ok) setOllamaModels(m.models || []);
      } else {
        setOllamaModels([]);
      }
    }
  }, []);

  // ---- Reassign a model role ----
  const handleRoleChange = useCallback(async (role: ModelRole, value: string) => {
    if (!modelRoles || value === modelRoles[role] || savingRole) return;
    const prev = modelRoles;
    setModelError(null);
    setSavingRole(role);
    setModelRoles({ ...modelRoles, [role]: value });
    try {
      const res = await api.setModels({ [role]: value } as Partial<ModelRoles>);
      if (!res.ok) {
        setModelRoles(prev);
        setModelError(res.error === 'indexing_in_progress' ? T.corpusConflict : T.modelSaveError);
        return;
      }
      setModelRoles(res.roles);
      // The contextual role enters index_recipe whenever contextual retrieval
      // is on (the default) -- reassigning it can make the active store
      // stale mid-session just like toggling the flag itself.
      setFingerprintStale(res.fingerprint_stale || false);
    } catch {
      setModelRoles(prev);
      setModelError(T.modelSaveError);
    } finally {
      setSavingRole(null);
    }
  }, [modelRoles, savingRole, T]);

  // ---- Start the local Ollama server ----
  const handleStartOllama = useCallback(async () => {
    if (ollamaStarting) return;
    setOllamaStarting(true);
    // A retry supersedes the failure of the attempt before it.
    setOllamaError(null);
    try {
      const res = await api.startOllama();
      setOllamaStatus({ running: !!res.running, version: res.version || null });
      if (res.running) {
        const m = await api.ollamaModels().catch(() => null);
        if (m?.ok) setOllamaModels(m.models || []);
      } else {
        // api.startOllama parses the body regardless of the HTTP status, so
        // the 404 that says the binary is missing arrives with its own code
        // and can name the cause instead of a generic failure.
        setOllamaError(res.error === 'ollama_not_found' ? T.ollamaNotFound : T.ollamaStartFailed);
      }
    } catch {
      // Transport failure: without this the row would keep saying "stopped"
      // as if nothing had been attempted.
      setOllamaError(T.ollamaStartFailed);
    } finally {
      setOllamaStarting(false);
    }
  }, [ollamaStarting, T]);

  // ---- Load control panel (stores / model roles / Ollama) once ready ----
  // The backend auto-starts Ollama at boot; if it is still down when the UI
  // loads, kick off a (server-blocking, idempotent) start so it comes up without
  // the user having to do it by hand.
  useEffect(() => {
    if (!isInitialized) return;
    loadStores();
    loadModelRoles();
    (async () => {
      const st = await api.ollamaStatus().catch(() => null);
      if (st?.ok && !st.running) {
        handleStartOllama();
      } else {
        refreshOllama();
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isInitialized, loadStores, loadModelRoles, refreshOllama]);

  return {
    ollamaStatus, ollamaStarting, ollamaError, ollamaModels,
    modelRoles, savingRole, modelError,
    refreshOllama, handleStartOllama, handleRoleChange,
  };
}
