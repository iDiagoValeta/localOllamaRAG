import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send, FileText, Database, Ollama,
  Search, Layers, Menu, X,
  RefreshCw, Loader2, AlertCircle, CheckCircle2, Trash2,
  Copy, Check, Eye,
  Sun, Moon, BookOpen
} from './lib/icons';
import { getStoredTheme, setTheme, type Theme } from './lib/theme';
import { motion, AnimatePresence } from 'motion/react';
import type {
  Citation,
  IndexingProgress,
  Lang,
  Message,
  Mode,
  ModelRole,
  ModelRoles,
  OllamaModel,
  PipelineSettings,
  StudyArtifact,
  StudyKind,
  VectorStore,
} from './lib/types';
import { fill, normalizeLang, STRINGS } from './lib/i18n';
import { MarkdownContent } from './components/Markdown';
import { StudyArtifactView } from './components/StudyArtifactView';
import { ShimmerText } from './components/ShimmerText';
import { PdfPane } from './components/PdfPane';
import { LanguageToggle } from './components/LanguageToggle';
import { ConnectionErrorScreen, IndexingScreen, LoadingScreen } from './components/StatusScreens';
import { SettingsOverlay } from './components/SettingsOverlay';

// =============================================================================
// API Service — connects to Flask backend
// =============================================================================

const API_BASE = '/api';

const api = {
  init: () =>
    fetch(`${API_BASE}/init`).then(r => r.json()),

  docs: () =>
    fetch(`${API_BASE}/docs`).then(r => r.json()),

  stats: () =>
    fetch(`${API_BASE}/stats`).then(r => r.json()),

  topics: () =>
    fetch(`${API_BASE}/topics`).then(r => r.json()),

  setMode: (mode: Mode) =>
    fetch(`${API_BASE}/mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    }).then(r => r.json()),

  chat: (message: string) =>
    fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, stream: true }),
    }),

  rag: (message: string) =>
    fetch(`${API_BASE}/rag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, stream: true }),
    }),

  clear: () =>
    fetch(`${API_BASE}/clear`, { method: 'POST' }).then(r => r.json()),

  listStores: () =>
    fetch(`${API_BASE}/stores`).then(r => r.json()),

  selectStore: (name: string) =>
    fetch(`${API_BASE}/stores/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).then(r => r.json()),

  ollamaStatus: () =>
    fetch(`${API_BASE}/ollama`).then(r => r.json()),

  startOllama: () =>
    fetch(`${API_BASE}/ollama/start`, { method: 'POST' }).then(r => r.json()),

  ollamaModels: () =>
    fetch(`${API_BASE}/ollama/models`).then(r => r.json()),

  getModels: () =>
    fetch(`${API_BASE}/models`).then(r => r.json()),

  setModels: (roles: Partial<ModelRoles>) =>
    fetch(`${API_BASE}/models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(roles),
    }).then(r => r.json()),

  reindex: (files?: File[]) => {
    const form = new FormData();
    files?.forEach(f => form.append('file', f));
    return fetch(`${API_BASE}/reindex`, { method: 'POST', body: form }).then(r => r.json());
  },

  getSettings: () =>
    fetch(`${API_BASE}/settings`).then(r => r.json()),

  updateSettings: (settings: Partial<PipelineSettings>) =>
    fetch(`${API_BASE}/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    }).then(r => r.json()),

  deleteDoc: (filename: string) =>
    fetch(`${API_BASE}/docs/${encodeURIComponent(filename)}`, { method: 'DELETE' }).then(r => r.json()),

  // One call for the three artifacts, mirroring the single route. The status
  // code is kept: 422 means the model ignored the format, which the panel
  // reports differently from a server failure because the user's next move
  // differs too (retry or change model, versus check the backend).
  study: (kind: StudyKind, document: string, language: string, questionCount?: number) =>
    fetch(`${API_BASE}/study`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, document, language, question_count: questionCount }),
    }).then(async r => ({ status: r.status, body: await r.json() })),
};

// =============================================================================
// SSE Stream Parser
// =============================================================================

async function streamSSE(
  response: Response,
  onToken: (token: string) => void,
  onDone: (sources: Citation[] | null) => void,
  onError: (msg: string) => void,
  fb?: { unknown?: string; queryError?: string; modelError?: string; closed?: string; no_results?: string },
) {
  const contentType = response.headers.get('content-type') || '';

  // If the response is NOT a stream (error responses come as JSON)
  if (!contentType.includes('event-stream')) {
    const data = await response.json().catch(() => ({ message: fb?.unknown ?? 'Error desconocido' }));
    if (!data.ok && data.message) {
      const mapped = data.error && fb?.[data.error as keyof typeof fb];
      onError(typeof mapped === 'string' ? mapped : data.message);
    } else if (!data.ok && data.error) {
      onError(typeof data.error === 'string' ? data.error : (fb?.queryError ?? 'Error en la consulta'));
    } else {
      onDone(null);
    }
    return;
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finished = false;

  const processBlock = (block: string) => {
    if (!block.trim()) return;

    let event = 'message';
    const dataLines: string[] = [];

    for (const rawLine of block.split(/\r?\n/)) {
      const line = rawLine.trimEnd();
      if (!line || line.startsWith(':')) continue;
      if (line.startsWith('event:')) {
        event = line.slice(6).trim() || 'message';
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart());
      }
    }

    if (!dataLines.length) return;

    try {
      const data = JSON.parse(dataLines.join('\n'));
      if (event === 'error' || data.error) {
        finished = true;
        onError(data.message || data.error || fb?.modelError || 'Error en la respuesta del modelo');
        return;
      }
      if (event === 'token' || data.token) onToken(data.token || '');
      if (event === 'done' || data.done) {
        finished = true;
        onDone(data.sources || null);
      }
    } catch {
      // Ignore malformed SSE blocks; the backend always sends JSON.
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || '';

    for (const block of blocks) processBlock(block);
  }

  buffer += decoder.decode();
  if (buffer.trim()) processBlock(buffer);
  if (!finished) onError(fb?.closed ?? 'La conexión se cerró antes de completar la respuesta.');
}

// =============================================================================
// App Component
// =============================================================================

const GREETING: Record<Lang, string> = { es: 'Hola', en: 'Hi', ca: 'Hola' };

export default function App() {
  const [lang, setLangState] = useState<Lang>(() =>
    normalizeLang(localStorage.getItem('monkeygrab_lang'))
  );
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    localStorage.setItem('monkeygrab_lang', l);
  }, []);
  const T = STRINGS[lang];

  const [mode, setMode] = useState<Mode>('rag');
  const [input, setInput] = useState('');
  // Study mode. The artifact kind is a control (three fixed choices); the
  // document is typed, because a corpus grows and a dropdown of every PDF
  // stops being a control and becomes a list to scroll.
  const [studyKind, setStudyKind] = useState<StudyKind>('summary');
  const [suggestionIndex, setSuggestionIndex] = useState(0);
  const [revealedQuiz, setRevealedQuiz] = useState<Record<string, boolean>>({});
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<'docs' | 'models' | 'settings'>('docs');
  const [documents, setDocuments] = useState<string[]>([]);
  const [totalFragments, setTotalFragments] = useState(0);
  // True when the active store's recorded index fingerprint disagrees with
  // the configuration in force (issue #36). Detection only -- the backend
  // never reindexes automatically on this; the user re-indexes explicitly.
  const [fingerprintStale, setFingerprintStale] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexingProgress, setIndexingProgress] = useState<IndexingProgress | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [isReindexing, setIsReindexing] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [pendingReindexFiles, setPendingReindexFiles] = useState<File[]>([]);
  const [theme, setThemeState] = useState<Theme>(getStoredTheme);
  const toggleTheme = useCallback(() => {
    setThemeState(prev => {
      const next: Theme = prev === 'dark' ? 'light' : 'dark';
      setTheme(next);
      return next;
    });
  }, []);
  const [stores, setStores] = useState<VectorStore[]>([]);
  const [activeStore, setActiveStore] = useState<string>('en');
  const [storeBusy, setStoreBusy] = useState(false);
  const [storeError, setStoreError] = useState<string | null>(null);
  const [ollamaStatus, setOllamaStatus] = useState<{ running: boolean; version: string | null }>({ running: true, version: null });
  const [ollamaStarting, setOllamaStarting] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [modelRoles, setModelRoles] = useState<ModelRoles | null>(null);
  const [savingRole, setSavingRole] = useState<ModelRole | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [settings, setSettings] = useState<PipelineSettings>({
    contextualRetrieval: true,
    queryDecomposition: true,
    hybridSearch: true,
    imageIndexing: true,
    imageDescription: false,
    reranker: true,
    expandContext: true,
    optimizeContext: true,
    recompSynthesis: true,
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const reindexFileInputRef = useRef<HTMLInputElement>(null);
  const [retryTrigger, setRetryTrigger] = useState(0);
  const [indexingError, setIndexingError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [pdfViewer, setPdfViewer] = useState<{ doc: string; page: number; mode: 'full' | 'split' } | null>(null);
  // Full-area overlay panels (Models / Pipeline) shown in the main column, like
  // the PDF viewer. Opening one closes any open PDF; closing returns to chat.
  const [mainPanel, setMainPanel] = useState<'models' | 'pipeline' | null>(null);
  const openMainPanel = useCallback((panel: 'models' | 'pipeline') => {
    setPdfViewer(null);
    setMainPanel(panel);
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  }, []);
  const [userName, setUserName] = useState('');

  // Remembers whether the sidebar was open before a split view collapsed it.
  const sidebarBeforePdfRef = useRef(true);

  // mode 'full' = viewer replaces the chat (opened from the sidebar);
  // mode 'split' = viewer sits left of the chat (opened from a source citation).
  // Split needs the width, so it collapses the sidebar (docs / pipeline) and the
  // close handler restores it if it was open.
  const openPdf = useCallback((doc: string, page = 1, mode: 'full' | 'split' = 'full') => {
    if (mode === 'split') {
      sidebarBeforePdfRef.current = isSidebarOpen;
      setIsSidebarOpen(false);
    }
    setPdfViewer({ doc, page, mode });
  }, [isSidebarOpen]);

  const closePdf = useCallback(() => {
    if (pdfViewer?.mode === 'split' && sidebarBeforePdfRef.current) setIsSidebarOpen(true);
    setPdfViewer(null);
  }, [pdfViewer]);

  // ---- Scroll to bottom ----
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

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
          setTotalFragments(initData.total_fragments || 0);
          setFingerprintStale(initData.fingerprint_stale || false);
          setActiveStore(initData.active_store || 'en');
          setUserName(initData.user || '');
          setIsInitialized(true);

          setMessages([]);
        } else {
          // ok: false = indexando o error previo → siempre pantalla de indexación (nunca "Error de conexión")
          setInitError(null);
          setIsIndexing(true);
          setIndexingError(initData.error || null);
          if (initData.progress) setIndexingProgress(initData.progress);
          setTimeout(() => { if (!cancelled) init(); }, 2000);
        }

        try {
          const settingsData = await api.getSettings();
          if (settingsData.ok) setSettings(prev => ({ ...prev, ...settingsData.settings }));
        } catch {
          /* ignorar fallo de settings */
        }
      } catch (err) {
        if (!cancelled) setInitError(STRINGS[lang].noServer);
      }
    }
    init();
    return () => { cancelled = true; };
  }, [retryTrigger]);

  // ---- Mode switching ----
  const handleModeChange = useCallback(async (newMode: Mode) => {
    const previousMode = mode;
    setMode(newMode);
    // The backend only knows chat and rag. A study turn is a POST to
    // /api/study, so telling the server about this mode would either 400 or
    // leave it holding a mode nothing reads.
    if (newMode === 'study') return;
    const result = await api.setMode(newMode).catch(() => null);
    if (!result?.ok) setMode(previousMode);
  }, [mode]);

  // ---- Pipeline settings toggle ----
  const toggleSetting = useCallback(async (key: keyof PipelineSettings) => {
    const previousVal = settings[key];
    const newVal = !previousVal;
    setSettingsError(null);
    setSettings(prev => ({ ...prev, [key]: newVal }));
    const result = await api.updateSettings({ [key]: newVal }).catch(() => null);
    if (result?.ok && key in result.settings) {
      // Server may override (e.g. reranker unavailable)
      setSettings(prev => ({ ...prev, [key]: result.settings[key] }));
      // contextualRetrieval / imageIndexing are index-time flags: flipping
      // either can make the active store stale right here, mid-session, with
      // no restart or store switch to otherwise trigger a fresh check.
      setFingerprintStale(result.fingerprint_stale || false);
    } else {
      setSettings(prev => ({ ...prev, [key]: previousVal }));
      setSettingsError(result?.error || T.settingsSaveError);
    }
  }, [settings, lang]);

  // ---- Reindex (full, with current pipeline settings) ----
  // Hands off to the full-screen blocking indexing UI (no chat system messages);
  // the init poller refreshes the document list + fragment count on completion.
  const handleReindex = useCallback(async () => {
    if (isReindexing) return;
    setIsReindexing(true);
    setSettingsError(null);
    const fileList = [...pendingReindexFiles];
    setPendingReindexFiles([]);
    try {
      const result = await api.reindex(fileList.length ? fileList : undefined);
      if (result.ok && result.indexing) {
        setIndexingError(null);
        setIndexingProgress(result.progress || null);
        setIsIndexing(true);
        setRetryTrigger(t => t + 1);
      } else if (result.ok) {
        setTotalFragments(result.total_fragments || 0);
        setDocuments(result.documents || []);
      } else {
        setSettingsError(fill(T.reindexError, { error: result.error }));
      }
    } catch {
      setSettingsError(T.reindexConnError);
    } finally {
      setIsReindexing(false);
      if (reindexFileInputRef.current) reindexFileInputRef.current.value = '';
    }
  }, [isReindexing, pendingReindexFiles, lang]);

  // ---- Delete document ----
  const handleDeleteDoc = useCallback(async (docName: string) => {
    if (deletingDoc) return;
    if (!window.confirm(fill(T.confirmDelete, { name: docName }))) return;
    setDeletingDoc(docName);
    try {
      const result = await api.deleteDoc(docName);
      if (result.ok) {
        setDocuments(result.documents || []);
        if (typeof result.total_fragments === 'number') setTotalFragments(result.total_fragments);
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
          content: fill(T.deleteError, { error: result.error }),
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
  }, [mode, deletingDoc, lang]);

  // ---- Control panel loaders (stores / model roles / Ollama) ----
  const loadStores = useCallback(async () => {
    const res = await api.listStores().catch(() => null);
    if (res?.ok) {
      setStores(res.stores || []);
      setActiveStore(res.active || 'en');
    }
  }, []);

  const loadModelRoles = useCallback(async () => {
    const res = await api.getModels().catch(() => null);
    if (res?.ok) setModelRoles(res.roles);
  }, []);

  const refreshOllama = useCallback(async () => {
    const st = await api.ollamaStatus().catch(() => null);
    if (st?.ok) {
      setOllamaStatus({ running: st.running, version: st.version });
      if (st.running) {
        const m = await api.ollamaModels().catch(() => null);
        if (m?.ok) setOllamaModels(m.models || []);
      } else {
        setOllamaModels([]);
      }
    }
  }, []);

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
      if (res.indexing) {
        setIndexingError(null);
        setIndexingProgress(null);
        setIsIndexing(true);
        setRetryTrigger(t => t + 1);
        return;
      }
      setDocuments(res.documents || []);
      setTotalFragments(res.total_fragments || 0);
      setFingerprintStale(res.fingerprint_stale || false);
    } catch {
      setActiveStore(prev);
      setStoreError(T.corpusConnError);
    } finally {
      setStoreBusy(false);
    }
  }, [activeStore, storeBusy, lang]);

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
  }, [modelRoles, savingRole, lang]);

  // ---- Start the local Ollama server ----
  const handleStartOllama = useCallback(async () => {
    if (ollamaStarting) return;
    setOllamaStarting(true);
    try {
      const res = await api.startOllama();
      setOllamaStatus({ running: !!res.running, version: res.version || null });
      if (res.running) {
        const m = await api.ollamaModels().catch(() => null);
        if (m?.ok) setOllamaModels(m.models || []);
      }
    } catch {
      /* keep stopped */
    } finally {
      setOllamaStarting(false);
    }
  }, [ollamaStarting]);

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

  // ---- Send message (streaming) ----
  // ---- Study mode: matching a typed document, and asking for the artifact ----

  /** Documents whose name contains what has been typed, best-first.
   *
   * Substring rather than prefix: nobody remembers whether a paper is filed as
   * "attention-transformers" or "transformers-attention", and typing the half
   * you do remember should find it. A name that STARTS with the query sorts
   * first, so the prefix case still behaves like a prefix search. */
  const documentSuggestions = React.useMemo(() => {
    if (mode !== 'study') return [];
    const query = input.trim().toLowerCase();
    // Nothing typed, nothing suggested. A list that opens on focus covers the
    // conversation the user just generated, and the sidebar already lists the
    // corpus for anyone who needs to look it up.
    if (!query) return [];
    const hits = documents.filter(d => d.toLowerCase().includes(query));
    // An exact match is a decision already made, not a suggestion to show.
    if (hits.length === 1 && hits[0].toLowerCase() === query) return [];
    return hits.sort((a, b) => {
      const aStarts = a.toLowerCase().startsWith(query) ? 0 : 1;
      const bStarts = b.toLowerCase().startsWith(query) ? 0 : 1;
      return aStarts - bStarts || a.localeCompare(b);
    });
  }, [mode, input, documents]);

  /** The document a typed message names, or null.
   *
   * Falls back to a unique substring match so "planck" sends, rather than
   * making the user complete the filename the autocomplete already showed. */
  const resolveDocument = useCallback((text: string): string | null => {
    const query = text.trim().toLowerCase();
    if (!query) return null;
    const exact = documents.find(d => d.toLowerCase() === query);
    if (exact) return exact;
    const contained = documents.filter(d => query.includes(d.toLowerCase()));
    if (contained.length === 1) return contained[0];
    const partial = documents.filter(d => d.toLowerCase().includes(query));
    return partial.length === 1 ? partial[0] : null;
  }, [documents]);

  const acceptSuggestion = useCallback((name: string) => {
    setInput(name);
    setSuggestionIndex(0);
    textareaRef.current?.focus();
  }, []);

  const sendStudy = useCallback(async (text: string) => {
    const doc = resolveDocument(text);
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: text, mode: 'study' };
    const assistantId = (Date.now() + 1).toString();

    if (!doc) {
      // Named before anything is spent: the failure is in the request, and a
      // spinner followed by "not found" would hide that.
      setMessages(prev => [...prev, userMsg, {
        id: assistantId, role: 'assistant', mode: 'study', isError: true,
        content: fill(T.studyNoMatch, { query: text }),
      }]);
      setInput('');
      return;
    }

    setMessages(prev => [...prev, userMsg, {
      id: assistantId, role: 'assistant', content: '', mode: 'study', isStreaming: true,
    }]);
    setInput('');
    setIsLoading(true);
    const started = Date.now();

    try {
      const langName = lang === 'en' ? 'English' : lang === 'ca' ? 'Valencià' : 'Castellano';
      const { status, body } = await api.study(studyKind, doc, langName);
      const elapsed = ((Date.now() - started) / 1000).toFixed(1);
      setMessages(prev => prev.map(m => m.id !== assistantId ? m : (
        status === 200 && body.ok
          ? { ...m, isStreaming: false, artifact: body.artifact as StudyArtifact,
              artifactKind: studyKind, content: '',
              metrics: { searchTime: `${elapsed}s`, chunks: 0 } }
          // 422 is the model ignoring the format, not the server breaking, and
          // the two need different advice.
          : { ...m, isStreaming: false, isError: true,
              content: status === 422 ? T.studyMalformed
                                      : fill(T.studyFailed, { error: String(body.error ?? status) }) }
      )));
    } catch (e) {
      setMessages(prev => prev.map(m => m.id !== assistantId ? m
        : { ...m, isStreaming: false, isError: true, content: fill(T.studyFailed, { error: String(e) }) }));
    } finally {
      setIsLoading(false);
    }
  }, [resolveDocument, studyKind, lang, T]);

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

    const startTime = performance.now();

    try {
      const response = mode === 'rag' ? await api.rag(text) : await api.chat(text);

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
                    metrics: mode === 'rag' ? { searchTime: `${elapsed}s`, chunks: sources?.reduce((acc, s) => acc + s.pages.length, 0) || 0 } : undefined,
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
        { unknown: T.unknownError, queryError: T.queryError, modelError: T.modelError, closed: T.connClosed, no_results: T.noResults },
      );
    } catch {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId
            ? { ...m, content: T.connClosed, isStreaming: false, isError: true }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  }, [input, mode, isLoading, lang, sendStudy]);

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

  // ---- Connection error screen ----
  const handleRetry = useCallback(async () => {
    setInitError(null);
    setIsIndexing(true);
    try {
      const result = await api.init();
      if (result.ok) {
        setMode(result.mode || 'rag');
        setDocuments(result.documents || []);
        setTotalFragments(result.total_fragments || 0);
        setFingerprintStale(result.fingerprint_stale || false);
        setActiveStore(result.active_store || 'en');
        setIndexingError(null);
        setIndexingProgress(null);
        setIsInitialized(true);
        setIsIndexing(false);
      } else if (result.indexing) {
        setIndexingError(result.error || null);
        setIndexingProgress(result.progress || null);
        setRetryTrigger(t => t + 1);
      } else {
        setInitError(result.error || T.retryError);
        setIsIndexing(false);
      }
    } catch {
      setInitError(T.noServer);
      setIsIndexing(false);
    }
  }, [lang]);

  // ---- Indexing screen (cuando ok: false desde API, nunca "Error de conexión") ----
  if (isIndexing) {
    const showRetry = indexingError && /falló|fallido|failed|refused|no se pudo/i.test(indexingError);
    return (
      <IndexingScreen
        title={T.indexingTitle}
        hint={T.indexingHint}
        error={indexingError}
        showRetry={Boolean(showRetry)}
        progress={indexingProgress}
        processingLabel={T.processing}
        fileUnit={T.fileUnit}
        fileUnitPlural={T.fileUnitPlural}
        autoRefreshLabel={T.autoRefresh}
        retryLabel={T.retry}
        onRetry={handleRetry}
      />
    );
  }

  // ---- Connection error screen ----
  if (initError) {
    return (
      <ConnectionErrorScreen
        title={T.connErrorTitle}
        error={initError}
        retryLabel={T.retry}
        onRetry={handleRetry}
      />
    );
  }

  // ---- Loading screen ----
  if (!isInitialized) {
    return (
      <LoadingScreen message={T.connecting} />
    );
  }

  // ---- Main UI ----
  return (
    <div className="flex h-screen bg-transparent text-ink font-sans overflow-hidden selection:bg-brand/25 p-2 md:p-4 gap-4">

      {/* Mobile Sidebar Overlay */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-field/80 backdrop-blur-sm z-40 md:hidden"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Hidden file input */}
      <input
        ref={reindexFileInputRef}
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={(e) => { const f = e.target.files; if (f?.length) setPendingReindexFiles(Array.from(f)); }}
      />

      {/* Sidebar */}
      <motion.aside
        className={`fixed md:relative z-50 h-[calc(100vh-16px)] md:h-full w-[320px] glass-panel rounded-3xl flex flex-col transition-transform duration-300 ease-in-out shadow-2xl ${isSidebarOpen ? 'translate-x-2 md:translate-x-0' : '-translate-x-[120%] md:translate-x-0 md:w-0 md:opacity-0 md:overflow-hidden md:ml-[-16px]'}`}
      >
        {/* Sidebar Header */}
        <div className="p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="MonkeyGrab" className="w-9 h-9 object-cover flex-shrink-0 grayscale" />
            <h1 className="flex font-extrabold text-lg tracking-tight"><ShimmerText text="MonkeyGrab" /></h1>
          </div>
          <button className="md:hidden text-ink-faint hover:text-ink transition-colors bg-field p-2 rounded-full" onClick={() => setIsSidebarOpen(false)}>
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Sidebar Tabs — Documents stays in the sidebar; Models & Pipeline
            open a full-area overlay in the main column. */}
        <div className="flex px-6 mb-2">
          <div className="flex w-full bg-field p-1 border border-edge rounded-xl">
            <button
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${mainPanel === null ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
              onClick={() => { setMainPanel(null); setActiveTab('docs'); }}
            >
              {T.tabDocs}
            </button>
            <button
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${mainPanel === 'models' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
              onClick={() => openMainPanel('models')}
            >
              {T.tabModels}
            </button>
            <button
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${mainPanel === 'pipeline' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
              onClick={() => openMainPanel('pipeline')}
            >
              {T.tabPipeline}
            </button>
          </div>
        </div>

        {/* Sidebar Content */}
        <div className="flex-1 min-h-0 overflow-y-auto px-6 py-4 custom-scrollbar">
          <AnimatePresence mode="wait">
            {activeTab === 'docs' ? (
              <motion.div
                key="docs"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="space-y-6"
              >
                <div className="space-y-2">
                  <div className="pl-2 text-[10px] font-bold text-ink-faint uppercase tracking-widest">
                    {T.storesLabel}
                  </div>
                  <div className={`rounded-2xl border border-edge bg-field p-1.5 space-y-1 ${storeBusy || isReindexing || isLoading ? 'opacity-50' : ''}`}>
                    {stores.map(store => {
                      const isActive = store.name === activeStore;
                      return (
                        <button
                          key={store.name}
                          type="button"
                          className={`group flex w-full items-center gap-2 rounded-xl border px-2.5 py-2 text-left transition-all focus:outline-none disabled:cursor-not-allowed ${isActive
                            ? 'border-divider bg-surface-raised'
                            : 'border-transparent hover:border-edge hover:bg-surface-raised/50'
                            }`}
                          onClick={() => handleStoreSelect(store.name)}
                          disabled={storeBusy || isReindexing || isLoading}
                        >
                          <span className="flex min-w-0 flex-1 flex-col">
                            <span className="flex items-center gap-1.5">
                              <span className={`truncate text-xs font-bold tracking-wide ${isActive ? 'text-ink' : 'text-ink-muted group-hover:text-ink'}`}>{store.label}</span>
                              {isActive && <Check className="h-3 w-3 flex-shrink-0 text-ink" />}
                            </span>
                            {/* Always ink-muted, active or not: this line carries
                                data (how many PDFs, whether they are indexed), and
                                ink-faint at 9px measures 3.35:1 against the surface.
                                Which store is active is already said twice over, by
                                the raised fill and the check. */}
                            <span className="mt-0.5 block truncate font-mono text-[9px] text-ink-muted">
                              {!store.indexed
                                ? `${store.pdf_count} PDF · ${T.storeNotIndexed}`
                                : store.fragments != null
                                  ? fill(T.fragments, { n: store.fragments })
                                  : `${store.pdf_count} PDF`}
                            </span>
                          </span>
                        </button>
                      );
                    })}
                  </div>
                  {storeError && <p className="pl-2 text-[11px] text-danger">{storeError}</p>}
                  {fingerprintStale && !isIndexing && (
                    <p className="flex items-start gap-1.5 rounded-xl border border-danger/30 bg-danger/10 px-2.5 py-2 text-[11px] text-danger">
                      <AlertCircle className="w-3 h-3 mt-0.5 shrink-0" />
                      <span>{T.fingerprintStaleWarning}</span>
                    </p>
                  )}
                </div>

                {/* Documents list */}
                <div className="space-y-3">
                  <div className="text-[10px] font-bold text-ink-faint uppercase tracking-widest pl-2">
                    {fill(T.collection, { n: documents.length })}
                  </div>

                  <div className="space-y-2">
                    {documents.length === 0 ? (
                      <p className="text-xs text-ink-muted text-center py-4">{T.noDocs}</p>
                    ) : (
                      documents.map((doc, i) => (
                        <div key={i} className="group flex items-center gap-3 p-3.5 rounded-2xl bg-field border border-edge hover:border-divider transition-all">
                          <div className="w-8 h-8 rounded-full bg-surface-raised flex items-center justify-center flex-shrink-0">
                            <FileText className="w-4 h-4 text-ink-muted" />
                          </div>
                          <span className="text-sm text-ink group-hover:text-ink truncate font-medium flex-1 min-w-0">{doc}</span>
                          <button
                            className="opacity-0 group-hover:opacity-100 p-1.5 rounded-full text-ink-faint hover:text-ink hover:bg-surface-raised transition-all flex-shrink-0"
                            onClick={() => openPdf(doc)}
                            title={T.viewPdf}
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            className="opacity-0 group-hover:opacity-100 p-1.5 rounded-full text-ink-faint hover:text-danger hover:bg-danger/10 transition-all flex-shrink-0 disabled:opacity-50"
                            onClick={() => handleDeleteDoc(doc)}
                            disabled={deletingDoc !== null}
                            title={T.deleteDoc}
                          >
                            {deletingDoc === doc ? (
                              <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                              <Trash2 className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>
      </motion.aside>

      {/* Main Content */}
      <main className="flex-1 flex min-w-0 relative glass-panel rounded-3xl overflow-hidden shadow-2xl">
        {/* PDF pane — 'full' replaces the chat (opened from the sidebar);
            'split' sits to the left of the chat (opened from a source citation). */}
        {pdfViewer && (
          <div className={`min-w-0 ${pdfViewer.mode === 'full' ? 'flex-1' : 'w-1/2 border-r border-divider'}`}>
            <PdfPane
              doc={pdfViewer.doc}
              page={pdfViewer.page}
              onClose={closePdf}
            />
          </div>
        )}

        {/* Models / Pipeline overlay — occupies the full main area, like the PDF viewer */}
        {mainPanel && pdfViewer?.mode !== 'full' && (
          <SettingsOverlay
            mainPanel={mainPanel}
            onClose={() => setMainPanel(null)}
            ollamaStatus={ollamaStatus}
            ollamaStarting={ollamaStarting}
            refreshOllama={refreshOllama}
            handleStartOllama={handleStartOllama}
            ollamaModels={ollamaModels}
            modelRoles={modelRoles}
            savingRole={savingRole}
            modelError={modelError}
            handleRoleChange={handleRoleChange}
            settings={settings}
            toggleSetting={toggleSetting}
            settingsError={settingsError}
            isReindexing={isReindexing}
            indexingProgress={indexingProgress}
            pendingReindexFiles={pendingReindexFiles}
            setPendingReindexFiles={setPendingReindexFiles}
            reindexFileInputRef={reindexFileInputRef}
            handleReindex={handleReindex}
            strings={T}
          />
        )}

        {/* Chat column — hidden while a full-screen PDF or a panel is open */}
        {pdfViewer?.mode !== 'full' && !mainPanel && (
        <div className="flex-1 flex flex-col min-w-0 relative">
        {/* Header */}
        <header className="h-20 border-b border-divider flex items-center justify-between gap-3 px-4 bg-surface-raised z-10">
          <div className="flex items-center gap-3 min-w-0">
            <button
              className="p-2.5 text-ink-muted hover:text-ink bg-field hover:bg-surface border border-edge rounded-lg transition-colors flex-shrink-0"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            >
              <Menu className="w-5 h-5" />
            </button>

            <div className="flex bg-field p-1 border border-edge rounded-xl flex-shrink-0">
              <button
                className={`min-w-[84px] justify-center px-4 py-2 text-xs font-bold tracking-wide rounded-lg transition-all flex items-center gap-2 ${mode === 'chat' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
                onClick={() => handleModeChange('chat')}
              >
                <Ollama className="w-4 h-4 text-ink" />
                CHAT
              </button>
              <button
                className={`min-w-[84px] justify-center px-4 py-2 text-xs font-bold tracking-wide rounded-lg transition-all flex items-center gap-2 ${mode === 'rag' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
                onClick={() => handleModeChange('rag')}
              >
                <Database className="w-4 h-4" />
                RAG
              </button>
              <button
                className={`min-w-[84px] justify-center px-4 py-2 text-xs font-bold tracking-wide rounded-lg transition-all flex items-center gap-2 ${mode === 'study' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
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
              className="p-2 text-ink-muted hover:text-ink bg-field hover:bg-surface border border-edge rounded-lg transition-colors"
              title={theme === 'dark' ? T.themeLight : T.themeDark}
              aria-label={theme === 'dark' ? T.themeLight : T.themeDark}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
            <LanguageToggle lang={lang} setLang={setLang} />
            <button
              className="p-2 text-ink-muted hover:text-ink bg-field hover:bg-surface border border-edge rounded-lg transition-colors"
              onClick={handleClear}
              title={T.clearChat}
              aria-label={T.clearChat}
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 custom-scrollbar scroll-smooth relative">
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
                          <span className="text-[9px] px-2 py-0.5 rounded-full uppercase tracking-widest font-bold bg-field text-ink-faint border border-edge">
                            {msg.mode}
                          </span>
                        )}
                        <button
                          onClick={() => handleCopyMessage(msg)}
                          className="p-1.5 rounded-full text-ink-faint hover:text-ink hover:bg-surface-raised transition-all opacity-0 group-hover/meta:opacity-100"
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
                            onReveal={() => setRevealedQuiz(prev => ({ ...prev, [msg.id]: true }))}
                            strings={T}
                          />
                        ) : msg.content ? (
                          <MarkdownContent text={msg.content} />
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
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
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
                      i === suggestionIndex
                        ? 'bg-surface-raised text-ink'
                        : 'text-ink-muted hover:text-ink'
                    }`}
                  >
                    <FileText className={`w-4 h-4 shrink-0 ${i === suggestionIndex ? 'text-ink' : 'text-ink-faint'}`} />
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
                      const next = e.key === 'ArrowDown' ? i + 1 : i - 1;
                      return (next + documentSuggestions.length) % documentSuggestions.length;
                    });
                    return;
                  }
                  // Tab completes; Enter sends. Keeping them apart means a
                  // half-typed name never becomes a request by reflex.
                  if (suggesting && e.key === 'Tab') {
                    e.preventDefault();
                    acceptSuggestion(documentSuggestions[suggestionIndex]);
                    return;
                  }
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder={mode === 'study' ? T.studyPlaceholder : mode === 'rag' ? T.placeholderRag : T.placeholderChat}
                className={`${mode === 'study' ? 'w-full max-h-40 min-h-[44px]' : 'flex-1 max-h-48 min-h-[52px]'} bg-transparent border-none focus:ring-0 focus:outline-none resize-none py-3.5 px-4 text-[15px] text-ink placeholder:text-ink-faint custom-scrollbar font-medium`}
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
                        onClick={() => setStudyKind(kind)}
                        className={`px-3 py-2 text-[11px] font-bold tracking-wide uppercase rounded transition-all ${
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
                    className="p-3 bg-ink text-surface hover:opacity-90 active:scale-95 disabled:opacity-40 disabled:bg-surface-raised disabled:text-ink-faint transition-all flex-shrink-0 rounded-xl"
                  >
                    {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5 ml-0.5" />}
                  </button>
                </div>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                  className="p-3.5 bg-ink text-surface hover:opacity-90 active:scale-95 disabled:opacity-40 disabled:bg-surface-raised disabled:text-ink-faint transition-all flex-shrink-0 rounded-2xl"
                >
                  {isLoading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5 ml-0.5" />
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
        </div>
        )}
      </main>
    </div>
  );

}
