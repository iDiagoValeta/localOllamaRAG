import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send, FileText, MessageSquare,
  Database,
  Search, Layers, FileUp, Menu, X,
  RefreshCw, Loader2, AlertCircle, CheckCircle2, Trash2,
  ChevronDown, ChevronRight, Copy, Check
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

// =============================================================================
// Types
// =============================================================================

type Mode = 'chat' | 'rag';

interface Citation {
  document: string;
  pages: number[];
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  mode: Mode;
  citations?: Citation[];
  metrics?: { searchTime: string; chunks: number };
  isStreaming?: boolean;
  isError?: boolean;
}

interface PipelineSettings {
  contextualRetrieval: boolean;
  queryDecomposition: boolean;
  hybridSearch: boolean;
  exhaustiveSearch: boolean;
  reranker: boolean;
  expandContext: boolean;
  optimizeContext: boolean;
  recompSynthesis: boolean;
}

interface IndexingProgress {
  file: string;
  file_index: number;
  total_files: number;
}

// =============================================================================
// API Service — connects to Flask backend
// =============================================================================

const API_BASE = '/api';
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

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

  upload: (file: File, addOnly = false) => {
    const form = new FormData();
    form.append('file', file);
    const url = addOnly ? `${API_BASE}/upload?add_only=1` : `${API_BASE}/upload`;
    return fetch(url, { method: 'POST', body: form }).then(r => r.json());
  },

  deleteDoc: (filename: string) =>
    fetch(`${API_BASE}/docs/${encodeURIComponent(filename)}`, { method: 'DELETE' }).then(r => r.json()),
};

// =============================================================================
// SSE Stream Parser
// =============================================================================

async function streamSSE(
  response: Response,
  onToken: (token: string) => void,
  onDone: (sources: Citation[] | null) => void,
  onError: (msg: string) => void,
) {
  const contentType = response.headers.get('content-type') || '';

  // If the response is NOT a stream (error responses come as JSON)
  if (!contentType.includes('event-stream')) {
    const data = await response.json().catch(() => ({ message: 'Error desconocido' }));
    if (!data.ok && data.message) {
      onError(data.message);
    } else if (!data.ok && data.error) {
      onError(typeof data.error === 'string' ? data.error : 'Error en la consulta');
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
        onError(data.message || data.error || 'Error en la respuesta del modelo');
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
  if (!finished) onError('La conexión se cerró antes de completar la respuesta.');
}

// =============================================================================
// Small safe Markdown renderer
// =============================================================================

function MathInline({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: false });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function MathBlock({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: true });
  return <div className="overflow-x-auto my-2" dangerouslySetInnerHTML={{ __html: html }} />;
}

function renderInlineMarkdown(text: string, keyPrefix: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const pattern = /(\$\$[^$\n]+\$\$|\$(?!\$)(?:[^$\n\\]|\\.)+\$|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const token = match[0];
    const key = `${keyPrefix}-${match.index}`;

    if (token.startsWith('$$')) {
      parts.push(<MathBlock key={key} tex={token.slice(2, -2)} />);
    } else if (token.startsWith('$')) {
      parts.push(<MathInline key={key} tex={token.slice(1, -1)} />);
    } else if (token.startsWith('`')) {
      parts.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith('**')) {
      parts.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else {
      parts.push(<em key={key}>{token.slice(1, -1)}</em>);
    }

    last = match.index + token.length;
  }

  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function MarkdownContent({ text, compact = false }: { text: string; compact?: boolean }) {
  if (!text) return null;

  const className = compact ? 'markdown-content compact' : 'markdown-content';

  if (text.includes('```')) {
    const nodes: React.ReactNode[] = [];
    let inCode = false;
    let codeLines: string[] = [];
    let paragraph: string[] = [];
    let inMath = false;
    let mathLines: string[] = [];

    const flushParagraph = (key: string) => {
      if (!paragraph.length) return;
      nodes.push(<p key={key}>{paragraph.map((line, i) => <React.Fragment key={i}>{i > 0 && <br />}{renderInlineMarkdown(line, `${key}-${i}`)}</React.Fragment>)}</p>);
      paragraph = [];
    };

    text.split('\n').forEach((line, i) => {
      if (line.trim() === '$$' && !inCode) {
        if (inMath) {
          flushParagraph(`p-${i}`);
          nodes.push(<MathBlock key={`math-${i}`} tex={mathLines.join('\n')} />);
          mathLines = [];
          inMath = false;
        } else {
          flushParagraph(`p-${i}`);
          inMath = true;
        }
        return;
      }

      if (inMath) {
        mathLines.push(line);
        return;
      }

      if (/^```/.test(line)) {
        if (inCode) {
          nodes.push(<pre key={`code-${i}`}><code>{codeLines.join('\n')}</code></pre>);
          codeLines = [];
          inCode = false;
        } else {
          flushParagraph(`p-${i}`);
          inCode = true;
        }
        return;
      }

      if (inCode) {
        codeLines.push(line);
      } else if (line.trim()) {
        paragraph.push(line);
      } else {
        flushParagraph(`p-${i}`);
      }
    });

    if (inCode) nodes.push(<pre key="code-final"><code>{codeLines.join('\n')}</code></pre>);
    flushParagraph('p-final');
    return <div className={className}>{nodes}</div>;
  }

  return (
    <div className={className}>
      {text.split(/\n{2,}/).map((block, i) => {
        const lines = block.split('\n').filter(Boolean);
        if (!lines.length) return null;

        const trimmed = block.trim();
        if (trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length > 4) {
          const inner = trimmed.slice(2, -2).trim();
          return <MathBlock key={i} tex={inner} />;
        }

        const heading = lines[0].match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
          const Tag = heading[1].length === 1 ? 'h2' : heading[1].length === 2 ? 'h3' : 'h4';
          return <Tag key={i}>{renderInlineMarkdown(heading[2], `h-${i}`)}</Tag>;
        }

        if (lines.every(line => /^[-*]\s+/.test(line))) {
          return (
            <ul key={i}>
              {lines.map((line, j) => (
                <li key={j}>{renderInlineMarkdown(line.replace(/^[-*]\s+/, ''), `ul-${i}-${j}`)}</li>
              ))}
            </ul>
          );
        }

        if (lines.every(line => /^\d+\.\s+/.test(line))) {
          return (
            <ol key={i}>
              {lines.map((line, j) => (
                <li key={j}>{renderInlineMarkdown(line.replace(/^\d+\.\s+/, ''), `ol-${i}-${j}`)}</li>
              ))}
            </ol>
          );
        }

        return (
          <p key={i}>
            {lines.map((line, j) => (
              <React.Fragment key={j}>
                {j > 0 && <br />}
                {renderInlineMarkdown(line, `p-${i}-${j}`)}
              </React.Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}

// =============================================================================
// App Component
// =============================================================================

export default function App() {
  const [mode, setMode] = useState<Mode>('rag');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<'docs' | 'settings'>('docs');
  const [documents, setDocuments] = useState<string[]>([]);
  const [totalFragments, setTotalFragments] = useState(0);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexingProgress, setIndexingProgress] = useState<IndexingProgress | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isReindexing, setIsReindexing] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [pendingReindexFiles, setPendingReindexFiles] = useState<File[]>([]);
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    indexacion: false,
    recuperacion: false,
    ranking: false,
    reindexacion: true,  // Reindexación abierta por defecto para ver los botones
  });

  const [settings, setSettings] = useState<PipelineSettings>({
    contextualRetrieval: true,
    queryDecomposition: true,
    hybridSearch: true,
    exhaustiveSearch: false,
    reranker: true,
    expandContext: true,
    optimizeContext: true,
    recompSynthesis: true,
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const reindexFileInputRef = useRef<HTMLInputElement>(null);
  const [retryTrigger, setRetryTrigger] = useState(0);
  const [indexingError, setIndexingError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);

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
          setIsInitialized(true);

          setMessages([]);
        } else {
          // ok: false = indexando o error previo → siempre pantalla de indexación (nunca "Error de conexión")
          setInitError(null);
          setIsIndexing(true);
          setIndexingError(initData.error || null);
          if (initData.progress) setIndexingProgress(initData.progress);
          setTimeout(() => { if (!cancelled) init(); }, 5000);
        }

        try {
          const settingsData = await api.getSettings();
          if (settingsData.ok) setSettings(prev => ({ ...prev, ...settingsData.settings }));
        } catch {
          /* ignorar fallo de settings */
        }
      } catch (err) {
        if (!cancelled) setInitError('No se pudo conectar con el servidor. ¿Está Flask ejecutándose?');
      }
    }
    init();
    return () => { cancelled = true; };
  }, [retryTrigger]);

  // ---- Mode switching ----
  const handleModeChange = useCallback(async (newMode: Mode) => {
    const previousMode = mode;
    setMode(newMode);
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
    } else {
      setSettings(prev => ({ ...prev, [key]: previousVal }));
      setSettingsError(result?.error || 'No se pudo guardar el ajuste.');
    }
  }, [settings]);

  const waitForIndexingToFinish = useCallback(async () => {
    for (;;) {
      await sleep(1500);
      const status = await api.init();
      if (status.ok) return status;
      if (status.indexing) {
        setIndexingProgress(status.progress || null);
        setIndexingError(status.error || null);
        continue;
      }
      throw new Error(status.message || status.error || 'La indexación no pudo completarse.');
    }
  }, []);

  // ---- Add PDF (add-only, sin reindexar) ----
  const handleAddPdf = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);

    try {
      const result = await api.upload(file, true);
      if (result.ok) {
        setDocuments(result.documents || []);
        setTotalFragments(result.total_fragments || 0);
        const names = result.files?.length ? result.files.join(', ') : result.filename || '';
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `✓ **${names}** añadido con los ajustes actuales (${result.total_fragments} fragmentos).`,
          mode,
        }]);
      } else {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `✗ Error al subir: ${result.error}`,
          mode,
          isError: true,
        }]);
      }
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: '✗ Error de conexión al subir el archivo.',
        mode,
        isError: true,
      }]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [mode]);

  // ---- Reindex (full, con ajustes del pipeline) ----
  const handleReindex = useCallback(async () => {
    if (isReindexing) return;
    setIsReindexing(true);
    const fileList = [...pendingReindexFiles];
    setPendingReindexFiles([]);
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      role: 'system',
      content: fileList.length ? `⟳ Añadiendo ${fileList.length} PDF(s) y re-indexando…` : '⟳ Re-indexando con ajustes actuales…',
      mode,
    }]);

    try {
      const result = await api.reindex(fileList.length ? fileList : undefined);
      if (result.ok && result.indexing) {
        setIndexingError(null);
        setIndexingProgress(result.progress || null);

        const finalStatus = await waitForIndexingToFinish();
        setMode(finalStatus.mode || mode);
        setTotalFragments(finalStatus.total_fragments || 0);
        setDocuments(finalStatus.documents || []);
        setIndexingProgress(null);

        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `✓ Re-indexación completada: ${finalStatus.total_fragments || 0} fragmentos, ${(finalStatus.documents || []).length} documentos.`,
          mode,
        }]);
      } else if (result.ok) {
        setTotalFragments(result.total_fragments || 0);
        setDocuments(result.documents || []);
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `✓ Re-indexación completada: ${result.total_fragments || 0} fragmentos, ${(result.documents || []).length} documentos.`,
          mode,
        }]);
      } else {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `✗ Error: ${result.error}`,
          mode,
          isError: true,
        }]);
      }
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: '✗ Error de conexión.',
        mode,
        isError: true,
      }]);
    } finally {
      setIsReindexing(false);
      if (reindexFileInputRef.current) reindexFileInputRef.current.value = '';
    }
  }, [mode, isReindexing, pendingReindexFiles, waitForIndexingToFinish]);

  // ---- Delete document ----
  const handleDeleteDoc = useCallback(async (docName: string) => {
    if (deletingDoc) return;
    if (!window.confirm(`¿Eliminar "${docName}"? Se borrará del índice y del disco.`)) return;
    setDeletingDoc(docName);
    try {
      const result = await api.deleteDoc(docName);
      if (result.ok) {
        setDocuments(result.documents || []);
        if (typeof result.total_fragments === 'number') setTotalFragments(result.total_fragments);
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `✓ Documento **${docName}** eliminado.`,
          mode,
        }]);
      } else {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `✗ Error al eliminar: ${result.error}`,
          mode,
          isError: true,
        }]);
      }
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: '✗ Error de conexión al eliminar.',
        mode,
        isError: true,
      }]);
    } finally {
      setDeletingDoc(null);
    }
  }, [mode, deletingDoc]);

  // ---- Send message (streaming) ----
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

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
      );
    } catch {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId
            ? { ...m, content: 'Error de conexión con el servidor.', isStreaming: false, isError: true }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  }, [input, mode, isLoading]);

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
  const handleClear = useCallback(async () => {
    await api.clear().catch(() => {});
    setMessages([{
      id: 'cleared',
      role: 'system',
      content: 'Historial limpiado.',
      mode,
    }]);
  }, [mode]);

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
        setIndexingError(null);
        setIndexingProgress(null);
        setIsInitialized(true);
        setIsIndexing(false);
      } else if (result.indexing) {
        setIndexingError(result.error || null);
        setIndexingProgress(result.progress || null);
        setRetryTrigger(t => t + 1);
      } else {
        setInitError(result.error || 'Error al reintentar');
        setIsIndexing(false);
      }
    } catch {
      setInitError('No se pudo conectar con el servidor. ¿Está Flask ejecutándose?');
      setIsIndexing(false);
    }
  }, []);

  // ---- Indexing screen (cuando ok: false desde API, nunca "Error de conexión") ----
  if (isIndexing) {
    const showRetry = indexingError && /falló|fallido|failed|refused|no se pudo/i.test(indexingError);
    return (
      <div className="flex h-screen items-center justify-center bg-[#050505] text-zinc-300 p-4">
        <div className="glass-panel rounded-3xl p-10 max-w-md text-center space-y-4">
          <Loader2 className="w-12 h-12 text-orange-500 animate-spin mx-auto" />
          <h2 className="text-xl font-semibold text-white">
            Se están indexando los documentos
          </h2>
          <p className="text-zinc-400 text-sm">
            Puede tardar unos minutos dependiendo de tu hardware.
          </p>
          {indexingError && showRetry && (
            <p className="text-amber-400 text-sm">{indexingError}</p>
          )}
          {indexingProgress ? (
            <div className="space-y-1">
              <p className="text-zinc-300 text-sm font-medium">
                Procesando: <span className="text-orange-500">{indexingProgress.file}</span>
              </p>
              <p className="text-zinc-500 text-xs">
                {indexingProgress.file_index} / {indexingProgress.total_files} archivo{indexingProgress.total_files !== 1 ? 's' : ''}
              </p>
              <div className="w-full bg-zinc-800 rounded-full h-1.5 mt-2">
                <div
                  className="bg-orange-500 h-1.5 rounded-full transition-all duration-500"
                  style={{ width: `${(indexingProgress.file_index / indexingProgress.total_files) * 100}%` }}
                />
              </div>
            </div>
          ) : (
            <p className="text-zinc-500 text-xs">
              La página se actualizará automáticamente al terminar.
            </p>
          )}
          {showRetry && (
            <button
              className="px-6 py-2 bg-orange-500 text-black rounded-full font-semibold hover:bg-orange-400 transition-colors"
              onClick={handleRetry}
            >
              Reintentar
            </button>
          )}
        </div>
      </div>
    );
  }

  // ---- Connection error screen ----
  if (initError) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#050505] text-zinc-300 p-4">
        <div className="glass-panel rounded-3xl p-10 max-w-md text-center space-y-4">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto" />
          <h2 className="text-xl font-semibold text-white">Error de conexión</h2>
          <p className="text-zinc-400 text-sm">{initError}</p>
          <button
            className="px-6 py-2 bg-orange-500 text-black rounded-full font-semibold hover:bg-orange-400 transition-colors"
            onClick={handleRetry}
          >
            Reintentar
          </button>
        </div>
      </div>
    );
  }

  // ---- Loading screen ----
  if (!isInitialized) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#050505] text-zinc-300">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 text-orange-400 animate-spin" />
          <p className="text-zinc-500 text-sm">Conectando con MonkeyGrab…</p>
        </div>
      </div>
    );
  }

  // ---- Main UI ----
  return (
    <div className="flex h-screen bg-[#050505] text-zinc-300 font-sans overflow-hidden selection:bg-orange-500/30 p-2 md:p-4 gap-4">

      {/* Mobile Sidebar Overlay */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Hidden file inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={handleAddPdf}
      />
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
        className={`fixed md:relative z-50 h-[calc(100vh-16px)] md:h-full w-[320px] glass-panel rounded-xl flex flex-col transition-transform duration-300 ease-in-out shadow-2xl ${isSidebarOpen ? 'translate-x-2 md:translate-x-0' : '-translate-x-[120%] md:translate-x-0 md:w-0 md:opacity-0 md:overflow-hidden md:ml-[-16px]'}`}
      >
        {/* Sidebar Header */}
        <div className="p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/logo.jpg" alt="MonkeyGrab" className="w-10 h-10 rounded-full object-cover flex-shrink-0" />
            <h1 className="font-semibold text-orange-400 text-lg tracking-tight">MonkeyGrab</h1>
          </div>
          <button className="md:hidden text-zinc-500 hover:text-white transition-colors bg-white/5 p-2 rounded-full" onClick={() => setIsSidebarOpen(false)}>
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Sidebar Tabs */}
        <div className="flex px-6 mb-2">
          <div className="flex w-full bg-black/40 rounded-full p-1 border border-white/5">
            <button
              className={`flex-1 py-2 text-xs font-semibold rounded-full transition-all ${activeTab === 'docs' ? 'bg-white/10 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
              onClick={() => setActiveTab('docs')}
            >
              Documentos
            </button>
            <button
              className={`flex-1 py-2 text-xs font-semibold rounded-full transition-all ${activeTab === 'settings' ? 'bg-white/10 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
              onClick={() => setActiveTab('settings')}
            >
              Pipeline RAG
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
                {/* Add PDF button (add-only, sin reindexar) */}
                <button
                  className="w-full py-4 px-4 rounded-2xl border border-dashed border-white/20 text-zinc-400 hover:text-white hover:border-orange-500/50 hover:bg-orange-500/5 transition-all flex flex-col items-center justify-center gap-2 text-sm group disabled:opacity-50"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                >
                  <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-orange-500/10 transition-colors">
                    {isUploading ? (
                      <Loader2 className="w-5 h-5 animate-spin text-orange-400" />
                    ) : (
                      <FileUp className="w-5 h-5 group-hover:text-orange-400" />
                    )}
                  </div>
                  <span className="font-medium">{isUploading ? 'Añadiendo…' : 'Añadir PDF'}</span>
                  <span className="text-[10px] text-zinc-600">Usa los ajustes actuales del pipeline</span>
                </button>

                {/* Documents list */}
                <div className="space-y-3">
                  <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest pl-2">
                    Colección ({documents.length} docs)
                  </div>

                  <div className="space-y-2">
                    {documents.length === 0 ? (
                      <p className="text-xs text-zinc-600 text-center py-4">No hay documentos indexados</p>
                    ) : (
                      documents.map((doc, i) => (
                        <div key={i} className="group flex items-center gap-3 p-3.5 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-white/10 transition-all">
                          <div className="w-8 h-8 rounded-full bg-black/30 flex items-center justify-center flex-shrink-0">
                            <FileText className="w-4 h-4 text-orange-400/80" />
                          </div>
                          <span className="text-sm text-zinc-300 group-hover:text-white truncate font-medium flex-1 min-w-0">{doc}</span>
                          <button
                            className="opacity-0 group-hover:opacity-100 p-1.5 rounded-full text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-all flex-shrink-0 disabled:opacity-50"
                            onClick={() => handleDeleteDoc(doc)}
                            disabled={deletingDoc !== null}
                            title="Eliminar documento"
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
            ) : (
              <motion.div
                key="settings"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="space-y-2 pb-6"
              >
                {(settingsError || isReindexing) && (
                  <div className={`rounded-lg border px-3 py-2 text-xs ${settingsError ? 'border-red-500/20 bg-red-500/10 text-red-300' : 'border-orange-500/25 bg-orange-500/10 text-orange-200'}`}>
                    {settingsError ? (
                      <span>{settingsError}</span>
                    ) : indexingProgress ? (
                      <div className="space-y-1">
                        <div className="flex items-center justify-between gap-3">
                          <span className="truncate">Indexando {indexingProgress.file}</span>
                          <span className="font-mono text-[10px] text-orange-300">{indexingProgress.file_index}/{indexingProgress.total_files}</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-black/30 overflow-hidden">
                          <div
                            className="h-full bg-orange-400 transition-all duration-500"
                            style={{ width: `${Math.max(5, (indexingProgress.file_index / indexingProgress.total_files) * 100)}%` }}
                          />
                        </div>
                      </div>
                    ) : (
                      <span>Re-indexación en curso...</span>
                    )}
                  </div>
                )}

                {/* 1. Indexación */}
                <div className="rounded-xl border border-white/5 overflow-hidden">
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2.5 text-[10px] font-bold text-orange-400 uppercase tracking-widest bg-white/[0.02] hover:bg-white/5 transition-colors"
                    onClick={() => setOpenSections(s => ({ ...s, indexacion: !s.indexacion }))}
                  >
                    {openSections.indexacion ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    <span className="w-4 h-[1px] bg-orange-400/50" />
                    1. Indexación
                  </button>
                  {openSections.indexacion && (
                    <div className="p-2 pt-0 space-y-1 bg-white/[0.02]">
                      <Toggle label="Contextual Retrieval" checked={settings.contextualRetrieval} onChange={() => toggleSetting('contextualRetrieval')} desc="Enriquece chunks con LLM" />
                    </div>
                  )}
                </div>

                {/* 2. Recuperación */}
                <div className="rounded-xl border border-white/5 overflow-hidden">
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2.5 text-[10px] font-bold text-orange-400 uppercase tracking-widest bg-white/[0.02] hover:bg-white/5 transition-colors"
                    onClick={() => setOpenSections(s => ({ ...s, recuperacion: !s.recuperacion }))}
                  >
                    {openSections.recuperacion ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    <span className="w-4 h-[1px] bg-orange-400/50" />
                    2. Recuperación
                  </button>
                  {openSections.recuperacion && (
                    <div className="p-2 pt-0 space-y-1 bg-white/[0.02]">
                      <Toggle label="Búsqueda Híbrida" checked={settings.hybridSearch} onChange={() => toggleSetting('hybridSearch')} desc="Semántica + Keywords" />
                      <Toggle label="Query Decomposition" checked={settings.queryDecomposition} onChange={() => toggleSetting('queryDecomposition')} desc="Sub-queries con LLM auxiliar" />
                      <Toggle label="Búsqueda Exhaustiva" checked={settings.exhaustiveSearch} onChange={() => toggleSetting('exhaustiveSearch')} desc="Escaneo profundo (lento)" />
                    </div>
                  )}
                </div>

                {/* 3. Ranking & Contexto */}
                <div className="rounded-xl border border-white/5 overflow-hidden">
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2.5 text-[10px] font-bold text-orange-400 uppercase tracking-widest bg-white/[0.02] hover:bg-white/5 transition-colors"
                    onClick={() => setOpenSections(s => ({ ...s, ranking: !s.ranking }))}
                  >
                    {openSections.ranking ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    <span className="w-4 h-[1px] bg-orange-400/50" />
                    3. Ranking & Contexto
                  </button>
                  {openSections.ranking && (
                    <div className="p-2 pt-0 space-y-1 bg-white/[0.02]">
                      <Toggle label="Cross-Encoder Reranker" checked={settings.reranker} onChange={() => toggleSetting('reranker')} desc="Reordenamiento de precisión" />
                      <Toggle label="Expandir Contexto" checked={settings.expandContext} onChange={() => toggleSetting('expandContext')} desc="Añade chunks adyacentes" />
                      <Toggle label="Optimizar Contexto" checked={settings.optimizeContext} onChange={() => toggleSetting('optimizeContext')} desc="Limpia artefactos PDF" />
                      <Toggle label="RECOMP Synthesis" checked={settings.recompSynthesis} onChange={() => toggleSetting('recompSynthesis')} desc="Sintetiza contexto con LLM" />
                    </div>
                  )}
                </div>

                {/* 4. Reindexación */}
                <div className="rounded-xl border border-orange-500/20 overflow-hidden">
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2.5 text-[10px] font-bold text-orange-400 uppercase tracking-widest bg-orange-500/5 hover:bg-orange-500/10 transition-colors"
                    onClick={() => setOpenSections(s => ({ ...s, reindexacion: !s.reindexacion }))}
                  >
                    {openSections.reindexacion ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    <span className="w-4 h-[1px] bg-orange-400/50" />
                    4. Reindexación
                  </button>
                  {openSections.reindexacion && (
                    <div className="p-3 pt-0 space-y-3 bg-orange-500/5">
                      <p className="text-xs text-zinc-500">
                        Ajusta las opciones arriba y reindexa para aplicarlas. Opcionalmente añade PDFs nuevos.
                      </p>
                      <div className="flex gap-2">
                        <button
                          className="flex-1 py-3 px-4 rounded-2xl border border-dashed border-white/20 text-zinc-400 hover:text-white hover:border-orange-500/50 hover:bg-orange-500/5 transition-all flex flex-col items-center justify-center gap-1 text-sm group disabled:opacity-50"
                          onClick={() => reindexFileInputRef.current?.click()}
                          disabled={isReindexing}
                        >
                          <FileUp className="w-5 h-5 group-hover:text-orange-400" />
                          <span className="font-medium">
                            {pendingReindexFiles.length ? `${pendingReindexFiles.length} PDF(s)` : 'Añadir PDFs'}
                          </span>
                        </button>
                        {pendingReindexFiles.length > 0 && (
                          <button
                            className="px-3 rounded-2xl text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-all"
                            onClick={() => { setPendingReindexFiles([]); if (reindexFileInputRef.current) reindexFileInputRef.current.value = ''; }}
                            title="Quitar"
                          >
                            <X className="w-5 h-5" />
                          </button>
                        )}
                      </div>
                      <button
                        className="w-full py-3 px-4 rounded-2xl bg-orange-500/20 border border-orange-500/40 text-orange-400 hover:bg-orange-500/30 transition-all flex items-center justify-center gap-2 text-sm font-semibold disabled:opacity-50"
                        onClick={handleReindex}
                        disabled={isReindexing}
                      >
                        {isReindexing ? (
                          <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                          <RefreshCw className="w-5 h-5" />
                        )}
                        Reindexar
                      </button>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Sidebar Footer */}
        <div className="p-5 border-t border-white/5 text-xs text-zinc-500 flex items-center justify-between bg-black/20 rounded-b-xl">
          <span className="font-mono text-[10px] tracking-wider">{totalFragments} fragmentos</span>
          <div className="flex items-center gap-2 bg-white/5 px-2.5 py-1 rounded-full border border-white/10">
            <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.8)]" />
            <span className="font-medium text-zinc-300">Ollama Local</span>
          </div>
        </div>
      </motion.aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 relative glass-panel rounded-xl overflow-hidden shadow-2xl">
        {/* Header */}
        <header className="h-20 border-b border-white/5 flex items-center justify-between px-6 bg-black/20 z-10">
          <div className="flex items-center gap-4">
            <button
              className="p-2.5 -ml-2 text-zinc-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-full transition-colors"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            >
              <Menu className="w-5 h-5" />
            </button>

            <div className="flex bg-black/40 rounded-full p-1 border border-white/5">
              <button
                className={`px-5 py-2 text-xs font-bold tracking-wide rounded-full transition-all flex items-center gap-2 ${mode === 'chat' ? 'bg-white/10 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
                onClick={() => handleModeChange('chat')}
              >
                <MessageSquare className="w-4 h-4" />
                CHAT
              </button>
              <button
                className={`px-5 py-2 text-xs font-bold tracking-wide rounded-full transition-all flex items-center gap-2 ${mode === 'rag' ? 'bg-orange-500 text-black shadow-[0_0_15px_rgba(242,125,38,0.3)]' : 'text-zinc-500 hover:text-zinc-300'}`}
                onClick={() => handleModeChange('rag')}
              >
                <Database className="w-4 h-4" />
                RAG
              </button>
            </div>
          </div>

          {/* Clear button */}
          <button
            className="text-xs text-zinc-500 hover:text-orange-400 transition-colors px-3 py-1.5 rounded-full hover:bg-white/5 border border-transparent hover:border-white/10"
            onClick={handleClear}
          >
            Limpiar chat
          </button>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 custom-scrollbar scroll-smooth relative">
          <div className="max-w-3xl mx-auto space-y-10 pb-20 relative z-10">
            {messages.map((msg) => (
              <motion.div
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                key={msg.id}
                className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {/* System messages */}
                {msg.role === 'system' ? (
                  <div className={`flex max-w-[85%] items-start gap-2 px-4 py-2.5 rounded-lg text-xs font-medium ${msg.isError ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-white/5 text-zinc-400 border border-white/5'}`}>
                    {msg.isError
                      ? <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                      : <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0 text-green-400" />
                    }
                    <MarkdownContent text={msg.content} compact />
                  </div>
                ) : (
                  <>
                    <div className={`flex flex-col gap-2 max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                      {/* Meta label + copy */}
                      <div className="flex items-center gap-2 px-2 group/meta">
                        <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                          {msg.role === 'user' ? 'Tú' : 'MonkeyGrab'}
                        </span>
                        {msg.role === 'assistant' && (
                          <span className={`text-[9px] px-2 py-0.5 rounded-full uppercase tracking-widest font-bold ${msg.mode === 'rag' ? 'bg-orange-500/10 text-orange-400 border border-orange-500/20' : 'bg-white/10 text-zinc-400 border border-white/5'}`}>
                            {msg.mode}
                          </span>
                        )}
                        <button
                          onClick={() => handleCopyMessage(msg)}
                          className="p-1.5 rounded-full text-zinc-500 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/20 transition-all opacity-60 group-hover/meta:opacity-100"
                          title="Copiar mensaje"
                        >
                          {copiedId === msg.id ? (
                            <Check className="w-3.5 h-3.5 text-green-400" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
                      </div>

                      {/* Message bubble */}
                      <div className={`p-5 text-[15px] leading-relaxed shadow-lg backdrop-blur-md ${
                        msg.role === 'user'
                          ? 'bg-white/10 text-zinc-200 border border-white/10 rounded-xl rounded-tr-md font-medium'
                          : msg.isError
                            ? 'bg-red-500/10 text-red-300 border border-red-500/20 rounded-xl rounded-tl-md'
                            : 'bg-white/5 text-zinc-200 border border-white/10 rounded-xl rounded-tl-md'
                      }`}>
                        {msg.content ? (
                          <MarkdownContent text={msg.content} />
                        ) : msg.isStreaming ? (
                          <span className="inline-block w-2 h-5 bg-orange-400 rounded-sm animate-pulse" />
                        ) : null}
                        {msg.isStreaming && msg.content && (
                          <span className="inline-block w-2 h-5 bg-orange-400 rounded-sm animate-pulse ml-1 align-text-bottom" />
                        )}
                      </div>

                      {/* Citations */}
                      {msg.citations && msg.citations.length > 0 && (
                        <div className="mt-3 w-full space-y-3 pl-2">
                          <div className="flex flex-wrap gap-2">
                            {msg.citations.map((cite, i) => (
                              <div key={i} className="inline-flex max-w-full items-center gap-2 px-3 py-1.5 rounded-lg bg-black/40 border border-white/5 text-xs text-zinc-300 hover:bg-white/10 hover:border-white/10 transition-all group">
                                <FileText className="w-3.5 h-3.5 text-orange-400/70 group-hover:text-orange-400" />
                                <span className="font-medium truncate min-w-0">{cite.document}</span>
                                <span className="text-zinc-600">|</span>
                                <span className="text-zinc-400 shrink-0">p. {cite.pages.join(', ')}</span>
                              </div>
                            ))}
                          </div>
                          {msg.metrics && (
                            <div className="flex items-center gap-4 text-[11px] text-zinc-500 font-mono bg-black/20 inline-flex px-3 py-1.5 rounded-full border border-white/5">
                              <span className="flex items-center gap-1.5"><Search className="w-3.5 h-3.5 text-zinc-400" /> {msg.metrics.searchTime}</span>
                              <span className="w-1 h-1 rounded-full bg-zinc-700"></span>
                              <span className="flex items-center gap-1.5"><Layers className="w-3.5 h-3.5 text-zinc-400" /> {msg.metrics.chunks} fuentes</span>
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
        <div className="p-6 bg-gradient-to-t from-[#050505] via-[#050505]/90 to-transparent absolute bottom-0 left-0 right-0 z-20">
          <div className="max-w-3xl mx-auto relative">
            <div className="relative flex items-end gap-3 bg-black/60 backdrop-blur-xl border border-white/10 rounded-xl p-2.5 shadow-2xl focus-within:border-orange-500/50 focus-within:ring-4 focus-within:ring-orange-500/10 transition-all">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder={mode === 'rag' ? 'Pregunta sobre tus documentos…' : 'Escribe un mensaje…'}
                className="flex-1 max-h-48 min-h-[52px] bg-transparent border-none focus:ring-0 focus:outline-none resize-none py-3.5 px-4 text-[15px] text-white placeholder:text-zinc-500 custom-scrollbar font-medium"
                rows={1}
                disabled={isLoading}
              />

              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="p-3.5 bg-orange-500 text-black rounded-full hover:bg-orange-400 hover:scale-105 active:scale-95 disabled:opacity-50 disabled:bg-white/10 disabled:text-zinc-500 disabled:hover:scale-100 transition-all shadow-[0_0_20px_rgba(242,125,38,0.3)] disabled:shadow-none flex-shrink-0"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5 ml-0.5" />
                )}
              </button>
            </div>
            <div className="text-center mt-4 text-[11px] font-medium text-zinc-600 tracking-wide">
              MonkeyGrab · RAG local con Ollama · {mode === 'rag' ? 'Modo documento' : 'Modo conversación'}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// =============================================================================
// Toggle Component
// =============================================================================

function Toggle({ label, checked, onChange, desc }: { label: string; checked: boolean; onChange: () => void; desc: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className="w-full flex items-center justify-between gap-4 p-2 rounded-lg hover:bg-white/5 focus:outline-none focus:ring-2 focus:ring-orange-500/50 transition-colors text-left group"
      onClick={onChange}
    >
      <span className="flex-1">
        <span className="block text-sm text-zinc-200 font-medium group-hover:text-white transition-colors">{label}</span>
        <span className="block text-[11px] text-zinc-500 leading-snug mt-1">{desc}</span>
      </span>
      <span className={`relative inline-flex h-6 w-11 flex-shrink-0 items-center justify-center rounded-full transition-colors duration-300 ease-in-out ${checked ? 'bg-orange-500 shadow-[0_0_10px_rgba(242,125,38,0.4)]' : 'bg-white/10'}`}>
        <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-md transition duration-300 ease-in-out ${checked ? 'translate-x-2.5' : '-translate-x-2.5'}`} />
      </span>
    </button>
  );
}
