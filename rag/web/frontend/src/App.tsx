import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send, FileText, MessageSquare,
  Database,
  Search, Layers, FileUp, Menu, X,
  RefreshCw, Loader2, AlertCircle, CheckCircle2, Trash2,
  ChevronDown, ChevronRight, Copy, Check, Languages, Eye
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
  best_page?: number;
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
  imageIndexing: boolean;
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
// i18n
// =============================================================================

type Lang = 'es' | 'en' | 'ca';

const STRINGS = {
  es: {
    tabDocs: 'Documentos', tabPipeline: 'Pipeline RAG',
    corpusLabel: 'Corpus (PDFs / base vectorial)',
    corpusEs: 'Castellano — rag/docs/es',
    corpusCa: 'Català / valencià — rag/docs/ca',
    corpusEn: 'English — rag/docs/en',
    corpusError: 'No se pudo cambiar el corpus.',
    corpusConflict: 'Hay indexación en curso; espera un momento.',
    corpusConnError: '✗ Error de conexión al cambiar el corpus.',
    collection: 'Colección ({n} docs)', noDocs: 'No hay documentos indexados',
    deleteDoc: 'Eliminar documento', viewPdf: 'Ver PDF',
    indexingFile: 'Indexando {file}', reindexingStatus: 'Re-indexación en curso...',
    section1: '1. Indexación',
    labelContextual: 'Recuperación contextual', descContextual: 'Enriquece fragmentos con LLM',
    section2: '2. Recuperación',
    labelHybrid: 'Búsqueda híbrida', descHybrid: 'Semántica + BM25',
    labelQueryDecomp: 'Descomposición de consultas', descQueryDecomp: 'Subconsultas con LLM auxiliar',
    labelImageIndex: 'Indexado de imágenes', descImageIndex: 'Descripciones con visión',
    section3: '3. Ranking y contexto',
    labelReranker: 'Reordenador cross-encoder', descReranker: 'Reordenamiento de precisión',
    labelExpandContext: 'Expandir contexto', descExpandContext: 'Añade fragmentos adyacentes',
    labelOptimizeContext: 'Optimizar contexto', descOptimizeContext: 'Limpia artefactos PDF',
    labelRecomp: 'Síntesis RECOMP', descRecomp: 'Sintetiza contexto con LLM',
    section4: '4. Reindexación',
    reindexHint: 'Ajusta las opciones arriba y reindexa para aplicarlas. Opcionalmente añade PDFs nuevos.',
    addPdfs: 'Añadir PDFs', remove: 'Quitar', reindexBtn: 'Reindexar',
    fragments: '{n} fragmentos', ollamaStatus: 'Ollama Local',
    clearChat: 'Limpiar chat', youLabel: 'Tú',
    sources: '{n} fuentes', copyMsg: 'Copiar mensaje',
    placeholderRag: 'Pregunta sobre tus documentos…', placeholderChat: 'Escribe un mensaje…',
    footerMode: 'Modo documento', footerModeChat: 'Modo conversación',
    indexingTitle: 'Se están indexando los documentos',
    indexingHint: 'Puede tardar unos minutos dependiendo de tu hardware.',
    processing: 'Procesando:', fileUnit: 'archivo', fileUnitPlural: 'archivos',
    autoRefresh: 'La página se actualizará automáticamente al terminar.',
    retry: 'Reintentar', connErrorTitle: 'Error de conexión',
    connecting: 'Conectando con MonkeyGrab…',
    addingMsg: '⟳ Añadiendo {n} PDF(s) y re-indexando…',
    reindexingMsg: '⟳ Re-indexando con ajustes actuales…',
    reindexDone: '✓ Re-indexación completada: {total} fragmentos, {docs} documentos.',
    reindexError: '✗ Error: {error}', reindexConnError: '✗ Error de conexión.',
    confirmDelete: '¿Eliminar "{name}"? Se borrará del índice y del disco.',
    docDeleted: '✓ Documento **{name}** eliminado.',
    deleteError: '✗ Error al eliminar: {error}',
    deleteConnError: '✗ Error de conexión al eliminar.',
    unknownError: 'Error desconocido', queryError: 'Error en la consulta',
    modelError: 'Error en la respuesta del modelo',
    connClosed: 'La conexión se cerró antes de completar la respuesta.',
    settingsSaveError: 'No se pudo guardar el ajuste.',
    noServer: 'No se pudo conectar con el servidor. ¿Está Flask ejecutándose?',
    retryError: 'Error al reintentar',
    indexingFailed: 'La indexación no pudo completarse.',
    historyCleared: 'Historial limpiado.',
    appFooter: 'RAG local con Ollama',
    noResults: 'No se encontró información relevante en los documentos.',
  },
  en: {
    tabDocs: 'Documents', tabPipeline: 'RAG Pipeline',
    corpusLabel: 'Corpus (PDFs / vector DB)',
    corpusEs: 'Spanish — rag/docs/es',
    corpusCa: 'Catalan — rag/docs/ca',
    corpusEn: 'English — rag/docs/en',
    corpusError: 'Could not switch corpus.',
    corpusConflict: 'Indexing in progress; please wait.',
    corpusConnError: '✗ Connection error while switching corpus.',
    collection: 'Collection ({n} docs)', noDocs: 'No documents indexed',
    deleteDoc: 'Delete document', viewPdf: 'View PDF',
    indexingFile: 'Indexing {file}', reindexingStatus: 'Re-indexing in progress...',
    section1: '1. Indexing',
    labelContextual: 'Contextual Retrieval', descContextual: 'Enrich chunks with LLM',
    section2: '2. Retrieval',
    labelHybrid: 'Hybrid Search', descHybrid: 'Semantic + BM25',
    labelQueryDecomp: 'Query Decomposition', descQueryDecomp: 'Sub-queries via auxiliary LLM',
    labelImageIndex: 'Image indexing', descImageIndex: 'Vision captions',
    section3: '3. Ranking & Context',
    labelReranker: 'Cross-Encoder Reranker', descReranker: 'Precision reordering',
    labelExpandContext: 'Expand Context', descExpandContext: 'Add adjacent chunks',
    labelOptimizeContext: 'Optimize Context', descOptimizeContext: 'Clean PDF artifacts',
    labelRecomp: 'RECOMP Synthesis', descRecomp: 'Synthesize context with LLM',
    section4: '4. Re-index',
    reindexHint: 'Adjust settings above and re-index to apply. Optionally add new PDFs.',
    addPdfs: 'Add PDFs', remove: 'Remove', reindexBtn: 'Re-index',
    fragments: '{n} fragments', ollamaStatus: 'Ollama Local',
    clearChat: 'Clear chat', youLabel: 'You',
    sources: '{n} sources', copyMsg: 'Copy message',
    placeholderRag: 'Ask about your documents…', placeholderChat: 'Type a message…',
    footerMode: 'Document mode', footerModeChat: 'Conversation mode',
    indexingTitle: 'Indexing documents',
    indexingHint: 'This may take a few minutes depending on your hardware.',
    processing: 'Processing:', fileUnit: 'file', fileUnitPlural: 'files',
    autoRefresh: 'The page will refresh automatically when done.',
    retry: 'Retry', connErrorTitle: 'Connection error',
    connecting: 'Connecting to MonkeyGrab…',
    addingMsg: '⟳ Adding {n} PDF(s) and re-indexing…',
    reindexingMsg: '⟳ Re-indexing with current settings…',
    reindexDone: '✓ Re-indexing complete: {total} fragments, {docs} documents.',
    reindexError: '✗ Error: {error}', reindexConnError: '✗ Connection error.',
    confirmDelete: 'Delete "{name}"? It will be removed from the index and disk.',
    docDeleted: '✓ Document **{name}** deleted.',
    deleteError: '✗ Error deleting: {error}',
    deleteConnError: '✗ Connection error while deleting.',
    unknownError: 'Unknown error', queryError: 'Query error',
    modelError: 'Error in model response',
    connClosed: 'The connection closed before the response was complete.',
    settingsSaveError: 'Could not save setting.',
    noServer: 'Could not connect to the server. Is Flask running?',
    retryError: 'Retry failed',
    indexingFailed: 'Indexing could not be completed.',
    historyCleared: 'History cleared.',
    appFooter: 'Local RAG with Ollama',
    noResults: 'No relevant information found in the documents.',
  },
  ca: {
    tabDocs: 'Documents', tabPipeline: 'Pipeline RAG',
    corpusLabel: 'Corpus (PDFs / base vectorial)',
    corpusEs: 'Castellà — rag/docs/es',
    corpusCa: 'Català / valencià — rag/docs/ca',
    corpusEn: 'Anglès — rag/docs/en',
    corpusError: 'No s\'ha pogut canviar el corpus.',
    corpusConflict: 'Hi ha indexació en curs; espera un moment.',
    corpusConnError: '✗ Error de connexió en canviar el corpus.',
    collection: "Col·lecció ({n} docs)", noDocs: 'No hi ha documents indexats',
    deleteDoc: 'Eliminar document', viewPdf: 'Veure PDF',
    indexingFile: 'Indexant {file}', reindexingStatus: 'Re-indexació en curs...',
    section1: '1. Indexació',
    labelContextual: 'Recuperació contextual', descContextual: 'Enriqueix fragments amb LLM',
    section2: '2. Recuperació',
    labelHybrid: 'Cerca híbrida', descHybrid: 'Semàntica + BM25',
    labelQueryDecomp: 'Descomposició de consultes', descQueryDecomp: 'Sub-consultes amb LLM auxiliar',
    labelImageIndex: 'Indexat d\'imatges', descImageIndex: 'Descripcions amb visió',
    section3: '3. Rànquing i context',
    labelReranker: 'Reordenador cross-encoder', descReranker: 'Reordenament de precisió',
    labelExpandContext: 'Expandir context', descExpandContext: 'Afig fragments adjacents',
    labelOptimizeContext: 'Optimitzar context', descOptimizeContext: 'Neteja artefactes PDF',
    labelRecomp: 'Síntesi RECOMP', descRecomp: 'Sintetitza context amb LLM',
    section4: '4. Re-indexació',
    reindexHint: "Ajusta les opcions dalt i torna a indexar per a aplicar-les. Opcionalment afig PDFs nous.",
    addPdfs: 'Afegir PDFs', remove: 'Llevar', reindexBtn: 'Re-indexar',
    fragments: '{n} fragments', ollamaStatus: 'Ollama Local',
    clearChat: 'Netejar xat', youLabel: 'Tu',
    sources: '{n} fonts', copyMsg: 'Copiar missatge',
    placeholderRag: 'Pregunta sobre els teus documents…', placeholderChat: 'Escriu un missatge…',
    footerMode: 'Mode document', footerModeChat: 'Mode conversa',
    indexingTitle: "S'estan indexant els documents",
    indexingHint: 'Pot tardar uns minuts depenent del teu maquinari.',
    processing: 'Processant:', fileUnit: 'arxiu', fileUnitPlural: 'arxius',
    autoRefresh: "La pàgina s'actualitzarà automàticament en acabar.",
    retry: 'Reintentar', connErrorTitle: 'Error de connexió',
    connecting: 'Connectant amb MonkeyGrab…',
    addingMsg: '⟳ Afegint {n} PDF(s) i re-indexant…',
    reindexingMsg: '⟳ Re-indexant amb ajustos actuals…',
    reindexDone: '✓ Re-indexació completada: {total} fragments, {docs} documents.',
    reindexError: '✗ Error: {error}', reindexConnError: '✗ Error de connexió.',
    confirmDelete: 'Eliminar "{name}"? S\'esborrarà de l\'índex i del disc.',
    docDeleted: '✓ Document **{name}** eliminat.',
    deleteError: '✗ Error en eliminar: {error}',
    deleteConnError: '✗ Error de connexió en eliminar.',
    unknownError: 'Error desconegut', queryError: 'Error en la consulta',
    modelError: 'Error en la resposta del model',
    connClosed: 'La connexió es va tancar abans de completar la resposta.',
    settingsSaveError: "No s'ha pogut guardar l'ajust.",
    noServer: "No s'ha pogut connectar amb el servidor. Està Flask executant-se?",
    retryError: 'Error en reintentar',
    indexingFailed: "La indexació no s'ha pogut completar.",
    historyCleared: 'Historial netejat.',
    appFooter: 'RAG local amb Ollama',
    noResults: "No s'ha trobat informació rellevant als documents.",
  },
} as const;

const LANG_OPTIONS: Array<{ code: Lang; label: string }> = [
  { code: 'es', label: 'ES' },
  { code: 'en', label: 'EN' },
  { code: 'ca', label: 'VAL' },
];

function normalizeLang(value: string | null): Lang {
  return value === 'en' || value === 'ca' || value === 'es' ? value : 'es';
}

function fill(tpl: string, vars: Record<string, string | number>): string {
  return tpl.replace(/\{(\w+)\}/g, (_, k) => String(vars[k] ?? ''));
}

type CorpusPreset = 'es' | 'ca' | 'en';

/** Align UI with backend ``docs_folder`` / ``corpus_preset`` (defaults to ES). */
function presetFromInit(initData: { corpus_preset?: string | null; docs_folder?: string }): CorpusPreset {
  const p = initData.corpus_preset;
  if (p === 'es' || p === 'ca' || p === 'en') return p;
  const folder = initData.docs_folder;
  if (folder) {
    const norm = folder.replace(/\\/g, '/').replace(/\/+$/, '');
    const parts = norm.split('/').filter(Boolean);
    const base = parts[parts.length - 1] ?? '';
    if (base === 'es' || base === 'ca' || base === 'en') return base;
  }
  return 'es';
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

  setCorpus: (preset: CorpusPreset) =>
    fetch(`${API_BASE}/corpus`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset }),
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
// Small safe Markdown renderer
// =============================================================================

const MathInline: React.FC<{ tex: string }> = ({ tex }) => {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: false });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
};

const MathBlock: React.FC<{ tex: string }> = ({ tex }) => {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: true });
  return <div className="overflow-x-auto my-2" dangerouslySetInnerHTML={{ __html: html }} />;
};

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
  const [isReindexing, setIsReindexing] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [pendingReindexFiles, setPendingReindexFiles] = useState<File[]>([]);
  const [corpusPreset, setCorpusPreset] = useState<CorpusPreset>('es');
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
    imageIndexing: true,
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
  const [pdfViewer, setPdfViewer] = useState<{ doc: string; page: number } | null>(null);

  const openPdf = useCallback((doc: string, page = 1) => {
    setPdfViewer({ doc, page });
  }, []);

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
          setCorpusPreset(presetFromInit(initData));
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
      setSettingsError(result?.error || T.settingsSaveError);
    }
  }, [settings, lang]);

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
      throw new Error(status.message || status.error || T.indexingFailed);
    }
  }, [lang]);

  // ---- Reindex (full, con ajustes del pipeline) ----
  const handleReindex = useCallback(async () => {
    if (isReindexing) return;
    setIsReindexing(true);
    const fileList = [...pendingReindexFiles];
    setPendingReindexFiles([]);
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      role: 'system',
      content: fileList.length ? fill(T.addingMsg, { n: fileList.length }) : T.reindexingMsg,
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
          content: fill(T.reindexDone, { total: finalStatus.total_fragments || 0, docs: (finalStatus.documents || []).length }),
          mode,
        }]);
      } else if (result.ok) {
        setTotalFragments(result.total_fragments || 0);
        setDocuments(result.documents || []);
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: fill(T.reindexDone, { total: result.total_fragments || 0, docs: (result.documents || []).length }),
          mode,
        }]);
      } else {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: fill(T.reindexError, { error: result.error }),
          mode,
          isError: true,
        }]);
      }
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: T.reindexConnError,
        mode,
        isError: true,
      }]);
    } finally {
      setIsReindexing(false);
      if (reindexFileInputRef.current) reindexFileInputRef.current.value = '';
    }
  }, [mode, isReindexing, pendingReindexFiles, waitForIndexingToFinish, lang]);

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

  const handleCorpusChange = useCallback(async (next: CorpusPreset) => {
    if (next === corpusPreset) return;
    const prev = corpusPreset;
    const S = STRINGS[lang];
    setCorpusPreset(next);
    try {
      const res = await api.setCorpus(next);
      if (!res.ok) {
        setCorpusPreset(prev);
        const err =
          res.error === 'indexing_in_progress'
            ? S.corpusConflict
            : `${S.corpusError} (${String(res.error ?? '')})`;
        setMessages(p => [...p, {
          id: Date.now().toString(),
          role: 'system',
          content: err,
          mode,
          isError: true,
        }]);
        return;
      }
      if (res.indexing) {
        setIndexingError(null);
        setIndexingProgress(null);
        setIsIndexing(true);
        setRetryTrigger(t => t + 1);
        return;
      }
      setDocuments(res.documents || []);
      setTotalFragments(res.total_fragments || 0);
    } catch {
      setCorpusPreset(prev);
      setMessages(p => [...p, {
        id: Date.now().toString(),
        role: 'system',
        content: S.corpusConnError,
        mode,
        isError: true,
      }]);
    }
  }, [corpusPreset, mode, lang]);

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
  }, [input, mode, isLoading, lang]);

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
      content: T.historyCleared,
      mode,
    }]);
  }, [mode, lang]);

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
        setCorpusPreset(presetFromInit(result));
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
      <div className="flex h-screen items-center justify-center bg-[#050505] text-zinc-300 p-4">
        <div className="glass-panel rounded-3xl p-10 max-w-md text-center space-y-4">
          <Loader2 className="w-12 h-12 text-orange-500 animate-spin mx-auto" />
          <h2 className="text-xl font-semibold text-white">
            {T.indexingTitle}
          </h2>
          <p className="text-zinc-400 text-sm">
            {T.indexingHint}
          </p>
          {indexingError && showRetry && (
            <p className="text-amber-400 text-sm">{indexingError}</p>
          )}
          {indexingProgress ? (
            <div className="space-y-1">
              <p className="text-zinc-300 text-sm font-medium">
                {T.processing} <span className="text-orange-500">{indexingProgress.file}</span>
              </p>
              <p className="text-zinc-500 text-xs">
                {indexingProgress.file_index} / {indexingProgress.total_files} {indexingProgress.total_files !== 1 ? T.fileUnitPlural : T.fileUnit}
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
              {T.autoRefresh}
            </p>
          )}
          {showRetry && (
            <button
              className="px-6 py-2 bg-orange-500 text-black rounded-full font-semibold hover:bg-orange-400 transition-colors"
              onClick={handleRetry}
            >
              {T.retry}
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
          <h2 className="text-xl font-semibold text-white">{T.connErrorTitle}</h2>
          <p className="text-zinc-400 text-sm">{initError}</p>
          <button
            className="px-6 py-2 bg-orange-500 text-black rounded-full font-semibold hover:bg-orange-400 transition-colors"
            onClick={handleRetry}
          >
            {T.retry}
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
          <p className="text-zinc-500 text-sm">{T.connecting}</p>
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
        className={`fixed md:relative z-50 h-[calc(100vh-16px)] md:h-full w-[320px] glass-panel rounded-xl flex flex-col transition-transform duration-300 ease-in-out shadow-2xl ${isSidebarOpen ? 'translate-x-2 md:translate-x-0' : '-translate-x-[120%] md:translate-x-0 md:w-0 md:opacity-0 md:overflow-hidden md:ml-[-16px]'}`}
      >
        {/* Sidebar Header */}
        <div className="p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="MonkeyGrab" className="w-10 h-10 rounded-full object-cover flex-shrink-0" />
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
              {T.tabDocs}
            </button>
            <button
              className={`flex-1 py-2 text-xs font-semibold rounded-full transition-all ${activeTab === 'settings' ? 'bg-white/10 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
              onClick={() => setActiveTab('settings')}
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
                  <div className="flex items-center gap-2 pl-2 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                    <Database className="h-3 w-3 text-orange-400/70" />
                    {T.corpusLabel}
                  </div>
                  <div
                    role="radiogroup"
                    aria-label={T.corpusLabel}
                    className={`rounded-2xl border border-white/10 bg-black/30 p-1.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] ${isReindexing || isLoading ? 'opacity-50' : ''}`}
                  >
                    <div className="grid grid-cols-3 gap-1.5">
                      {[
                        { preset: 'es' as CorpusPreset, label: 'ES', detail: 'rag/docs/es', title: T.corpusEs },
                        { preset: 'ca' as CorpusPreset, label: 'VAL', detail: 'rag/docs/ca', title: T.corpusCa },
                        { preset: 'en' as CorpusPreset, label: 'EN', detail: 'rag/docs/en', title: T.corpusEn },
                      ].map(option => {
                        const isActive = corpusPreset === option.preset;
                        return (
                          <button
                            key={option.preset}
                            type="button"
                            role="radio"
                            aria-checked={isActive}
                            title={option.title}
                            className={`min-w-0 rounded-xl border px-2.5 py-2 text-left transition-all focus:outline-none focus:ring-2 focus:ring-orange-500/40 disabled:cursor-not-allowed ${isActive
                              ? 'border-orange-500/50 bg-orange-500/15 text-white shadow-[0_0_18px_rgba(242,125,38,0.16)]'
                              : 'border-transparent bg-white/[0.03] text-zinc-500 hover:border-white/10 hover:bg-white/[0.06] hover:text-zinc-300'
                              }`}
                            onClick={() => handleCorpusChange(option.preset)}
                            disabled={isReindexing || isLoading}
                          >
                            <span className="flex items-center justify-between gap-1">
                              <span className="truncate text-xs font-bold tracking-wide">{option.label}</span>
                              {isActive && <Check className="h-3 w-3 flex-shrink-0 text-orange-300" />}
                            </span>
                            <span className={`mt-1 block truncate font-mono text-[9px] ${isActive ? 'text-orange-200/80' : 'text-zinc-600'}`}>
                              {option.detail}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Documents list */}
                <div className="space-y-3">
                  <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest pl-2">
                    {fill(T.collection, { n: documents.length })}
                  </div>

                  <div className="space-y-2">
                    {documents.length === 0 ? (
                      <p className="text-xs text-zinc-600 text-center py-4">{T.noDocs}</p>
                    ) : (
                      documents.map((doc, i) => (
                        <div key={i} className="group flex items-center gap-3 p-3.5 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-white/10 transition-all">
                          <div className="w-8 h-8 rounded-full bg-black/30 flex items-center justify-center flex-shrink-0">
                            <FileText className="w-4 h-4 text-orange-400/80" />
                          </div>
                          <span className="text-sm text-zinc-300 group-hover:text-white truncate font-medium flex-1 min-w-0">{doc}</span>
                          <button
                            className="opacity-0 group-hover:opacity-100 p-1.5 rounded-full text-zinc-500 hover:text-orange-400 hover:bg-orange-500/10 transition-all flex-shrink-0"
                            onClick={() => openPdf(doc)}
                            title={T.viewPdf}
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            className="opacity-0 group-hover:opacity-100 p-1.5 rounded-full text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-all flex-shrink-0 disabled:opacity-50"
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
                          <span className="truncate">{fill(T.indexingFile, { file: indexingProgress.file })}</span>
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
                      <span>{T.reindexingStatus}</span>
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
                    {T.section1}
                  </button>
                  {openSections.indexacion && (
                    <div className="p-2 pt-0 space-y-1 bg-white/[0.02]">
                      <Toggle label={T.labelContextual} checked={settings.contextualRetrieval} onChange={() => toggleSetting('contextualRetrieval')} desc={T.descContextual} />
                      <Toggle label={T.labelImageIndex} checked={settings.imageIndexing} onChange={() => toggleSetting('imageIndexing')} desc={T.descImageIndex} />
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
                    {T.section2}
                  </button>
                  {openSections.recuperacion && (
                    <div className="p-2 pt-0 space-y-1 bg-white/[0.02]">
                      <Toggle label={T.labelHybrid} checked={settings.hybridSearch} onChange={() => toggleSetting('hybridSearch')} desc={T.descHybrid} />
                      <Toggle label={T.labelQueryDecomp} checked={settings.queryDecomposition} onChange={() => toggleSetting('queryDecomposition')} desc={T.descQueryDecomp} />
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
                    {T.section3}
                  </button>
                  {openSections.ranking && (
                    <div className="p-2 pt-0 space-y-1 bg-white/[0.02]">
                      <Toggle label={T.labelReranker} checked={settings.reranker} onChange={() => toggleSetting('reranker')} desc={T.descReranker} />
                      <Toggle label={T.labelExpandContext} checked={settings.expandContext} onChange={() => toggleSetting('expandContext')} desc={T.descExpandContext} />
                      <Toggle label={T.labelOptimizeContext} checked={settings.optimizeContext} onChange={() => toggleSetting('optimizeContext')} desc={T.descOptimizeContext} />
                      <Toggle label={T.labelRecomp} checked={settings.recompSynthesis} onChange={() => toggleSetting('recompSynthesis')} desc={T.descRecomp} />
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
                    {T.section4}
                  </button>
                  {openSections.reindexacion && (
                    <div className="p-3 pt-0 space-y-3 bg-orange-500/5">
                      <p className="text-xs text-zinc-500">
                        {T.reindexHint}
                      </p>
                      <div className="flex gap-2">
                        <button
                          className="flex-1 py-3 px-4 rounded-2xl border border-dashed border-white/20 text-zinc-400 hover:text-white hover:border-orange-500/50 hover:bg-orange-500/5 transition-all flex flex-col items-center justify-center gap-1 text-sm group disabled:opacity-50"
                          onClick={() => reindexFileInputRef.current?.click()}
                          disabled={isReindexing}
                        >
                          <FileUp className="w-5 h-5 group-hover:text-orange-400" />
                          <span className="font-medium">
                            {pendingReindexFiles.length ? `${pendingReindexFiles.length} PDF(s)` : T.addPdfs}
                          </span>
                        </button>
                        {pendingReindexFiles.length > 0 && (
                          <button
                            className="px-3 rounded-2xl text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-all"
                            onClick={() => { setPendingReindexFiles([]); if (reindexFileInputRef.current) reindexFileInputRef.current.value = ''; }}
                            title={T.remove}
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
                        {T.reindexBtn}
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
          <span className="font-mono text-[10px] tracking-wider">{fill(T.fragments, { n: totalFragments })}</span>
          <div className="flex items-center gap-2 bg-white/5 px-2.5 py-1 rounded-full border border-white/10">
            <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.8)]" />
            <span className="font-medium text-zinc-300">{T.ollamaStatus}</span>
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

          <div className="flex items-center gap-3">
            <LanguageToggle lang={lang} setLang={setLang} />
            {/* Clear button */}
            <button
              className="text-xs text-zinc-500 hover:text-orange-400 transition-colors px-3 py-1.5 rounded-full hover:bg-white/5 border border-transparent hover:border-white/10"
              onClick={handleClear}
            >
              {T.clearChat}
            </button>
          </div>
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
                          {msg.role === 'user' ? T.youLabel : 'MonkeyGrab'}
                        </span>
                        {msg.role === 'assistant' && (
                          <span className={`text-[9px] px-2 py-0.5 rounded-full uppercase tracking-widest font-bold ${msg.mode === 'rag' ? 'bg-orange-500/10 text-orange-400 border border-orange-500/20' : 'bg-white/10 text-zinc-400 border border-white/5'}`}>
                            {msg.mode}
                          </span>
                        )}
                        <button
                          onClick={() => handleCopyMessage(msg)}
                          className="p-1.5 rounded-full text-zinc-500 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/20 transition-all opacity-60 group-hover/meta:opacity-100"
                          title={T.copyMsg}
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
                              <button
                                key={i}
                                className="inline-flex max-w-full items-center gap-2 px-3 py-1.5 rounded-lg bg-black/40 border border-white/5 text-xs text-zinc-300 hover:bg-white/10 hover:border-orange-500/30 hover:text-orange-300 transition-all group cursor-pointer"
                                onClick={() => openPdf(cite.document, cite.best_page ?? cite.pages[0] ?? 1)}
                                title={T.viewPdf}
                              >
                                <FileText className="w-3.5 h-3.5 text-orange-400/70 group-hover:text-orange-400" />
                                <span className="font-medium truncate min-w-0">{cite.document}</span>
                                <span className="text-zinc-600">|</span>
                                <span className="text-zinc-400 shrink-0">p. {cite.pages.join(', ')}</span>
                              </button>
                            ))}
                          </div>
                          {msg.metrics && (
                            <div className="flex items-center gap-4 text-[11px] text-zinc-500 font-mono bg-black/20 inline-flex px-3 py-1.5 rounded-full border border-white/5">
                              <span className="flex items-center gap-1.5"><Search className="w-3.5 h-3.5 text-zinc-400" /> {msg.metrics.searchTime}</span>
                              <span className="w-1 h-1 rounded-full bg-zinc-700"></span>
                              <span className="flex items-center gap-1.5"><Layers className="w-3.5 h-3.5 text-zinc-400" /> {fill(T.sources, { n: msg.metrics.chunks })}</span>
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

        {/* PDF Viewer Modal */}
        {pdfViewer && (
          <PdfViewerModal
            doc={pdfViewer.doc}
            page={pdfViewer.page}
            onClose={() => setPdfViewer(null)}
          />
        )}

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
                placeholder={mode === 'rag' ? T.placeholderRag : T.placeholderChat}
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
              MonkeyGrab · {T.appFooter} · {mode === 'rag' ? T.footerMode : T.footerModeChat}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// =============================================================================
// PDF Viewer Modal
// =============================================================================

function PdfViewerModal({ doc, page, onClose }: { doc: string; page: number; onClose: () => void }) {
  const src = `/api/pdf/${encodeURIComponent(doc)}#page=${page}`;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black/95 backdrop-blur-sm">
      <div className="flex items-center justify-between px-4 py-3 bg-[#0a0a0a] border-b border-white/10 flex-shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <FileText className="w-4 h-4 text-orange-400 flex-shrink-0" />
          <span className="text-sm font-medium text-zinc-200 truncate">{doc}</span>
          {page > 1 && (
            <span className="text-xs text-zinc-500 flex-shrink-0 ml-1">— p. {page}</span>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-2 rounded-lg text-zinc-500 hover:text-white hover:bg-white/10 transition-all flex-shrink-0"
          title="Cerrar (Esc)"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      <iframe
        src={src}
        className="flex-1 w-full border-none bg-zinc-900"
        title={doc}
      />
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

function LanguageToggle({ lang, setLang }: { lang: Lang; setLang: (lang: Lang) => void }) {
  return (
    <div className="flex items-center gap-1 rounded-full border border-white/10 bg-black/30 p-1">
      <Languages className="ml-2 h-3.5 w-3.5 text-zinc-500" />
      {LANG_OPTIONS.map(option => (
        <button
          key={option.code}
          type="button"
          className={`rounded-full px-2.5 py-1 text-[10px] font-bold tracking-wide transition-all ${
            lang === option.code
              ? 'bg-orange-500 text-black shadow-[0_0_12px_rgba(242,125,38,0.25)]'
              : 'text-zinc-500 hover:bg-white/5 hover:text-zinc-300'
          }`}
          onClick={() => setLang(option.code)}
          aria-pressed={lang === option.code}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
