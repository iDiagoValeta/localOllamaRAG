import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send, FileText, Database, Ollama,
  Search, Layers, FileUp, Menu, X,
  RefreshCw, Loader2, AlertCircle, CheckCircle2, Trash2,
  ChevronDown, Copy, Check, Languages, Eye,
  Power, Sun, Moon
} from './lib/icons';
import { getStoredTheme, setTheme, type Theme } from './lib/theme';
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

interface OllamaModel {
  name: string;
  size?: number;
  family?: string;
  parameter_size?: string;
  capabilities?: string[];
  embedding?: boolean;
  vision?: boolean;
}

type ModelRole = 'rag' | 'chat' | 'contextual' | 'recomp';
type ModelRoles = Record<ModelRole, string>;

interface VectorStore {
  name: string;
  label: string;
  docs_folder: string;
  pdf_count: number;
  indexed: boolean;
  active: boolean;
  fragments: number | null;
}

// =============================================================================
// i18n
// =============================================================================

type Lang = 'es' | 'en' | 'ca';

const STRINGS = {
  es: {
    tabDocs: 'Documentos', tabPipeline: 'Pipeline RAG',
    corpusError: 'No se pudo cambiar el almacén.',
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
    labelImageIndex: 'Indexado de imágenes', descImageIndex: 'Recuperación visual directa',
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
    themeLight: 'Modo claro', themeDark: 'Modo oscuro', close: 'Cerrar',
    sources: '{n} fuentes', copyMsg: 'Copiar mensaje',
    placeholderRag: 'Pregunta sobre tus documentos…', placeholderChat: 'Escribe un mensaje…',
    footerMode: 'Modo documento', footerModeChat: 'Modo conversación',
    indexingTitle: 'Se están indexando los documentos',
    indexingHint: 'Puede tardar unos minutos dependiendo de tu hardware.',
    processing: 'Procesando:', fileUnit: 'archivo', fileUnitPlural: 'archivos',
    autoRefresh: 'La página se actualizará automáticamente al terminar.',
    retry: 'Reintentar', connErrorTitle: 'Error de conexión',
    connecting: 'Conectando con MonkeyGrab…',
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
    noResults: 'No se encontró información relevante en los documentos.',
    tabModels: 'Modelos',
    storesLabel: 'Almacén vectorial',
    storeNotIndexed: 'sin indexar',
    storeEmptyHint: 'Almacén vacío. Sube PDFs y reindexa para activarlo.',
    fingerprintStaleWarning: 'El índice guardado no coincide con la configuración activa, pero la app sigue funcionando con él. Reindexa cuando quieras: puede tardar una hora o más.',
    modelsRoles: 'Roles de modelo',
    ollamaTitle: 'Servidor Ollama',
    ollamaOnline: 'En ejecución',
    ollamaOffline: 'Detenido',
    ollamaStartBtn: 'Arrancar Ollama',
    ollamaStarting: 'Arrancando…',
    ollamaStartFailed: 'No se pudo arrancar Ollama.',
    ollamaNotFound: 'No se encontró el ejecutable de Ollama en el PATH.',
    refreshModels: 'Actualizar lista',
    noModels: 'No hay modelos instalados. Descárgalos con «ollama pull».',
    roleRag: 'Generador RAG', descRoleRag: 'Respuesta final en modo documento',
    roleChat: 'Subconsultas', descRoleChat: 'Conversación y descomposición de consultas',
    roleContextual: 'Recuperación contextual', descRoleContextual: 'Enriquece fragmentos al indexar',
    roleRecomp: 'Síntesis RECOMP', descRoleRecomp: 'Resume el contexto antes de generar',
    modelSaveError: 'No se pudo cambiar el modelo.',
  },
  en: {
    tabDocs: 'Documents', tabPipeline: 'RAG Pipeline',
    corpusError: 'Could not switch store.',
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
    labelImageIndex: 'Image indexing', descImageIndex: 'Direct visual retrieval',
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
    themeLight: 'Light mode', themeDark: 'Dark mode', close: 'Close',
    sources: '{n} sources', copyMsg: 'Copy message',
    placeholderRag: 'Ask about your documents…', placeholderChat: 'Type a message…',
    footerMode: 'Document mode', footerModeChat: 'Conversation mode',
    indexingTitle: 'Indexing documents',
    indexingHint: 'This may take a few minutes depending on your hardware.',
    processing: 'Processing:', fileUnit: 'file', fileUnitPlural: 'files',
    autoRefresh: 'The page will refresh automatically when done.',
    retry: 'Retry', connErrorTitle: 'Connection error',
    connecting: 'Connecting to MonkeyGrab…',
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
    noResults: 'No relevant information found in the documents.',
    tabModels: 'Models',
    storesLabel: 'Vector store',
    storeNotIndexed: 'not indexed',
    storeEmptyHint: 'Empty store. Upload PDFs and re-index to activate it.',
    fingerprintStaleWarning: 'The saved index no longer matches the active configuration, but the app keeps working with it. Re-index whenever you like: it can take an hour or more.',
    modelsRoles: 'Model roles',
    ollamaTitle: 'Ollama server',
    ollamaOnline: 'Running',
    ollamaOffline: 'Stopped',
    ollamaStartBtn: 'Start Ollama',
    ollamaStarting: 'Starting…',
    ollamaStartFailed: 'Could not start Ollama.',
    ollamaNotFound: 'Ollama executable not found on PATH.',
    refreshModels: 'Refresh list',
    noModels: 'No models installed. Pull some with “ollama pull”.',
    roleRag: 'RAG generator', descRoleRag: 'Final answer in document mode',
    roleChat: 'Sub-queries', descRoleChat: 'Conversation and query decomposition',
    roleContextual: 'Contextual retrieval', descRoleContextual: 'Enriches chunks at indexing',
    roleRecomp: 'RECOMP synthesis', descRoleRecomp: 'Summarizes context before generation',
    modelSaveError: 'Could not change the model.',
  },
  ca: {
    tabDocs: 'Documents', tabPipeline: 'Pipeline RAG',
    corpusError: 'No s\'ha pogut canviar el magatzem.',
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
    labelImageIndex: 'Indexat d\'imatges', descImageIndex: 'Recuperació visual directa',
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
    themeLight: 'Mode clar', themeDark: 'Mode fosc', close: 'Tancar',
    sources: '{n} fonts', copyMsg: 'Copiar missatge',
    placeholderRag: 'Pregunta sobre els teus documents…', placeholderChat: 'Escriu un missatge…',
    footerMode: 'Mode document', footerModeChat: 'Mode conversa',
    indexingTitle: "S'estan indexant els documents",
    indexingHint: 'Pot tardar uns minuts depenent del teu maquinari.',
    processing: 'Processant:', fileUnit: 'arxiu', fileUnitPlural: 'arxius',
    autoRefresh: "La pàgina s'actualitzarà automàticament en acabar.",
    retry: 'Reintentar', connErrorTitle: 'Error de connexió',
    connecting: 'Connectant amb MonkeyGrab…',
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
    noResults: "No s'ha trobat informació rellevant als documents.",
    tabModels: 'Models',
    storesLabel: 'Magatzem vectorial',
    storeNotIndexed: 'sense indexar',
    storeEmptyHint: 'Magatzem buit. Puja PDFs i reindexa per a activar-lo.',
    fingerprintStaleWarning: "L'índex guardat no coincideix amb la configuració activa, però l'app continua funcionant amb ell. Reindexa quan vulgues: pot trigar una hora o més.",
    modelsRoles: 'Rols de model',
    ollamaTitle: 'Servidor Ollama',
    ollamaOnline: 'En execució',
    ollamaOffline: 'Aturat',
    ollamaStartBtn: 'Arrancar Ollama',
    ollamaStarting: 'Arrancant…',
    ollamaStartFailed: "No s'ha pogut arrancar Ollama.",
    ollamaNotFound: "No s'ha trobat l'executable d'Ollama al PATH.",
    refreshModels: 'Actualitzar llista',
    noModels: 'No hi ha models instal·lats. Descarrega\'n amb «ollama pull».',
    roleRag: 'Generador RAG', descRoleRag: 'Resposta final en mode document',
    roleChat: 'Subconsultes', descRoleChat: 'Conversa i descomposició de consultes',
    roleContextual: 'Recuperació contextual', descRoleContextual: 'Enriqueix fragments en indexar',
    roleRecomp: 'Síntesi RECOMP', descRoleRecomp: 'Resumeix el context abans de generar',
    modelSaveError: 'No s\'ha pogut canviar el model.',
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
      <div className="flex h-screen items-center justify-center bg-transparent text-zinc-300 p-4">
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
              className="px-6 py-2 bg-[var(--accent)] text-[var(--accent-contrast)] font-semibold hover:bg-[var(--accent-hover)] transition-colors"
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
      <div className="flex h-screen items-center justify-center bg-transparent text-zinc-300 p-4">
        <div className="glass-panel rounded-3xl p-10 max-w-md text-center space-y-4">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto" />
          <h2 className="text-xl font-semibold text-white">{T.connErrorTitle}</h2>
          <p className="text-zinc-400 text-sm">{initError}</p>
          <button
            className="px-6 py-2 bg-[var(--accent)] text-[var(--accent-contrast)] font-semibold hover:bg-[var(--accent-hover)] transition-colors"
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
      <div className="flex h-screen items-center justify-center bg-transparent text-zinc-300">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 text-orange-400 animate-spin" />
          <p className="text-zinc-500 text-sm">{T.connecting}</p>
        </div>
      </div>
    );
  }

  // ---- Overlay panels (rendered full-area in the main column) ----
  const renderModelsPanel = () => (
    <div className="mx-auto w-full max-w-4xl space-y-4">
      {/* Ollama server status */}
      <div className={`border p-3 ${ollamaStatus.running ? 'border-[var(--border)] bg-[var(--surface)]' : 'border-amber-500/30 bg-amber-500/10'}`}>
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <span className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${ollamaStatus.running ? 'bg-green-500' : 'bg-red-500'}`} />
            <div className="min-w-0">
              <div className="t-h3 text-[var(--text)]">{T.ollamaTitle}</div>
              <div className="truncate t-body-sm text-[var(--text-muted)]">
                {ollamaStatus.running ? T.ollamaOnline : T.ollamaOffline}
              </div>
            </div>
          </div>
          {ollamaStatus.running ? (
            <button
              type="button"
              className="flex-shrink-0 p-2 text-[var(--text-muted)] transition-all hover:bg-[var(--surface-2)] hover:text-[var(--accent)]"
              onClick={refreshOllama}
              title={T.refreshModels}
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          ) : (
            <button
              type="button"
              className="flex flex-shrink-0 items-center gap-1.5 border border-amber-500/40 bg-amber-500/15 px-3 py-1.5 text-[11px] font-semibold text-amber-500 transition-all hover:bg-amber-500/25 disabled:opacity-50"
              onClick={handleStartOllama}
              disabled={ollamaStarting}
            >
              {ollamaStarting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Power className="h-3.5 w-3.5" />}
              {ollamaStarting ? T.ollamaStarting : T.ollamaStartBtn}
            </button>
          )}
        </div>
      </div>

      {modelError && (
        <div className="border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-400">{modelError}</div>
      )}

      {/* Model role selectors — two columns in the wide overlay */}
      <div className="space-y-2">
        <div className="t-label text-[var(--text-muted)] pl-1">{T.modelsRoles}</div>
        {ollamaStatus.running && ollamaModels.length === 0 ? (
          <p className="px-1 py-3 text-xs text-[var(--text-muted)]">{T.noModels}</p>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2">
            {([
              { role: 'rag' as ModelRole, label: T.roleRag, desc: T.descRoleRag },
              { role: 'chat' as ModelRole, label: T.roleChat, desc: T.descRoleChat },
              { role: 'contextual' as ModelRole, label: T.roleContextual, desc: T.descRoleContextual },
              { role: 'recomp' as ModelRole, label: T.roleRecomp, desc: T.descRoleRecomp },
            ]).map(({ role, label, desc }) => {
              const current = modelRoles?.[role] ?? '';
              const names = ollamaModels.map(m => m.name);
              const options = current && !names.includes(current) ? [current, ...names] : names;
              return (
                <div key={role} className="border border-[var(--border)] bg-[var(--surface)] p-2.5">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="t-h3 text-[var(--accent)]">{label}</span>
                    {savingRole === role && <Loader2 className="h-3 w-3 animate-spin text-[var(--accent)]" />}
                  </div>
                  <p className="mb-2 t-body-sm text-[var(--text-muted)]">{desc}</p>
                  <ModelSelect
                    value={current}
                    options={options}
                    disabled={!ollamaStatus.running || savingRole !== null}
                    onChange={v => handleRoleChange(role, v)}
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );

  const renderPipelinePanel = () => (
    <div className="mx-auto w-full max-w-4xl space-y-4">
      {(settingsError || isReindexing) && (
        <div className={`border px-3 py-2 text-xs ${settingsError ? 'border-red-500/20 bg-red-500/10 text-red-400' : 'border-[var(--accent)]/30 bg-[var(--accent)]/10 text-[var(--accent)]'}`}>
          {settingsError ? (
            <span>{settingsError}</span>
          ) : indexingProgress ? (
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-3">
                <span className="truncate">{fill(T.indexingFile, { file: indexingProgress.file })}</span>
                <span className="font-mono text-[10px] text-[var(--accent)]">{indexingProgress.file_index}/{indexingProgress.total_files}</span>
              </div>
              <div className="h-1.5 bg-[var(--surface-2)] overflow-hidden">
                <div
                  className="h-full bg-[var(--accent)] transition-all duration-500"
                  style={{ width: `${Math.max(5, (indexingProgress.file_index / indexingProgress.total_files) * 100)}%` }}
                />
              </div>
            </div>
          ) : (
            <span>{T.reindexingStatus}</span>
          )}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <section className="flex flex-col border border-[var(--border)] bg-[var(--popover)] p-3">
          <h3 className="t-label mb-3 text-[var(--accent)]">{T.section1}</h3>
          <div className="space-y-0.5">
            <Toggle label={T.labelContextual} checked={settings.contextualRetrieval} onChange={() => toggleSetting('contextualRetrieval')} desc={T.descContextual} />
            <Toggle label={T.labelImageIndex} checked={settings.imageIndexing} onChange={() => toggleSetting('imageIndexing')} desc={T.descImageIndex} />
          </div>
        </section>

        <section className="flex flex-col border border-[var(--border)] bg-[var(--popover)] p-3">
          <h3 className="t-label mb-3 text-[var(--accent)]">{T.section2}</h3>
          <div className="space-y-0.5">
            <Toggle label={T.labelHybrid} checked={settings.hybridSearch} onChange={() => toggleSetting('hybridSearch')} desc={T.descHybrid} />
            <Toggle label={T.labelQueryDecomp} checked={settings.queryDecomposition} onChange={() => toggleSetting('queryDecomposition')} desc={T.descQueryDecomp} />
          </div>
        </section>

        <section className="flex flex-col border border-[var(--border)] bg-[var(--popover)] p-3 md:col-span-2">
          <h3 className="t-label mb-3 text-[var(--accent)]">{T.section3}</h3>
          <div className="grid gap-0 sm:grid-cols-2">
            <Toggle label={T.labelReranker} checked={settings.reranker} onChange={() => toggleSetting('reranker')} desc={T.descReranker} />
            <Toggle label={T.labelExpandContext} checked={settings.expandContext} onChange={() => toggleSetting('expandContext')} desc={T.descExpandContext} />
            <Toggle label={T.labelOptimizeContext} checked={settings.optimizeContext} onChange={() => toggleSetting('optimizeContext')} desc={T.descOptimizeContext} />
            <Toggle label={T.labelRecomp} checked={settings.recompSynthesis} onChange={() => toggleSetting('recompSynthesis')} desc={T.descRecomp} />
          </div>
        </section>

        <section className="flex flex-col border border-[var(--border)] bg-[var(--popover)] p-4 md:col-span-2">
          <h3 className="t-label mb-2 text-[var(--accent)]">{T.section4}</h3>
          <p className="mb-4 text-xs text-[var(--text-muted)]">{T.reindexHint}</p>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
            <div className="flex flex-1 gap-2">
              <button
                className="flex flex-1 flex-col items-center justify-center gap-1 border border-dashed border-[var(--border-strong)] px-4 py-3 text-sm text-[var(--text-muted)] transition-all hover:border-[var(--accent)] hover:bg-[var(--popover-hover)] hover:text-[var(--text)] disabled:opacity-50 group"
                onClick={() => reindexFileInputRef.current?.click()}
                disabled={isReindexing}
              >
                <FileUp className="h-5 w-5 group-hover:text-[var(--accent)]" />
                <span className="font-medium">
                  {pendingReindexFiles.length ? `${pendingReindexFiles.length} PDF(s)` : T.addPdfs}
                </span>
              </button>
              {pendingReindexFiles.length > 0 && (
                <button
                  className="px-3 text-[var(--text-muted)] transition-all hover:bg-red-500/10 hover:text-red-400"
                  onClick={() => { setPendingReindexFiles([]); if (reindexFileInputRef.current) reindexFileInputRef.current.value = ''; }}
                  title={T.remove}
                >
                  <X className="h-5 w-5" />
                </button>
              )}
            </div>
            <button
              className="flex items-center justify-center gap-2 bg-[var(--accent)] px-6 py-3 text-sm font-semibold text-[var(--accent-contrast)] transition-all hover:bg-[var(--accent-hover)] disabled:opacity-50 sm:min-w-[10rem]"
              onClick={handleReindex}
              disabled={isReindexing}
            >
              {isReindexing ? <Loader2 className="h-5 w-5 animate-spin" /> : <RefreshCw className="h-5 w-5" />}
              {T.reindexBtn}
            </button>
          </div>
        </section>
      </div>
    </div>
  );

  // ---- Main UI ----
  return (
    <div className="flex h-screen bg-transparent text-zinc-300 font-sans overflow-hidden selection:bg-orange-500/30 p-2 md:p-4 gap-4">

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
        className={`fixed md:relative z-50 h-[calc(100vh-16px)] md:h-full w-[320px] glass-panel rounded-3xl flex flex-col transition-transform duration-300 ease-in-out shadow-2xl ${isSidebarOpen ? 'translate-x-2 md:translate-x-0' : '-translate-x-[120%] md:translate-x-0 md:w-0 md:opacity-0 md:overflow-hidden md:ml-[-16px]'}`}
      >
        {/* Sidebar Header */}
        <div className="p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src={theme === 'dark' ? '/logo-dark.png' : '/logo-light.png'} alt="MonkeyGrab" className="w-9 h-9 object-cover flex-shrink-0" />
            <h1 className="flex font-extrabold text-lg tracking-tight"><ShimmerText text="MonkeyGrab" /></h1>
          </div>
          <button className="md:hidden text-zinc-500 hover:text-white transition-colors bg-white/5 p-2 rounded-full" onClick={() => setIsSidebarOpen(false)}>
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Sidebar Tabs — Documents stays in the sidebar; Models & Pipeline
            open a full-area overlay in the main column. */}
        <div className="flex px-6 mb-2">
          <div className="flex w-full bg-[var(--surface)] p-1 border border-[var(--border)]">
            <button
              className={`flex-1 py-2 text-xs font-semibold transition-all ${mainPanel === null ? 'bg-[var(--surface-2)] text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
              onClick={() => { setMainPanel(null); setActiveTab('docs'); }}
            >
              {T.tabDocs}
            </button>
            <button
              className={`flex-1 py-2 text-xs font-semibold transition-all ${mainPanel === 'models' ? 'bg-[var(--surface-2)] text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
              onClick={() => openMainPanel('models')}
            >
              {T.tabModels}
            </button>
            <button
              className={`flex-1 py-2 text-xs font-semibold transition-all ${mainPanel === 'pipeline' ? 'bg-[var(--surface-2)] text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
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
                  <div className="flex items-center gap-2 pl-2 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                    <Database className="h-3 w-3" />
                    {T.storesLabel}
                  </div>
                  <div className={`rounded-2xl border border-white/10 bg-black/30 p-1.5 space-y-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] ${storeBusy || isReindexing || isLoading ? 'opacity-50' : ''}`}>
                    {stores.map(store => {
                      const isActive = store.name === activeStore;
                      return (
                        <button
                          key={store.name}
                          type="button"
                          className={`group flex w-full items-center gap-2 rounded-xl border px-2.5 py-2 text-left transition-all focus:outline-none disabled:cursor-not-allowed ${isActive
                            ? 'border-orange-500/50 bg-orange-500/15 shadow-[0_0_18px_rgba(230,140,82,0.16)]'
                            : 'border-transparent bg-white/[0.03] hover:border-white/10 hover:bg-white/[0.06]'
                            }`}
                          onClick={() => handleStoreSelect(store.name)}
                          disabled={storeBusy || isReindexing || isLoading}
                        >
                          <span className="flex min-w-0 flex-1 flex-col">
                            <span className="flex items-center gap-1.5">
                              <span className={`truncate text-xs font-bold tracking-wide ${isActive ? 'text-white' : 'text-zinc-400 group-hover:text-zinc-200'}`}>{store.label}</span>
                              {isActive && <Check className="h-3 w-3 flex-shrink-0 text-[var(--accent)]" />}
                            </span>
                            <span className={`mt-0.5 block truncate font-mono text-[9px] ${isActive ? 'text-[var(--accent)]' : 'text-[var(--text-faint)]'}`}>
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
                  {storeError && <p className="pl-2 text-[11px] text-red-400">{storeError}</p>}
                  {fingerprintStale && !isIndexing && (
                    <p className="flex items-start gap-1.5 pl-2 text-[11px] text-amber-400">
                      <AlertCircle className="w-3 h-3 mt-0.5 shrink-0" />
                      <span>{T.fingerprintStaleWarning}</span>
                    </p>
                  )}
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
                            <FileText className="w-4 h-4 text-[var(--accent)]" />
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
            ) : null}
          </AnimatePresence>
        </div>
      </motion.aside>

      {/* Main Content */}
      <main className="flex-1 flex min-w-0 relative glass-panel rounded-3xl overflow-hidden shadow-2xl">
        {/* PDF pane — 'full' replaces the chat (opened from the sidebar);
            'split' sits to the left of the chat (opened from a source citation). */}
        {pdfViewer && (
          <div className={`min-w-0 ${pdfViewer.mode === 'full' ? 'flex-1' : 'w-1/2 border-r border-white/10'}`}>
            <PdfPane
              doc={pdfViewer.doc}
              page={pdfViewer.page}
              onClose={closePdf}
            />
          </div>
        )}

        {/* Models / Pipeline overlay — occupies the full main area, like the PDF viewer */}
        {mainPanel && pdfViewer?.mode !== 'full' && (
          <div className="flex-1 flex flex-col min-w-0 relative">
            <header className="h-20 border-b border-[var(--border)] flex items-center justify-between gap-3 px-6 bg-[var(--surface)] z-10">
              <div className="flex items-center gap-2 min-w-0">
                {mainPanel === 'models'
                  ? <Ollama className="w-5 h-5 flex-shrink-0 text-[var(--text)]" />
                  : <Database className="w-5 h-5 flex-shrink-0" />}
                <h2 className="t-h2 text-[var(--text)] truncate">{mainPanel === 'models' ? T.tabModels : T.tabPipeline}</h2>
              </div>
              <button
                type="button"
                onClick={() => setMainPanel(null)}
                className="p-2 text-[var(--text-muted)] hover:text-[var(--text)] bg-[var(--surface)] hover:bg-[var(--surface-2)] border border-[var(--border)] transition-colors flex-shrink-0"
                title={T.close}
                aria-label={T.close}
              >
                <X className="w-4 h-4" />
              </button>
            </header>
            <div className="flex-1 overflow-y-auto p-6 md:p-8 custom-scrollbar">
              {mainPanel === 'models' ? renderModelsPanel() : renderPipelinePanel()}
            </div>
          </div>
        )}

        {/* Chat column — hidden while a full-screen PDF or a panel is open */}
        {pdfViewer?.mode !== 'full' && !mainPanel && (
        <div className="flex-1 flex flex-col min-w-0 relative">
        {/* Header */}
        <header className="h-20 border-b border-[var(--border)] flex items-center justify-between gap-3 px-4 bg-[var(--surface)] z-10">
          <div className="flex items-center gap-3 min-w-0">
            <button
              className="p-2.5 text-[var(--text-muted)] hover:text-[var(--text)] bg-[var(--surface)] hover:bg-[var(--surface-2)] border border-[var(--border)] transition-colors flex-shrink-0"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            >
              <Menu className="w-5 h-5" />
            </button>

            <div className="flex bg-[var(--surface)] p-1 border border-[var(--border)] flex-shrink-0">
              <button
                className={`min-w-[84px] justify-center px-4 py-2 text-xs font-bold tracking-wide transition-all flex items-center gap-2 ${mode === 'chat' ? 'bg-[var(--surface-2)] text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
                onClick={() => handleModeChange('chat')}
              >
                <Ollama className="w-4 h-4 text-[var(--text)]" />
                CHAT
              </button>
              <button
                className={`min-w-[84px] justify-center px-4 py-2 text-xs font-bold tracking-wide transition-all flex items-center gap-2 ${mode === 'rag' ? 'bg-[var(--accent)] text-[var(--accent-contrast)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
                onClick={() => handleModeChange('rag')}
              >
                <Database className="w-4 h-4" />
                RAG
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              type="button"
              onClick={toggleTheme}
              className="p-2 text-[var(--text-muted)] hover:text-[var(--text)] bg-[var(--surface)] hover:bg-[var(--surface-2)] border border-[var(--border)] transition-colors"
              title={theme === 'dark' ? T.themeLight : T.themeDark}
              aria-label={theme === 'dark' ? T.themeLight : T.themeDark}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
            <LanguageToggle lang={lang} setLang={setLang} />
            <button
              className="p-2 text-[var(--text-muted)] hover:text-[var(--accent)] bg-[var(--surface)] hover:bg-[var(--surface-2)] border border-[var(--border)] transition-colors"
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
          <div className="max-w-3xl mx-auto space-y-10 pb-20 relative z-10">
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
                  <div className={`flex max-w-[85%] items-start gap-2 px-4 py-2.5 rounded-2xl text-xs font-medium ${msg.isError ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-white/5 text-zinc-400 border border-white/5'}`}>
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
                      <div className={`text-[15px] leading-relaxed ${
                        msg.role === 'user'
                          ? 'text-[var(--text)] font-medium text-left'
                          : msg.isError
                            ? 'text-red-400'
                            : 'text-[var(--text)]'
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
                                className="inline-flex max-w-full items-center gap-2 px-3 py-1.5 rounded-full bg-black/40 border border-white/5 text-xs text-zinc-300 hover:bg-white/10 hover:border-orange-500/30 hover:text-[var(--accent)] transition-all group cursor-pointer"
                                onClick={() => openPdf(cite.document, cite.best_page ?? cite.pages[0] ?? 1, 'split')}
                                title={T.viewPdf}
                              >
                                <FileText className="w-3.5 h-3.5 text-[var(--accent)] group-hover:text-orange-400" />
                                <span className="font-medium truncate min-w-0">{cite.document}</span>
                                <span className="text-zinc-600">|</span>
                                <span className="text-zinc-400 shrink-0">p. {cite.best_page ?? cite.pages[0]}</span>
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

        {/* Input Area */}
        <div
          className="p-6 absolute bottom-0 left-0 right-0 z-20"
          style={{ background: 'linear-gradient(to top, var(--bg), color-mix(in srgb, var(--bg) 88%, transparent) 55%, transparent)' }}
        >
          <div className="max-w-3xl mx-auto relative">
            <div className="glass-panel relative flex items-end gap-3 p-2.5 focus-within:border-[var(--accent)] transition-all">
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
                className="flex-1 max-h-48 min-h-[52px] bg-transparent border-none focus:ring-0 focus:outline-none resize-none py-3.5 px-4 text-[15px] text-[var(--text)] placeholder:text-[var(--text-faint)] custom-scrollbar font-medium"
                rows={1}
                disabled={isLoading}
              />

              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="p-3.5 bg-[var(--accent)] text-[var(--accent-contrast)] hover:bg-[var(--accent-hover)] active:scale-95 disabled:opacity-40 disabled:bg-[var(--surface-2)] disabled:text-[var(--text-faint)] transition-all flex-shrink-0"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5 ml-0.5" />
                )}
              </button>
            </div>
          </div>
        </div>
        </div>
        )}
      </main>
    </div>
  );
}

// =============================================================================
// PDF Viewer Modal
// =============================================================================

const GREETING: Record<Lang, string> = { es: 'Hola', en: 'Hi', ca: 'Hola' };

/** Letters cycle white → accent → white (staggered): a colour shimmer, no motion.
    Render inside a flex container that sets the font size and weight. */
function ShimmerText({ text }: { text: string }) {
  return (
    <>
      {Array.from(text).map((ch, i) => (
        <span
          key={i}
          className="shimmer-char"
          style={{ animationDelay: `${i * 0.12}s` }}
        >
          {ch}
        </span>
      ))}
    </>
  );
}

function PdfPane({ doc, page, onClose }: { doc: string; page: number; onClose: () => void }) {
  const base = `/api/pdf/${encodeURIComponent(doc)}`;
  // toolbar=0 hides the browser's built-in PDF bar (which repeats the file name).
  const src = `${base}#page=${page}&toolbar=0`;
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Reset the loading overlay whenever the target document/page changes.
  useEffect(() => { setLoaded(false); }, [src]);

  return (
    <div className="flex flex-col w-full h-full bg-[#0d0c12]">
      <div className="h-20 flex items-center justify-between px-5 bg-white/[0.03] border-b border-white/5 flex-shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <FileText className="w-4 h-4 text-orange-400 flex-shrink-0" />
          <span className="text-sm font-medium text-zinc-200 truncate">{doc}</span>
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          {/* Fallback: open in the system browser if the embedded viewer stays blank. */}
          <a
            href={base}
            target="_blank"
            rel="noreferrer"
            className="p-2 rounded-full text-zinc-500 hover:text-orange-400 hover:bg-white/10 transition-all"
            title="Abrir en el navegador"
          >
            <Eye className="w-5 h-5" />
          </a>
          <button
            onClick={onClose}
            className="p-2 rounded-full text-zinc-500 hover:text-white hover:bg-white/10 transition-all"
            title="Cerrar (Esc)"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
      <div className="relative flex-1 min-h-0">
        {!loaded && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-[#0d0c12]">
            <Loader2 className="w-8 h-8 text-orange-500 animate-spin" />
          </div>
        )}
        <iframe
          key={src}
          src={src}
          onLoad={() => setLoaded(true)}
          className="w-full h-full border-none bg-zinc-900"
          title={doc}
        />
      </div>
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
      className="flex w-full items-center justify-between gap-4 rounded-lg p-2 text-left transition-colors hover:bg-[var(--popover-hover)] focus:outline-none focus:ring-2 focus:ring-orange-500/50 group"
      onClick={onChange}
    >
      <span className="flex-1">
        <span className="block text-sm font-medium text-[var(--text)] transition-colors">{label}</span>
        <span className="mt-1 block text-[11px] leading-snug text-[var(--text-muted)]">{desc}</span>
      </span>
      <span className={`relative inline-flex h-6 w-11 flex-shrink-0 items-center justify-center rounded-full transition-colors duration-300 ease-in-out ${checked ? 'bg-orange-500 shadow-[0_0_10px_rgba(230,140,82,0.4)]' : 'bg-[var(--popover-hover)]'}`}>
        <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-md transition duration-300 ease-in-out ${checked ? 'translate-x-2.5' : '-translate-x-2.5'}`} />
      </span>
    </button>
  );
}

function LanguageToggle({ lang, setLang }: { lang: Lang; setLang: (lang: Lang) => void }) {
  return (
    <div className="flex items-center gap-1 border border-[var(--border)] bg-[var(--surface)] p-1">
      <Languages className="ml-1.5 h-3.5 w-3.5 text-[var(--text-faint)]" />
      {LANG_OPTIONS.map(option => (
        <button
          key={option.code}
          type="button"
          className={`w-9 text-center px-1 py-1 text-[10px] font-bold tracking-wide transition-all ${
            lang === option.code
              ? 'bg-[var(--accent)] text-[var(--accent-contrast)]'
              : 'text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]'
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

// Styled replacement for the native <select> used to pick a model per role.
// Uses semantic theme tokens so the trigger and panel match light/dark mode.
function ModelSelect({
  value, options, disabled, onChange,
}: {
  value: string;
  options: string[];
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen(o => !o)}
        className="flex w-full items-center justify-between gap-2 rounded-xl border border-[var(--border)] bg-[var(--popover)] px-2.5 py-2 text-xs text-[var(--text)] transition-colors hover:border-[var(--border-strong)] focus:border-orange-500/50 focus:outline-none focus:ring-2 focus:ring-orange-500/20 disabled:opacity-50"
      >
        <span className="truncate">{value || '—'}</span>
        <ChevronDown className={`h-3.5 w-3.5 flex-shrink-0 text-[var(--text-faint)] transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="custom-scrollbar absolute z-50 mt-1.5 max-h-56 w-full overflow-y-auto rounded-xl border border-[var(--border)] bg-[var(--popover)] p-1 shadow-xl"
          >
            {options.map(name => {
              const selected = name === value;
              return (
                <li key={name}>
                  <button
                    type="button"
                    onClick={() => { onChange(name); setOpen(false); }}
                    className={`flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors ${
                      selected ? 'bg-[var(--popover-active)] text-[var(--accent)]' : 'text-[var(--text)] hover:bg-[var(--popover-hover)]'
                    }`}
                  >
                    <span className="truncate">{name}</span>
                    {selected && <Check className="h-3.5 w-3.5 flex-shrink-0 text-[var(--accent)]" />}
                  </button>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
