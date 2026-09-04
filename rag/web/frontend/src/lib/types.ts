// =============================================================================
// Types
// =============================================================================

// 'study' is a front-end-only mode: the backend's /api/mode knows chat and rag,
// and a study turn is a POST to /api/study rather than a mode the server holds.
export type Mode = 'chat' | 'rag' | 'study';
export type StudyKind = 'summary' | 'outline' | 'quiz';

export interface OutlineNode { title: string; children: OutlineNode[]; }
export interface StudyArtifact {
  source_document?: string;
  sections?: { heading: string; body: string; source_pages: number[] }[];
  nodes?: OutlineNode[];
  questions?: { prompt: string; options: string[]; correct_index: number; source_pages: number[] }[];
}

export interface Citation {
  document: string;
  pages: number[];
  best_page?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  mode: Mode;
  citations?: Citation[];
  metrics?: { searchTime: string; chunks: number };
  isStreaming?: boolean;
  isError?: boolean;
  // A study turn answers with structure rather than prose, so it is carried
  // as data and rendered by the bubble instead of being flattened to text.
  artifact?: StudyArtifact;
  artifactKind?: StudyKind;
}

export interface PipelineSettings {
  contextualRetrieval: boolean;
  queryDecomposition: boolean;
  hybridSearch: boolean;
  imageIndexing: boolean;
  imageDescription: boolean;
  reranker: boolean;
  expandContext: boolean;
  optimizeContext: boolean;
  recompSynthesis: boolean;
}

export interface IndexingProgress {
  file: string;
  file_index: number;
  total_files: number;
}

export interface OllamaModel {
  name: string;
  size?: number;
  family?: string;
  parameter_size?: string;
  capabilities?: string[];
  embedding?: boolean;
  vision?: boolean;
}

export type ModelRole = 'rag' | 'chat' | 'contextual' | 'recomp';
export type ModelRoles = Record<ModelRole, string>;

export interface VectorStore {
  name: string;
  label: string;
  docs_folder: string;
  pdf_count: number;
  indexed: boolean;
  active: boolean;
  fragments: number | null;
}

export type Lang = 'es' | 'en' | 'ca';
