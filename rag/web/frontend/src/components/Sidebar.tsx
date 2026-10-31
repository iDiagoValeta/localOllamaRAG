import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { AlertCircle, Check, Eye, FileText, Loader2, Trash2, X } from '../lib/icons';
import { fill, type Strings } from '../lib/i18n';
import type { VectorStore } from '../lib/types';
import { ShimmerText } from './ShimmerText';

interface SidebarProps {
  isSidebarOpen: boolean;
  setIsSidebarOpen: (open: boolean) => void;
  mainPanel: 'models' | 'pipeline' | null;
  setMainPanel: (panel: 'models' | 'pipeline' | null) => void;
  openMainPanel: (panel: 'models' | 'pipeline') => void;

  stores: VectorStore[];
  activeStore: string;
  storeBusy: boolean;
  storeError: string | null;
  handleStoreSelect: (name: string) => void;
  fingerprintStale: boolean;
  isIndexing: boolean;
  isReindexing: boolean;
  isLoading: boolean;

  documents: string[];
  deletingDoc: string | null;
  openPdf: (doc: string) => void;
  handleDeleteDoc: (doc: string) => void;
  // The picker the Pipeline panel's "Add PDFs" button clicks; it renders here,
  // between the backdrop and the panel, where it always has.
  reindexFileInputRef: React.RefObject<HTMLInputElement | null>;
  handleAddDocs: (files: File[]) => void;

  strings: Strings;
}

export function Sidebar({
  isSidebarOpen,
  setIsSidebarOpen,
  mainPanel,
  setMainPanel,
  openMainPanel,
  stores,
  activeStore,
  storeBusy,
  storeError,
  handleStoreSelect,
  fingerprintStale,
  isIndexing,
  isReindexing,
  isLoading,
  documents,
  deletingDoc,
  openPdf,
  handleDeleteDoc,
  reindexFileInputRef,
  handleAddDocs,
  strings: T,
}: SidebarProps) {
  const [activeTab, setActiveTab] = useState<'docs' | 'models' | 'settings'>('docs');

  return (
    <>
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
        onChange={(e) => { const f = e.target.files; if (f?.length) handleAddDocs(Array.from(f)); }}
      />


      <motion.aside
        id="app-sidebar"
        // Closing only slides or shrinks the panel; without inert Tab still
        // walks its invisible controls and a screen reader still reads them (#476).
        inert={!isSidebarOpen}
        aria-hidden={!isSidebarOpen}
        className={`fixed md:relative z-50 h-[calc(100vh-16px)] md:h-full w-[320px] glass-panel rounded-3xl flex flex-col transition-transform duration-300 ease-in-out shadow-2xl ${isSidebarOpen ? 'translate-x-2 md:translate-x-0' : '-translate-x-[120%] md:translate-x-0 md:w-0 md:opacity-0 md:overflow-hidden md:ml-[-16px]'}`}
      >
        {/* Sidebar Header */}
        <div className="p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="MonkeyGrab" className="w-9 h-9 object-cover flex-shrink-0" />
            <h1 className="flex font-extrabold text-lg tracking-tight"><ShimmerText text="MonkeyGrab" /></h1>
          </div>
          <button aria-label={T.closeSidebar} className="md:hidden text-ink-faint hover:text-ink transition-colors bg-field p-2 rounded-full" onClick={() => setIsSidebarOpen(false)}>
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Sidebar Tabs — Documents stays in the sidebar; Models & Pipeline
            open a full-area overlay in the main column. */}
        <div className="flex px-6 mb-2">
          <div className="flex w-full bg-field p-1 border border-edge rounded-xl">
            <button
              aria-pressed={mainPanel === null}
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all focus:outline-none focus-visible:ring-1 focus-visible:ring-divider ${mainPanel === null ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
              onClick={() => { setMainPanel(null); setActiveTab('docs'); }}
            >
              {T.tabDocs}
            </button>
            <button
              aria-pressed={mainPanel === 'models'}
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all focus:outline-none focus-visible:ring-1 focus-visible:ring-divider ${mainPanel === 'models' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
              onClick={() => openMainPanel('models')}
            >
              {T.tabModels}
            </button>
            <button
              aria-pressed={mainPanel === 'pipeline'}
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all focus:outline-none focus-visible:ring-1 focus-visible:ring-divider ${mainPanel === 'pipeline' ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:text-ink'}`}
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
                  <div className="pl-2 text-[10px] font-bold text-ink-muted uppercase tracking-widest">
                    {T.storesLabel}
                  </div>
                  <div className={`rounded-2xl border border-edge bg-field p-1.5 space-y-1 ${storeBusy || isReindexing || isLoading ? 'opacity-50' : ''}`}>
                    {stores.map(store => {
                      const isActive = store.name === activeStore;
                      return (
                        <button
                          key={store.name}
                          type="button"
                          className={`group flex w-full items-center gap-2 rounded-xl border px-2.5 py-2 text-left transition-all focus:outline-none focus-visible:ring-1 focus-visible:ring-divider disabled:cursor-not-allowed ${isActive
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
                  <div className="text-[10px] font-bold text-ink-muted uppercase tracking-widest pl-2">
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
                            className="p-1.5 rounded-full text-ink-faint hover:text-ink hover:bg-surface-raised focus-visible:text-ink transition-all flex-shrink-0"
                            onClick={() => openPdf(doc)}
                            title={T.viewPdf}
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            className="p-1.5 rounded-full text-ink-faint hover:text-danger hover:bg-danger/10 focus-visible:text-danger transition-all flex-shrink-0 disabled:opacity-50"
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
    </>
  );
}
