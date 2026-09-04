import React from 'react';
import {
  Database, FileUp, Loader2, Ollama,
  Power, RefreshCw, X,
} from '../lib/icons';
import { fill, type Strings } from '../lib/i18n';
import type {
  IndexingProgress,
  ModelRole,
  ModelRoles,
  OllamaModel,
  PipelineSettings,
} from '../lib/types';
import { ModelSelect } from './ModelSelect';
import { Toggle } from './Toggle';

export interface SettingsOverlayProps {
  // Navigation & overlay state
  mainPanel: 'models' | 'pipeline';
  onClose: () => void;

  // Ollama server status
  ollamaStatus: { running: boolean };
  ollamaStarting: boolean;
  refreshOllama: () => void;
  handleStartOllama: () => void;

  // Model roles
  ollamaModels: OllamaModel[];
  modelRoles: ModelRoles | null;
  savingRole: ModelRole | null;
  modelError: string | null;
  handleRoleChange: (role: ModelRole, value: string) => void;

  // Pipeline settings and reindexing
  settings: PipelineSettings;
  toggleSetting: (key: keyof PipelineSettings) => void;
  settingsError: string | null;
  isReindexing: boolean;
  indexingProgress: IndexingProgress | null;
  pendingReindexFiles: File[];
  setPendingReindexFiles: React.Dispatch<React.SetStateAction<File[]>>;
  reindexFileInputRef: React.RefObject<HTMLInputElement | null>;
  handleReindex: () => void;

  // Localized strings
  strings: Strings;
}

export function SettingsOverlay({
  mainPanel,
  onClose,
  ollamaStatus,
  ollamaStarting,
  refreshOllama,
  handleStartOllama,
  ollamaModels,
  modelRoles,
  savingRole,
  modelError,
  handleRoleChange,
  settings,
  toggleSetting,
  settingsError,
  isReindexing,
  indexingProgress,
  pendingReindexFiles,
  setPendingReindexFiles,
  reindexFileInputRef,
  handleReindex,
  strings,
}: SettingsOverlayProps) {
  const renderModelsPanel = () => (
    <div className="mx-auto w-full max-w-4xl space-y-4">
      {/* Ollama server status */}
      <div className={`border p-3 rounded-xl ${ollamaStatus.running ? 'border-edge bg-surface-raised' : 'border-warning/30 bg-warning/10'}`}>
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <span className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${ollamaStatus.running ? 'bg-ok' : 'bg-danger'}`} />
            <div className="min-w-0">
              <div className="t-h3 text-ink">{strings.ollamaTitle}</div>
              <div className="truncate t-body-sm text-ink-muted">
                {ollamaStatus.running ? strings.ollamaOnline : strings.ollamaOffline}
              </div>
            </div>
          </div>
          {ollamaStatus.running ? (
            <button
              type="button"
              className="flex-shrink-0 p-2 text-ink-muted transition-all hover:bg-surface hover:text-ink rounded-lg"
              onClick={refreshOllama}
              title={strings.refreshModels}
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          ) : (
            <button
              type="button"
              className="flex flex-shrink-0 items-center gap-1.5 border border-warning/40 bg-warning/15 px-3 py-1.5 text-[11px] font-semibold text-warning transition-all hover:bg-warning/25 disabled:opacity-50 rounded-lg"
              onClick={handleStartOllama}
              disabled={ollamaStarting}
            >
              {ollamaStarting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Power className="h-3.5 w-3.5" />}
              {ollamaStarting ? strings.ollamaStarting : strings.ollamaStartBtn}
            </button>
          )}
        </div>
      </div>

      {modelError && (
        <div className="border border-danger/20 bg-danger/10 px-3 py-2 text-xs text-danger rounded-lg">{modelError}</div>
      )}

      {/* Model role selectors — two columns in the wide overlay */}
      <div className="space-y-2">
        <div className="t-label text-ink-faint pl-1">{strings.modelsRoles}</div>
        {ollamaStatus.running && ollamaModels.length === 0 ? (
          <p className="px-1 py-3 text-xs text-ink-muted">{strings.noModels}</p>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2">
            {([
              { role: 'rag' as ModelRole, label: strings.roleRag, desc: strings.descRoleRag },
              { role: 'chat' as ModelRole, label: strings.roleChat, desc: strings.descRoleChat },
              { role: 'contextual' as ModelRole, label: strings.roleContextual, desc: strings.descRoleContextual },
              { role: 'recomp' as ModelRole, label: strings.roleRecomp, desc: strings.descRoleRecomp },
            ]).map(({ role, label, desc }) => {
              const current = modelRoles?.[role] ?? '';
              const names = ollamaModels.map(m => m.name);
              const options = current && !names.includes(current) ? [current, ...names] : names;
              return (
                <div key={role} className="border border-edge bg-surface-raised p-2.5 rounded-xl">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="t-h3 text-ink">{label}</span>
                    {savingRole === role && <Loader2 className="h-3 w-3 animate-spin text-ink-muted" />}
                  </div>
                  <p className="mb-2 t-body-sm text-ink-muted">{desc}</p>
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
        <div className={`border px-3 py-2 text-xs rounded-xl ${settingsError ? 'border-danger/20 bg-danger/10 text-danger' : 'border-edge bg-surface-raised text-ink'}`}>
          {settingsError ? (
            <span>{settingsError}</span>
          ) : indexingProgress ? (
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-3">
                <span className="truncate">{fill(strings.indexingFile, { file: indexingProgress.file })}</span>
                <span className="font-mono text-[10px] text-ink-muted">{indexingProgress.file_index}/{indexingProgress.total_files}</span>
              </div>
              <div className="h-1.5 bg-field border border-edge rounded-full overflow-hidden">
                <div
                  className="h-full bg-ink-muted transition-all duration-500 rounded-full"
                  style={{ width: `${Math.max(5, (indexingProgress.file_index / indexingProgress.total_files) * 100)}%` }}
                />
              </div>
            </div>
          ) : (
            <span>{strings.reindexingStatus}</span>
          )}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <section className="flex flex-col border border-edge bg-surface-raised p-3 rounded-xl">
          <h3 className="t-label mb-3 text-ink-muted">{strings.section1}</h3>
          <div className="space-y-0.5">
            <Toggle label={strings.labelContextual} checked={settings.contextualRetrieval} onChange={() => toggleSetting('contextualRetrieval')} desc={strings.descContextual} />
            <Toggle label={strings.labelImageIndex} checked={settings.imageIndexing} onChange={() => toggleSetting('imageIndexing')} desc={strings.descImageIndex} />
            <Toggle label={strings.labelImageDesc} checked={settings.imageDescription} onChange={() => toggleSetting('imageDescription')} desc={strings.descImageDesc} />
          </div>
        </section>

        <section className="flex flex-col border border-edge bg-surface-raised p-3 rounded-xl">
          <h3 className="t-label mb-3 text-ink-muted">{strings.section2}</h3>
          <div className="space-y-0.5">
            <Toggle label={strings.labelHybrid} checked={settings.hybridSearch} onChange={() => toggleSetting('hybridSearch')} desc={strings.descHybrid} />
            <Toggle label={strings.labelQueryDecomp} checked={settings.queryDecomposition} onChange={() => toggleSetting('queryDecomposition')} desc={strings.descQueryDecomp} />
          </div>
        </section>

        <section className="flex flex-col border border-edge bg-surface-raised p-3 rounded-xl md:col-span-2">
          <h3 className="t-label mb-3 text-ink-muted">{strings.section3}</h3>
          <div className="grid gap-0 sm:grid-cols-2">
            <Toggle label={strings.labelReranker} checked={settings.reranker} onChange={() => toggleSetting('reranker')} desc={strings.descReranker} />
            <Toggle label={strings.labelExpandContext} checked={settings.expandContext} onChange={() => toggleSetting('expandContext')} desc={strings.descExpandContext} />
            <Toggle label={strings.labelOptimizeContext} checked={settings.optimizeContext} onChange={() => toggleSetting('optimizeContext')} desc={strings.descOptimizeContext} />
            <Toggle label={strings.labelRecomp} checked={settings.recompSynthesis} onChange={() => toggleSetting('recompSynthesis')} desc={strings.descRecomp} />
          </div>
        </section>

        <section className="flex flex-col border border-edge bg-surface-raised p-4 rounded-xl md:col-span-2">
          <h3 className="t-label mb-2 text-ink-muted">{strings.section4}</h3>
          <p className="mb-4 text-xs text-ink-muted">{strings.reindexHint}</p>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
            <div className="flex flex-1 gap-2">
              <button
                className="flex flex-1 flex-col items-center justify-center gap-1 border border-dashed border-edge px-4 py-3 text-sm text-ink-muted transition-all hover:border-divider hover:bg-field hover:text-ink disabled:opacity-50 group rounded-xl"
                onClick={() => reindexFileInputRef.current?.click()}
                disabled={isReindexing}
              >
                <FileUp className="h-5 w-5 group-hover:text-ink" />
                <span className="font-medium">
                  {pendingReindexFiles.length ? `${pendingReindexFiles.length} PDF(s)` : strings.addPdfs}
                </span>
              </button>
              {pendingReindexFiles.length > 0 && (
                <button
                  className="px-3 text-ink-muted transition-all hover:bg-danger/10 hover:text-danger rounded-xl"
                  onClick={() => { setPendingReindexFiles([]); if (reindexFileInputRef.current) reindexFileInputRef.current.value = ''; }}
                  title={strings.remove}
                >
                  <X className="h-5 w-5" />
                </button>
              )}
            </div>
            <button
              className="flex items-center justify-center gap-2 bg-field border border-edge text-ink px-6 py-3 text-sm font-semibold hover:bg-surface hover:border-divider disabled:opacity-50 sm:min-w-[10rem] rounded-xl transition-all"
              onClick={handleReindex}
              disabled={isReindexing}
            >
              {isReindexing ? <Loader2 className="h-5 w-5 animate-spin" /> : <RefreshCw className="h-5 w-5" />}
              {strings.reindexBtn}
            </button>
          </div>
        </section>
      </div>
    </div>
  );

  return (
    <div className="flex-1 flex flex-col min-w-0 relative">
      <header className="h-20 border-b border-divider flex items-center justify-between gap-3 px-6 bg-surface-raised z-10">
        <div className="flex items-center gap-2 min-w-0">
          {mainPanel === 'models'
            ? <Ollama className="w-5 h-5 flex-shrink-0 text-ink" />
            : <Database className="w-5 h-5 flex-shrink-0 text-ink" />}
          <h2 className="t-h2 text-ink truncate">{mainPanel === 'models' ? strings.tabModels : strings.tabPipeline}</h2>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="p-2 text-ink-muted hover:text-ink bg-field hover:bg-surface border border-edge rounded-lg transition-colors flex-shrink-0"
          title={strings.close}
          aria-label={strings.close}
        >
          <X className="w-4 h-4" />
        </button>
      </header>
      <div className="flex-1 overflow-y-auto p-6 md:p-8 custom-scrollbar">
        {mainPanel === 'models' ? renderModelsPanel() : renderPipelinePanel()}
      </div>
    </div>
  );
}

export default SettingsOverlay;
