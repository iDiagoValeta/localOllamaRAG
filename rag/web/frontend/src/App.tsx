import { useState } from 'react';
import type { Message } from './lib/types';
import { PdfPane } from './components/PdfPane';
import { ConnectionErrorScreen, IndexingScreen, LoadingScreen } from './components/StatusScreens';
import { SettingsOverlay } from './components/SettingsOverlay';
import { Sidebar } from './components/Sidebar';
import { ChatHeader } from './components/ChatHeader';
import { MessageList } from './components/MessageList';
import { Composer } from './components/Composer';
import { useAutoScroll } from './hooks/useAutoScroll';
import { useChat } from './hooks/useChat';
import { useCorpus } from './hooks/useCorpus';
import { useCorpusActions } from './hooks/useCorpusActions';
import { useIndexing } from './hooks/useIndexing';
import { useLanguage } from './hooks/useLanguage';
import { useMode } from './hooks/useMode';
import { useModels } from './hooks/useModels';
import { usePanes } from './hooks/usePanes';
import { useServerSession } from './hooks/useServerSession';
import { useSettings } from './hooks/useSettings';
import { useTheme } from './hooks/useTheme';

// The hooks each own one concern and the pieces that must be shared between
// them (the conversation, the active corpus, the indexing state) are created
// here and handed down, so no hook reaches into another's state.
export default function App() {
  const { lang, setLang, T } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const panes = usePanes();
  const {
    isSidebarOpen, setIsSidebarOpen, pdfViewer, setPdfViewer, openPdf, closePdf,
    mainPanel, setMainPanel, openMainPanel,
  } = panes;

  const [messages, setMessages] = useState<Message[]>([]);
  const { mode, setMode, handleModeChange } = useMode();
  const indexing = useIndexing();
  const corpus = useCorpus();
  const { documents, fingerprintStale, stores, activeStore, loadStores, setFingerprintStale } = corpus;
  const { settings, setSettings, settingsError, setSettingsError, toggleSetting } = useSettings(T, setFingerprintStale);

  const chat = useChat({ T, lang, mode, documents, setMessages });
  const { isLoading } = chat;
  const actions = useCorpusActions({
    T, mode, isLoading, corpus, indexing, setMessages, setPdfViewer, setSettingsError,
  });

  const session = useServerSession({ T, corpus, indexing, setMode, setMessages, setSettings });
  const { isInitialized, initError, userName } = session;
  const models = useModels({ T, isInitialized, loadStores, setFingerprintStale });

  const messagesContainerRef = useAutoScroll(messages, mode);

  // ---- Indexing screen (only a live run reaches here) ----
  if (indexing.isIndexing) {
    return (
      <IndexingScreen
        title={T.indexingTitle}
        hint={T.indexingHint}
        progress={indexing.indexingProgress}
        processingLabel={T.processing}
        fileUnit={T.fileUnit}
        fileUnitPlural={T.fileUnitPlural}
        autoRefreshLabel={T.autoRefresh}
      />
    );
  }

  // ---- Indexing failed permanently ----
  if (indexing.indexingError) {
    return (
      <ConnectionErrorScreen
        title={T.indexingFailed}
        error={indexing.indexingError}
        retryLabel={T.retry}
        onRetry={session.handleRetryIndexing}
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
        onRetry={session.handleRetry}
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

      <Sidebar
        isSidebarOpen={isSidebarOpen}
        setIsSidebarOpen={setIsSidebarOpen}
        mainPanel={mainPanel}
        setMainPanel={setMainPanel}
        openMainPanel={openMainPanel}
        stores={stores}
        activeStore={activeStore}
        storeBusy={actions.storeBusy}
        storeError={actions.storeError}
        handleStoreSelect={actions.handleStoreSelect}
        fingerprintStale={fingerprintStale}
        isIndexing={indexing.isIndexing}
        isReindexing={actions.isReindexing}
        isLoading={isLoading}
        documents={documents}
        deletingDoc={actions.deletingDoc}
        openPdf={openPdf}
        handleDeleteDoc={actions.handleDeleteDoc}
        reindexFileInputRef={actions.reindexFileInputRef}
        handleAddDocs={actions.handleAddDocs}
        strings={T}
      />

      <main className="flex-1 flex min-w-0 relative glass-panel rounded-3xl overflow-hidden shadow-2xl">
        {/* PDF pane — 'full' replaces the chat (opened from the sidebar);
            'split' sits to the left of the chat (opened from a source citation). */}
        {pdfViewer && (
          <div className={`min-w-0 ${pdfViewer.mode === 'full' ? 'flex-1' : 'w-1/2 border-r border-divider'}`}>
            <PdfPane
              doc={pdfViewer.doc}
              page={pdfViewer.page}
              onClose={closePdf}
              strings={T}
            />
          </div>
        )}

        {/* Models / Pipeline overlay — occupies the full main area, like the PDF viewer */}
        {mainPanel && pdfViewer?.mode !== 'full' && (
          <SettingsOverlay
            mainPanel={mainPanel}
            onClose={() => setMainPanel(null)}
            ollamaStatus={models.ollamaStatus}
            ollamaStarting={models.ollamaStarting}
            ollamaError={models.ollamaError}
            refreshOllama={models.refreshOllama}
            handleStartOllama={models.handleStartOllama}
            ollamaModels={models.ollamaModels}
            modelRoles={models.modelRoles}
            savingRole={models.savingRole}
            modelError={models.modelError}
            handleRoleChange={models.handleRoleChange}
            settings={settings}
            toggleSetting={toggleSetting}
            settingsError={settingsError}
            isReindexing={actions.isReindexing}
            isAnswering={isLoading}
            indexingProgress={indexing.indexingProgress}
            isAddingDocs={actions.isAddingDocs}
            reindexFileInputRef={actions.reindexFileInputRef}
            handleReindex={actions.handleReindex}
            strings={T}
          />
        )}

        {/* Chat column — hidden while a full-screen PDF or a panel is open */}
        {pdfViewer?.mode !== 'full' && !mainPanel && (
        <div className="flex-1 flex flex-col min-w-0 relative">
          <ChatHeader
            isSidebarOpen={isSidebarOpen}
            setIsSidebarOpen={setIsSidebarOpen}
            mode={mode}
            handleModeChange={handleModeChange}
            theme={theme}
            toggleTheme={toggleTheme}
            lang={lang}
            setLang={setLang}
            handleClear={chat.handleClear}
            strings={T}
          />

          <MessageList
            containerRef={messagesContainerRef}
            mode={mode}
            messages={messages}
            isLoading={isLoading}
            userName={userName}
            lang={lang}
            copiedId={chat.copiedId}
            handleCopyMessage={chat.handleCopyMessage}
            revealedQuiz={chat.revealedQuiz}
            revealQuiz={chat.revealQuiz}
            openPdf={openPdf}
            strings={T}
          />

          <Composer
            mode={mode}
            input={chat.input}
            setInput={chat.setInput}
            isLoading={isLoading}
            textareaRef={chat.textareaRef}
            documentSuggestions={chat.documentSuggestions}
            activeSuggestion={chat.activeSuggestion}
            setSuggestionIndex={chat.setSuggestionIndex}
            acceptSuggestion={chat.acceptSuggestion}
            studyKind={chat.studyKind}
            setStudyKind={chat.setStudyKind}
            handleSend={chat.handleSend}
            handleStop={chat.handleStop}
            strings={T}
          />
        </div>
        )}
      </main>
    </div>
  );
}
