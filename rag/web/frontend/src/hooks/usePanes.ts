import { useCallback, useRef, useState } from 'react';

export interface PdfViewerState { doc: string; page: number; mode: 'full' | 'split' }

/** What shares the screen with the chat: the sidebar, the PDF viewer and the Models / Pipeline panel. */
export function usePanes() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [pdfViewer, setPdfViewer] = useState<PdfViewerState | null>(null);
  // Full-area overlay panels (Models / Pipeline) shown in the main column, like
  // the PDF viewer. Opening one closes any open PDF; closing returns to chat.
  const [mainPanel, setMainPanel] = useState<'models' | 'pipeline' | null>(null);
  const openMainPanel = useCallback((panel: 'models' | 'pipeline') => {
    setPdfViewer(null);
    setMainPanel(panel);
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  }, []);

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

  return {
    isSidebarOpen, setIsSidebarOpen,
    pdfViewer, setPdfViewer, openPdf, closePdf,
    mainPanel, setMainPanel, openMainPanel,
  };
}
