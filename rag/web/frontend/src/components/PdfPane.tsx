import { useState, useEffect } from 'react';
import { Eye, FileText, Loader2, X } from '../lib/icons';

export function PdfPane({ doc, page, onClose }: { doc: string; page: number; onClose: () => void }) {
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
    <div className="flex flex-col w-full h-full bg-surface">
      <div className="h-20 flex items-center justify-between px-5 bg-surface-raised border-b border-divider flex-shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <FileText className="w-4 h-4 text-ink-muted flex-shrink-0" />
          <span className="text-sm font-medium text-ink truncate">{doc}</span>
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          {/* Fallback: open in the system browser if the embedded viewer stays blank. */}
          <a
            href={base}
            target="_blank"
            rel="noreferrer"
            className="p-2 rounded-full text-ink-faint hover:text-ink hover:bg-surface transition-all"
            title="Abrir en el navegador"
          >
            <Eye className="w-5 h-5" />
          </a>
          <button
            onClick={onClose}
            className="p-2 rounded-full text-ink-faint hover:text-ink hover:bg-surface transition-all"
            title="Cerrar (Esc)"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
      <div className="relative flex-1 min-h-0">
        {!loaded && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-surface">
            <Loader2 className="w-8 h-8 text-ink-muted animate-spin" />
          </div>
        )}
        <iframe
          key={src}
          src={src}
          onLoad={() => setLoaded(true)}
          className="w-full h-full border-none bg-field"
          title={doc}
        />
      </div>
    </div>
  );
}

export default PdfPane;
