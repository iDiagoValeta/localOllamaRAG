import { AlertCircle, Loader2 } from '../lib/icons';
import type { IndexingProgress } from '../lib/types';

export function IndexingScreen({
  title,
  hint,
  error,
  showRetry,
  progress,
  processingLabel,
  fileUnit,
  fileUnitPlural,
  autoRefreshLabel,
  retryLabel,
  onRetry,
}: {
  title: string;
  hint: string;
  error?: string | null;
  showRetry?: boolean;
  progress?: IndexingProgress | null;
  processingLabel: string;
  fileUnit: string;
  fileUnitPlural: string;
  autoRefreshLabel: string;
  retryLabel: string;
  onRetry: () => void;
}) {
  return (
    <div className="flex h-screen items-center justify-center bg-transparent text-zinc-300 p-4">
      <div className="glass-panel rounded-3xl p-10 max-w-md text-center space-y-4">
        <Loader2 className="w-12 h-12 text-orange-500 animate-spin mx-auto" />
        <h2 className="text-xl font-semibold text-white">
          {title}
        </h2>
        <p className="text-zinc-400 text-sm">
          {hint}
        </p>
        {error && showRetry && (
          <p className="text-amber-400 text-sm">{error}</p>
        )}
        {progress ? (
          <div className="space-y-1">
            <p className="text-zinc-300 text-sm font-medium">
              {processingLabel} <span className="text-orange-500">{progress.file}</span>
            </p>
            <p className="text-zinc-500 text-xs">
              {progress.file_index} / {progress.total_files} {progress.total_files !== 1 ? fileUnitPlural : fileUnit}
            </p>
            <div className="w-full bg-zinc-800 rounded-full h-1.5 mt-2">
              <div
                className="bg-orange-500 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${(progress.file_index / progress.total_files) * 100}%` }}
              />
            </div>
          </div>
        ) : (
          <p className="text-zinc-500 text-xs">
            {autoRefreshLabel}
          </p>
        )}
        {showRetry && (
          <button
            className="px-6 py-2 bg-[var(--accent)] text-[var(--accent-contrast)] font-semibold hover:bg-[var(--accent-hover)] transition-colors"
            onClick={onRetry}
          >
            {retryLabel}
          </button>
        )}
      </div>
    </div>
  );
}

export function ConnectionErrorScreen({
  title,
  error,
  retryLabel,
  onRetry,
}: {
  title: string;
  error: string;
  retryLabel: string;
  onRetry: () => void;
}) {
  return (
    <div className="flex h-screen items-center justify-center bg-transparent text-zinc-300 p-4">
      <div className="glass-panel rounded-3xl p-10 max-w-md text-center space-y-4">
        <AlertCircle className="w-12 h-12 text-red-400 mx-auto" />
        <h2 className="text-xl font-semibold text-white">{title}</h2>
        <p className="text-zinc-400 text-sm">{error}</p>
        <button
          className="px-6 py-2 bg-[var(--accent)] text-[var(--accent-contrast)] font-semibold hover:bg-[var(--accent-hover)] transition-colors"
          onClick={onRetry}
        >
          {retryLabel}
        </button>
      </div>
    </div>
  );
}

export function LoadingScreen({ message }: { message: string }) {
  return (
    <div className="flex h-screen items-center justify-center bg-transparent text-zinc-300">
      <div className="flex flex-col items-center gap-4">
        <Loader2 className="w-8 h-8 text-orange-400 animate-spin" />
        <p className="text-zinc-500 text-sm">{message}</p>
      </div>
    </div>
  );
}
