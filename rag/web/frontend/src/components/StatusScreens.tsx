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
    <div className="flex h-screen items-center justify-center bg-transparent text-ink p-4">
      <div className="glass-panel rounded-3xl p-10 max-w-md text-center space-y-4">
        <Loader2 className="w-12 h-12 text-ink-muted animate-spin mx-auto" />
        <h2 className="text-xl font-semibold text-ink">
          {title}
        </h2>
        <p className="text-ink-muted text-sm">
          {hint}
        </p>
        {error && showRetry && (
          <p className="text-warning text-sm">{error}</p>
        )}
        {progress ? (
          <div className="space-y-1">
            <p className="text-ink text-sm font-medium">
              {processingLabel} <span className="text-ink-soft font-semibold">{progress.file}</span>
            </p>
            <p className="text-ink-faint text-xs">
              {progress.file_index} / {progress.total_files} {progress.total_files !== 1 ? fileUnitPlural : fileUnit}
            </p>
            <div className="w-full bg-field border border-edge rounded-full h-1.5 mt-2 overflow-hidden">
              <div
                className="bg-ink-muted h-full rounded-full transition-all duration-500"
                style={{ width: `${(progress.file_index / progress.total_files) * 100}%` }}
              />
            </div>
          </div>
        ) : (
          <p className="text-ink-faint text-xs">
            {autoRefreshLabel}
          </p>
        )}
        {showRetry && (
          <button
            className="px-6 py-2 bg-field text-ink border border-edge hover:border-divider hover:bg-surface-raised font-semibold transition-colors rounded-lg"
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
    <div className="flex h-screen items-center justify-center bg-transparent text-ink p-4">
      <div className="glass-panel rounded-3xl p-10 max-w-md text-center space-y-4">
        <AlertCircle className="w-12 h-12 text-danger mx-auto" />
        <h2 className="text-xl font-semibold text-ink">{title}</h2>
        <p className="text-ink-muted text-sm">{error}</p>
        <button
          className="px-6 py-2 bg-field text-ink border border-edge hover:border-divider hover:bg-surface-raised font-semibold transition-colors rounded-lg"
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
    <div className="flex h-screen items-center justify-center bg-transparent text-ink">
      <div className="flex flex-col items-center gap-4">
        <Loader2 className="w-8 h-8 text-ink-muted animate-spin" />
        <p className="text-ink-muted text-sm">{message}</p>
      </div>
    </div>
  );
}
