import { useCallback, useState } from 'react';
import { api } from '../lib/api';
import type { Mode } from '../lib/types';

export function useMode() {
  const [mode, setMode] = useState<Mode>('rag');

  const handleModeChange = useCallback(async (newMode: Mode) => {
    const previousMode = mode;
    setMode(newMode);
    // The backend only knows chat and rag. A study turn is a POST to
    // /api/study, so telling the server about this mode would either 400 or
    // leave it holding a mode nothing reads.
    if (newMode === 'study') return;
    const result = await api.setMode(newMode).catch(() => null);
    if (!result?.ok) setMode(previousMode);
  }, [mode]);

  return { mode, setMode, handleModeChange };
}
