import { useCallback, useState } from 'react';
import { api, apiErrorText } from '../lib/api';
import type { Strings } from '../lib/i18n';
import type { PipelineSettings } from '../lib/types';

/** The pipeline flags the user can flip, and the error line of the Pipeline panel. */
export function useSettings(T: Strings, setFingerprintStale: (stale: boolean) => void) {
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [settings, setSettings] = useState<PipelineSettings>({
    contextualRetrieval: true,
    queryDecomposition: true,
    hybridSearch: true,
    imageIndexing: true,
    imageDescription: false,
    reranker: true,
    expandContext: true,
    optimizeContext: true,
    recompSynthesis: true,
  });

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
      setSettingsError(apiErrorText(T, result, T.settingsSaveError));
    }
  }, [settings, T]);

  return { settings, setSettings, settingsError, setSettingsError, toggleSetting };
}
