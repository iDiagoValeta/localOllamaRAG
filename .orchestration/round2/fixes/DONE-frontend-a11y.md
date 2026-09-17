# DONE — frontend a11y residuals

Brief: `.orchestration/round2/fixes/brief-mg-frontend-a11y.md`.

## Fixed

1. **Visible focus** (`focus-visible:ring-1 focus-visible:ring-divider`, mouse users unchanged):
   - `App.tsx`: store buttons, Docs|Models|Pipeline tabs, CHAT|RAG|Study mode buttons,
     sidebar toggle, theme toggle, clear-chat, both send buttons, study-kind buttons.
   - `ModelSelect.tsx`: trigger + option buttons.
   - `SettingsOverlay.tsx`: close button.
   - `Toggle.tsx`: `focus:ring-*` switched to `focus-visible:ring-*` (kept `ring-edge`).
   - `LanguageToggle.tsx`: focus-visible ring on language buttons.
   - Composer textarea keeps `focus:outline-none`: its visible indicator is the
     panel's `focus-within:border-divider`.
2. **Selected-state semantics** (`aria-pressed`, mirroring `LanguageToggle`):
   sidebar tabs, CHAT|RAG|Study mode buttons, study-kind buttons,
   theme toggle (`aria-pressed={theme === 'dark'}`).
3. **Reduced motion**: `main.tsx` wraps the app in
   `<MotionConfig reducedMotion="user">` (`motion/react`, verified exported).
   Shimmer `prefers-reduced-motion` media query in `index.css` untouched.
4. **Escape closes Models/Pipeline overlay**: `useEffect` keydown in
   `SettingsOverlay`, same pattern as `PdfPane`. No full focus-trap (per brief).

## Verify

- `pnpm run lint` (tsc --noEmit): clean.
- `pnpm run build`: succeeds (chunk-size warning pre-existing).
- No pipeline flags touched.
