import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Check, ChevronDown } from '../lib/icons';

// Styled replacement for the native <select> used to pick a model per role.
// Uses semantic theme tokens so the trigger and panel match light/dark mode.
export function ModelSelect({
  value, options, disabled, onChange,
}: {
  value: string;
  options: string[];
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen(o => !o)}
        className="flex w-full items-center justify-between gap-2 rounded-xl border border-[var(--border)] bg-[var(--popover)] px-2.5 py-2 text-xs text-[var(--text)] transition-colors hover:border-[var(--border-strong)] focus:border-orange-500/50 focus:outline-none focus:ring-2 focus:ring-orange-500/20 disabled:opacity-50"
      >
        <span className="truncate">{value || '—'}</span>
        <ChevronDown className={`h-3.5 w-3.5 flex-shrink-0 text-[var(--text-faint)] transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="custom-scrollbar absolute z-50 mt-1.5 max-h-56 w-full overflow-y-auto rounded-xl border border-[var(--border)] bg-[var(--popover)] p-1 shadow-xl"
          >
            {options.map(name => {
              const selected = name === value;
              return (
                <li key={name}>
                  <button
                    type="button"
                    onClick={() => { onChange(name); setOpen(false); }}
                    className={`flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors ${
                      selected ? 'bg-[var(--popover-active)] text-[var(--accent)]' : 'text-[var(--text)] hover:bg-[var(--popover-hover)]'
                    }`}
                  >
                    <span className="truncate">{name}</span>
                    {selected && <Check className="h-3.5 w-3.5 flex-shrink-0 text-[var(--accent)]" />}
                  </button>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}

export default ModelSelect;
