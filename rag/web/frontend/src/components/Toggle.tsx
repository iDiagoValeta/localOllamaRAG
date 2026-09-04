export function Toggle({ label, checked, onChange, desc }: { label: string; checked: boolean; onChange: () => void; desc: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className="flex w-full items-center justify-between gap-4 rounded-lg p-2 text-left transition-colors hover:bg-[var(--popover-hover)] focus:outline-none focus:ring-2 focus:ring-orange-500/50 group"
      onClick={onChange}
    >
      <span className="flex-1">
        <span className="block text-sm font-medium text-[var(--text)] transition-colors">{label}</span>
        <span className="mt-1 block text-[11px] leading-snug text-[var(--text-muted)]">{desc}</span>
      </span>
      <span className={`relative inline-flex h-6 w-11 flex-shrink-0 items-center justify-center rounded-full transition-colors duration-300 ease-in-out ${checked ? 'bg-orange-500 shadow-[0_0_10px_rgba(230,140,82,0.4)]' : 'bg-[var(--popover-hover)]'}`}>
        <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-md transition duration-300 ease-in-out ${checked ? 'translate-x-2.5' : '-translate-x-2.5'}`} />
      </span>
    </button>
  );
}

export default Toggle;
