export function Toggle({ label, checked, onChange, desc }: { label: string; checked: boolean; onChange: () => void; desc: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className="flex w-full items-center justify-between gap-4 rounded-lg p-2 text-left transition-colors hover:bg-surface-raised focus:outline-none focus:ring-1 focus:ring-edge group"
      onClick={onChange}
    >
      <span className="flex-1">
        <span className="block text-sm font-medium text-ink transition-colors">{label}</span>
        <span className="mt-1 block text-[11px] leading-snug text-ink-muted">{desc}</span>
      </span>
      <span className={`relative inline-flex h-6 w-11 flex-shrink-0 items-center justify-center rounded-full transition-colors duration-200 ease-in-out ${checked ? 'bg-ink' : 'bg-field border border-edge'}`}>
        <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full transition duration-200 ease-in-out ${checked ? 'translate-x-2.5 bg-field' : '-translate-x-2.5 bg-ink-faint'}`} />
      </span>
    </button>
  );
}

export default Toggle;
