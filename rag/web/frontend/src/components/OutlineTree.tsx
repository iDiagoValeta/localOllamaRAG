import type { OutlineNode } from '../lib/types';

export function OutlineTree({ nodes, depth = 0 }: { nodes: OutlineNode[]; depth?: number }) {
  // Top level is a stack of headings; every level below hangs off a rule.
  // Indentation alone did not read as hierarchy — 16px against 15px text is
  // noise, and the result looked like a flat list. The rule is what makes a
  // child visibly belong to its parent.
  return (
    <ul className={depth === 0 ? 'space-y-3' : 'mt-1.5 space-y-1.5 border-l border-[var(--border)] pl-4'}>
      {nodes.map((node, i) => (
        <li key={`${depth}-${i}-${node.title}`} className="relative">
          {depth > 0 && (
            // Ticks the rule at each child, so siblings are countable.
            <span className="absolute -left-4 top-[0.7em] w-2 h-px bg-[var(--border)]" aria-hidden="true" />
          )}
          <span
            className={
              depth === 0
                ? 'block text-[var(--text)] font-semibold text-[15px]'
                : depth === 1
                  ? 'block text-[var(--text)] text-[14px]'
                  : 'block text-[var(--text-muted)] text-[13px]'
            }
          >
            {node.title}
          </span>
          {node.children?.length ? <OutlineTree nodes={node.children} depth={depth + 1} /> : null}
        </li>
      ))}
    </ul>
  );
}

export default OutlineTree;
