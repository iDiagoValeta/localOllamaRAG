import React from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const MathInline: React.FC<{ tex: string }> = ({ tex }) => {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: false });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
};

const MathBlock: React.FC<{ tex: string }> = ({ tex }) => {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: true });
  return <div className="overflow-x-auto my-2" dangerouslySetInnerHTML={{ __html: html }} />;
};

function renderInlineMarkdown(text: string, keyPrefix: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const pattern = /(\$\$[^$\n]+\$\$|\$(?!\$)(?:[^$\n\\]|\\.)+\$|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const token = match[0];
    const key = `${keyPrefix}-${match.index}`;

    if (token.startsWith('$$')) {
      parts.push(<MathBlock key={key} tex={token.slice(2, -2)} />);
    } else if (token.startsWith('$')) {
      parts.push(<MathInline key={key} tex={token.slice(1, -1)} />);
    } else if (token.startsWith('`')) {
      parts.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith('**')) {
      parts.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else {
      parts.push(<em key={key}>{token.slice(1, -1)}</em>);
    }

    last = match.index + token.length;
  }

  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export function MarkdownContent({ text, compact = false }: { text: string; compact?: boolean }) {
  if (!text) return null;

  const className = compact ? 'markdown-content compact' : 'markdown-content';

  if (text.includes('```')) {
    const nodes: React.ReactNode[] = [];
    let inCode = false;
    let codeLines: string[] = [];
    let paragraph: string[] = [];
    let inMath = false;
    let mathLines: string[] = [];

    const flushParagraph = (key: string) => {
      if (!paragraph.length) return;
      nodes.push(<p key={key}>{paragraph.map((line, i) => <React.Fragment key={i}>{i > 0 && <br />}{renderInlineMarkdown(line, `${key}-${i}`)}</React.Fragment>)}</p>);
      paragraph = [];
    };

    text.split('\n').forEach((line, i) => {
      if (line.trim() === '$$' && !inCode) {
        if (inMath) {
          flushParagraph(`p-${i}`);
          nodes.push(<MathBlock key={`math-${i}`} tex={mathLines.join('\n')} />);
          mathLines = [];
          inMath = false;
        } else {
          flushParagraph(`p-${i}`);
          inMath = true;
        }
        return;
      }

      if (inMath) {
        mathLines.push(line);
        return;
      }

      if (/^```/.test(line)) {
        if (inCode) {
          nodes.push(<pre key={`code-${i}`}><code>{codeLines.join('\n')}</code></pre>);
          codeLines = [];
          inCode = false;
        } else {
          flushParagraph(`p-${i}`);
          inCode = true;
        }
        return;
      }

      if (inCode) {
        codeLines.push(line);
      } else if (line.trim()) {
        paragraph.push(line);
      } else {
        flushParagraph(`p-${i}`);
      }
    });

    if (inCode) nodes.push(<pre key="code-final"><code>{codeLines.join('\n')}</code></pre>);
    flushParagraph('p-final');
    return <div className={className}>{nodes}</div>;
  }

  return (
    <div className={className}>
      {text.split(/\n{2,}/).map((block, i) => {
        const lines = block.split('\n').filter(Boolean);
        if (!lines.length) return null;

        const trimmed = block.trim();
        if (trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length > 4) {
          const inner = trimmed.slice(2, -2).trim();
          return <MathBlock key={i} tex={inner} />;
        }

        const heading = lines[0].match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
          const Tag = heading[1].length === 1 ? 'h2' : heading[1].length === 2 ? 'h3' : 'h4';
          return <Tag key={i}>{renderInlineMarkdown(heading[2], `h-${i}`)}</Tag>;
        }

        if (lines.every(line => /^[-*]\s+/.test(line))) {
          return (
            <ul key={i}>
              {lines.map((line, j) => (
                <li key={j}>{renderInlineMarkdown(line.replace(/^[-*]\s+/, ''), `ul-${i}-${j}`)}</li>
              ))}
            </ul>
          );
        }

        if (lines.every(line => /^\d+\.\s+/.test(line))) {
          return (
            <ol key={i}>
              {lines.map((line, j) => (
                <li key={j}>{renderInlineMarkdown(line.replace(/^\d+\.\s+/, ''), `ol-${i}-${j}`)}</li>
              ))}
            </ol>
          );
        }

        return (
          <p key={i}>
            {lines.map((line, j) => (
              <React.Fragment key={j}>
                {j > 0 && <br />}
                {renderInlineMarkdown(line, `p-${i}-${j}`)}
              </React.Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
