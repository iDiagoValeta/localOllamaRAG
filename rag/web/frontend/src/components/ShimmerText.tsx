/** Letters cycle white → accent → white (staggered): a colour shimmer, no motion.
    Render inside a flex container that sets the font size and weight. */
export function ShimmerText({ text }: { text: string }) {
  return (
    <>
      {Array.from(text).map((ch, i) => (
        <span
          key={i}
          className="shimmer-char"
          style={{ animationDelay: `${i * 0.12}s` }}
        >
          {ch}
        </span>
      ))}
    </>
  );
}

export default ShimmerText;
