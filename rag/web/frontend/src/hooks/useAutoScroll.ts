import { useCallback, useEffect, useLayoutEffect, useRef } from 'react';
import type { Message, Mode } from '../lib/types';

/** Keeps the message pane at the bottom; returns the ref to put on the pane. */
export function useAutoScroll(messages: Message[], mode: Mode) {
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Scrolls the container itself rather than calling scrollIntoView on a
  // zero-height anchor: that anchor sat inside the list's pb-32, so aligning
  // it left the last message short of the bottom, and scrollIntoView also
  // walks every scrollable ancestor, which is not what "keep this pane at the
  // bottom" means.
  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    const el = messagesContainerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Switching mode swaps the whole conversation at once. Animating that
  // scroll means racing the incoming messages' own entry animation, and the
  // smooth scroll settles against heights that are still changing -- which
  // is what left the conversation part-way up the pane, cut off at the
  // bottom. A jump is also what the user expects here: they did not scroll,
  // they changed view, and arriving mid-conversation reads as lost state.
  //
  // useLayoutEffect, not useEffect: this runs after the DOM is updated but
  // before the browser paints, so the new conversation is never shown at the
  // wrong offset first. requestAnimationFrame would do the same job one frame
  // later -- visibly, and not at all while the tab is in the background,
  // where rAF is throttled to a stop.
  useLayoutEffect(() => {
    scrollToBottom('auto');
  }, [mode, scrollToBottom]);

  return messagesContainerRef;
}
