import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// Vitest runs without globals, so Testing Library cannot register its own
// cleanup; without it every test would render into the previous one's DOM.
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  localStorage.clear();
});

// jsdom implements neither: the chat scrolls its container after each message.
Element.prototype.scrollTo = () => {};
