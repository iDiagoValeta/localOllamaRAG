import { useCallback, useMemo, useState } from 'react';
import { api } from '../lib/api';
import { fill, type Strings } from '../lib/i18n';
import type { Lang, Message, Mode, StudyArtifact, StudyKind } from '../lib/types';

interface Options {
  T: Strings;
  lang: Lang;
  mode: Mode;
  input: string;
  setInput: (input: string) => void;
  documents: string[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  setIsLoading: (loading: boolean) => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}

/** Study mode: matching a typed document, and asking for the artifact. */
export function useStudy({
  T, lang, mode, input, setInput, documents, setMessages, setIsLoading, textareaRef,
}: Options) {
  // The artifact kind is a control (three fixed choices); the document is
  // typed, because a corpus grows and a dropdown of every PDF stops being a
  // control and becomes a list to scroll.
  const [studyKind, setStudyKind] = useState<StudyKind>('summary');
  const [suggestionIndex, setSuggestionIndex] = useState(0);

  /** Documents whose name contains what has been typed, best-first.
   *
   * Substring rather than prefix: nobody remembers whether a paper is filed as
   * "attention-transformers" or "transformers-attention", and typing the half
   * you do remember should find it. A name that STARTS with the query sorts
   * first, so the prefix case still behaves like a prefix search. */
  const documentSuggestions = useMemo(() => {
    if (mode !== 'study') return [];
    const query = input.trim().toLowerCase();
    // Nothing typed, nothing suggested. A list that opens on focus covers the
    // conversation the user just generated, and the sidebar already lists the
    // corpus for anyone who needs to look it up.
    if (!query) return [];
    const hits = documents.filter(d => d.toLowerCase().includes(query));
    // An exact match is a decision already made, not a suggestion to show.
    if (hits.length === 1 && hits[0].toLowerCase() === query) return [];
    return hits.sort((a, b) => {
      const aStarts = a.toLowerCase().startsWith(query) ? 0 : 1;
      const bStarts = b.toLowerCase().startsWith(query) ? 0 : 1;
      return aStarts - bStarts || a.localeCompare(b);
    });
  }, [mode, input, documents]);
  // The list can shrink without input changing (store switch, a PDF removed),
  // which leaves the stored index past its end; Tab would then set input to
  // undefined (issue #407). Read it clamped instead of resetting on every change.
  const activeSuggestion = Math.min(suggestionIndex, Math.max(documentSuggestions.length - 1, 0));

  /** The document a typed message names, or null.
   *
   * Falls back to a unique substring match so "planck" sends, rather than
   * making the user complete the filename the autocomplete already showed. */
  const resolveDocument = useCallback((text: string): string | null => {
    const query = text.trim().toLowerCase();
    if (!query) return null;
    const exact = documents.find(d => d.toLowerCase() === query);
    if (exact) return exact;
    const contained = documents.filter(d => query.includes(d.toLowerCase()));
    if (contained.length === 1) return contained[0];
    const partial = documents.filter(d => d.toLowerCase().includes(query));
    return partial.length === 1 ? partial[0] : null;
  }, [documents]);

  const acceptSuggestion = useCallback((name: string) => {
    setInput(name);
    setSuggestionIndex(0);
    textareaRef.current?.focus();
  }, []);

  const sendStudy = useCallback(async (text: string) => {
    const doc = resolveDocument(text);
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: text, mode: 'study' };
    const assistantId = (Date.now() + 1).toString();

    if (!doc) {
      // Named before anything is spent: the failure is in the request, and a
      // spinner followed by "not found" would hide that.
      setMessages(prev => [...prev, userMsg, {
        id: assistantId, role: 'assistant', mode: 'study', isError: true,
        content: fill(T.studyNoMatch, { query: text }),
      }]);
      setInput('');
      return;
    }

    setMessages(prev => [...prev, userMsg, {
      id: assistantId, role: 'assistant', content: '', mode: 'study', isStreaming: true,
    }]);
    setInput('');
    setIsLoading(true);
    const started = Date.now();

    try {
      const langName = lang === 'en' ? 'English' : lang === 'ca' ? 'Valencià' : 'Castellano';
      const { status, body } = await api.study(studyKind, doc, langName);
      const elapsed = ((Date.now() - started) / 1000).toFixed(1);
      // The route's validation failures carry a stable code; translate it
      // instead of dropping its English `message` into a Spanish template.
      const studyErrors: Record<string, string> = {
        invalid_kind: T.studyInvalidKind,
        document_required: T.studyDocumentRequired,
        invalid_language: T.studyInvalidLanguage,
        document_not_indexed: T.studyNotIndexed,
        document_without_fragments: T.studyNoFragments,
      };
      const detail = studyErrors[String(body.error)]
        ?? String(body.message ?? body.error ?? status);
      setMessages(prev => prev.map(m => m.id !== assistantId ? m : (
        status === 200 && body.ok
          ? { ...m, isStreaming: false, artifact: body.artifact as StudyArtifact,
              artifactKind: studyKind, content: '',
              metrics: { searchTime: `${elapsed}s`, chunks: 0 } }
          // 422 is the model ignoring the format, not the server breaking, and
          // the two need different advice.
          : { ...m, isStreaming: false, isError: true,
              content: status === 422 ? T.studyMalformed
                                      : fill(T.studyFailed, { error: detail }) }
      )));
    } catch (e) {
      setMessages(prev => prev.map(m => m.id !== assistantId ? m
        : { ...m, isStreaming: false, isError: true, content: fill(T.studyFailed, { error: String(e) }) }));
    } finally {
      setIsLoading(false);
    }
  }, [resolveDocument, studyKind, lang, T]);

  return {
    studyKind, setStudyKind,
    setSuggestionIndex,
    documentSuggestions, activeSuggestion, acceptSuggestion,
    sendStudy,
  };
}
