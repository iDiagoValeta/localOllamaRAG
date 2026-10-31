#!/usr/bin/env bash
# Remove what tools/install.sh created (issue #559), and nothing else.
#
#   tools/uninstall.sh            # show the plan, ask, remove
#   tools/uninstall.sh --yes      # skip the question
#   tools/uninstall.sh --dry-run  # show the plan only
#
# The install manifest is the whole authority: a path or Ollama model that is
# not listed there is never touched. System packages, Ollama itself and any
# model, venv or index that existed before install.sh ran are not listed.
set -euo pipefail

# shellcheck source=tools/_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $arg (see --help)" ;;
  esac
done

if [ ! -f "$MANIFEST" ]; then
  say "No install manifest at $MANIFEST: nothing recorded as installed, nothing to remove."
  exit 0
fi

owner="$(awk -F'\t' '$1 == "repo" { print $2; exit }' "$MANIFEST")"
[ "$owner" = "$REPO_ROOT" ] \
  || die "the manifest belongs to $owner, not this checkout ($REPO_ROOT). Run that checkout's tools/uninstall.sh."

# A hand-edited or corrupted manifest must not be able to aim rm -rf at the
# filesystem root or the home directory.
safe_path() {
  case "$1" in
    /*) ;;
    *) return 1 ;;
  esac
  [ "$1" != "/" ] && [ "$1" != "$HOME" ] && [ "$(awk -F/ '{ print NF - 1 }' <<< "$1")" -ge 2 ]
}

paths=(); dirs=(); models=()
while IFS=$'\t' read -r kind value; do
  case "$kind" in
    path) paths+=("$value") ;;
    dir) dirs+=("$value") ;;
    ollama-model) models+=("$value") ;;
    repo|""|\#*) ;;
    *) warn "ignoring unknown manifest entry kind: $kind" ;;
  esac
done < "$MANIFEST"

say "Will remove (from $MANIFEST):"
for p in "${paths[@]+"${paths[@]}"}"; do
  if ! safe_path "$p"; then say "  REFUSED (unsafe path in manifest): $p"
  elif [ -e "$p" ] || [ -L "$p" ]; then say "  $p"
  else say "  $p (already gone)"; fi
done
for d in "${dirs[@]+"${dirs[@]}"}"; do say "  $d (only if empty afterwards)"; done
for m in "${models[@]+"${models[@]}"}"; do say "  Ollama model $m (ollama rm)"; done
say "Left in place: Ollama and its other models, system packages, llama.cpp, rag/settings.json,"
say "the MinerU model weights in the Hugging Face cache, and any path not listed above."

[ "$DRY_RUN" -eq 0 ] || exit 0

if [ "$ASSUME_YES" -eq 0 ]; then
  [ -t 0 ] || die "not a terminal and no --yes: refusing to delete without confirmation."
  read -r -p "Remove all of the above? [y/N] " answer
  case "$answer" in y|Y|yes) ;; *) say "Aborted."; exit 1 ;; esac
fi

# Entries that could not be removed stay in a rewritten manifest, so a re-run
# retries them instead of forgetting them.
remaining="$(mktemp)"
printf '%s\t%s\n' repo "$REPO_ROOT" > "$remaining"
failed=0

for p in "${paths[@]+"${paths[@]}"}"; do
  safe_path "$p" || { printf 'path\t%s\n' "$p" >> "$remaining"; failed=1; continue; }
  rm -rf -- "$p" || { printf 'path\t%s\n' "$p" >> "$remaining"; failed=1; }
done

for d in "${dirs[@]+"${dirs[@]}"}"; do
  safe_path "$d" || continue
  rmdir -- "$d" 2>/dev/null || true
  # The default models dir sits in a monkeygrab/ directory install may have created too.
  if [ "$(basename "$d")" = "models" ] && [ "$(basename "$(dirname "$d")")" = "monkeygrab" ]; then
    rmdir -- "$(dirname "$d")" 2>/dev/null || true
  fi
  [ ! -d "$d" ] || warn "$d is not empty (files this script did not create); left in place."
done

# An unreachable Ollama must not read as "the model is already gone": that
# would drop it from the manifest and leave gigabytes nobody tracks.
if [ "${#models[@]}" -gt 0 ]; then
  if command -v ollama >/dev/null 2>&1 && ollama list >/dev/null 2>&1; then
    for m in "${models[@]}"; do
      if ollama show "$m" >/dev/null 2>&1 && ! ollama rm "$m"; then
        warn "could not remove Ollama model $m; kept in the manifest."
        printf 'ollama-model\t%s\n' "$m" >> "$remaining"
        failed=1
      fi
    done
  else
    warn "Ollama is not reachable; its models stay in the manifest. Start it (ollama serve) and re-run."
    for m in "${models[@]}"; do printf 'ollama-model\t%s\n' "$m" >> "$remaining"; done
    failed=1
  fi
fi

if [ "$failed" -eq 0 ]; then
  rm -f -- "$MANIFEST" "$remaining"
  rmdir -- "$STATE_DIR" 2>/dev/null || true
  say "Uninstalled."
else
  mv -- "$remaining" "$MANIFEST"
  die "some entries could not be removed; they are still listed in $MANIFEST. Fix the cause and re-run."
fi
