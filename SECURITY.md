# Security policy

## Supported versions

Security fixes land on `main`. There are no maintained release branches, so
update to the latest `main` before reporting.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or a suspected
secret leak. Report it privately through GitHub's private vulnerability
reporting (repository Security tab, Report a vulnerability), so the report
stays private until a fix ships.

## Leaked secrets: rotate first, then report

If a secret (`MONKEYGRAB_OPENAI_API_KEY`, `MONKEYGRAB_HEADLESS_TOKEN`, or
anything from `.env`) may have reached a public place — a commit, an issue,
a log, a screenshot:

1. Rotate or revoke the secret first, at its provider.
2. Then report it privately as above, saying what kind of secret leaked and
   where, without reposting the secret value itself.
3. Never file a public `bug` issue with the file path and commit hash of a
   leak: that advertises the secret while it is still valid.

Day-to-day rules (never commit `.env`, dumps hold no keys, headless token
discipline) live in `docs/standards/security.md`.
