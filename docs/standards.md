# Code Standards & Process

## Respectful interaction with the User
Always provide code snippets for the user to run in code blocks not as hyperlinked text.

Never commit or push unless explicitly asked to do so directly every time. 


## Branching & PRs
- `main` is protected. Short-lived feature branches.
- PR size: aim < 400 changed lines; larger needs rationale.
- Every PR must include: tests, docs/comments, passing CI.

## Testing
- `pytest -q` locally; CI runs lint + type + tests.
- Minimum coverage (line): 70% for v0.1; raise later.

## Determinism
- All stages accept `seed` and/or explicit RNG.
- Every run writes `out/map.json` capturing: seed, config, git SHA, lib versions.

## Security & Supply Chain
- Pin versions in `requirements.txt`.
- No network calls in core; reproducible offline.

## AI Assistants (Codex/Grok) Guidance
- Write small, verifiable units with docstrings + types.
- Prefer pure functions; keep I/O thin.
- Always add/modify tests alongside code.
- Include performance notes if a loop is O(N×M).
- Never add dependencies without justification in PR description.

## Licensing & Attribution
- MIT license.
