# Code Standards & Process

## Overriding directives (Super important)
Always provide code snippets for the user to run in code blocks not as hyperlinked text.

Never commit or push unless explicitly asked to do so directly every time. 

Apply The Algorithm constantly: (1) Question every requirement. (2) Delete any part of the process you can. (3) Simplify and optimize. (4) Accelerate cycle time. (5) Automate. (6) Find and set great default values to simplify user experience
    (1) The User is smart (assume 145IQ) but stupid in some ways and definitely ignorant in many ways; This means his *intent* and his *phrasing* can sometimes diverge. So freely question requirements, especially stupid ones;
    (2) If we don't need code, we delete it. If a module can be reduced to a function, great. If a function can be reduced to a parameter value someplace, better. If we don't need that parameter, wipe it out with prejudice.
    (3) Our codebase should be like bacterial DNA, simple, elegant, pull only the critically needed things and ruthlessly prunes everything else. Our runtimes should make Carmack proud. If stuff can be written in C or Rust, we do it. Multi-second run loops make us cry. If something can be done 80-90% as well with a 5x simpler algorithm, maybe we should. Optimize, but only after going through steps (1) and (2), optimize optimize. When uncommited changes get over 400-800 lines, seek a natural stopping point and let the user know. 
    (4) We want to present the user with a maximally useful set of choices and sitrep at every iteration. His time and attention are limited, and OODA loops need to be tight. a) numbered questions with labeled trades for easy reply (i.e. uses chose 1d,2a,3c,4c): Ask the user any questions with the trades clearly specified (+/- for each choice and your recommendation)
    b) sitrep: IF you're only partly through an implementation, MAKE THAT CLEAR, and be obvious about what is DONE and what is TODO still; He needs crystal clarity on what works well, what is a shitty stub, and what is not built at all. If possible, try to produce visual artifacts to reflect the stage of work we're in, they're far more information dense than logs or reports, and humans process visual info super well, as you know. 

    (5) Automate. Don't ever ask the user to run large python snippets, or run debuggers manually like a proletarian. No, we produce tooling to enable us both to get metadata and (visual? or json?) artifacts from live code execution, be generous with logging, again this stuff will come in handy for debugging. 
    (6) The user actually hates running CLI commands, especially with ntheen parameters that could fuck up the pipeline if misset. Explore the space a bit and think and find great defaults so that the user can run artifact generation with minimal cognitive load. 


## Branching & PRs
- `main` is protected. Short-lived feature branches.
- PR size: aim < 800 changed lines; larger needs rationale.
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
