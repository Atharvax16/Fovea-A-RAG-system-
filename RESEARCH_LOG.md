# Research Log

The lab notebook. One entry per step: goal, what I observed, what
surprised me, what's next. Newest entries at the top.

---

## Step 1 — Ingest prose  (2026-06-13)

**Goal.** Get one clean prose document in, chunk it on section boundaries
with overlap, and eyeball the chunks until each is a coherent, self-contained
passage.

**Built.** `src/ingest/prose.py` — a stdlib-only pipeline:
`load_tex -> extract_body -> strip_non_prose -> split_into_sections ->
clean_latex -> window_long_sections -> [Chunk]`. Source:
`data/raw/paper (1).tex` (the IEEE/OMIA conference paper — smallest, cleanest
`.tex`). Run with `python -m src.ingest.prose` (add `--full` to see whole chunks).

**Observed.**
- 22 chunks. Size: min=223, max=1179, avg≈587 chars (target cap 1200, overlap 150).
- Section-aware splitting gives clean units. Subsections carry their parent in
  the title (e.g. `Methods > Restoration Models`), so a chunk that starts "We
  evaluate six restoration strategies..." is still self-contained.
- Only longer sections (Abstract, Introduction, Related Work) get windowed into
  2–3 overlapping pieces; short subsections stay as one chunk each.
- The "one lane per step" rule held: the three results tables were excised and
  replaced with a `[NON-PROSE ENVIRONMENT OMITTED -- handled in Step 3]` marker,
  so no PSNR/QWK number-soup leaked into prose chunks.

**Surprised me.**
- Overlap was visibly working but ugly at first: windowed chunks began mid-word
  ("ster than accuracy") because the step-back by `overlap` chars ignores word
  boundaries. Fixed by snapping the next window's start forward to the next
  space. Lesson: where you *resume* matters as much as where you *cut*.
- The Windows console printed `?`/garbage for the em-dash — a terminal encoding
  artifact, not a data problem (the file is UTF-8). Switched printed markers to
  ASCII to keep the eyeball step honest.

**Known rough edges (deliberately deferred).**
- `clean_latex` strips `\cite{...}` entirely, so sentences read "...like ResNet
  , ConvNeXt , and..." with floating commas. Fine for now; revisit if it hurts
  retrieval in Step 2.
- Inline math is mostly stripped; acceptable for this prose-only paper.

**Next — Step 2: embed + retrieve.** Turn these chunks into vectors, store them
in a local index, and write `retrieve(query, k)`. Goal: find a query that
returns visibly relevant chunks — and one where it doesn't, and note why.
