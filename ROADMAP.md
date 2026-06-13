# Fovea — Roadmap

A multimodal RAG over my diabetic retinopathy thesis materials, built
step by step as a learning project. Research focus: **faithfulness** —
measuring when the assistant states claims *not grounded* in the
retrieved source (LLM/RAG hallucination), which is distinct from the
diffusion hallucination studied inside the thesis itself.

## How we work

A researcher loop, repeated every step:

> **build one thing → observe what it does → log the observation → commit → next**

Two rules that keep it sane:

- **One source type per step.** Prose, tables, code, and images each get
  their own lane, introduced when that lane is built. Never ingest
  several formats at once — if retrieval breaks, the cause must be obvious.
- **One commit per step.** History reads like a lab notebook. Conventions:
  `feat:` new capability, `chore:` setup, `docs:` log/notes, `fix:` repairs.

Every step ends with a `RESEARCH_LOG.md` entry: what was the goal, what did
I observe, what surprised me, what's next.

## Corpus map — what enters when

Nothing is skipped; each source enters at the step built to handle it.

| Material | Unit of meaning | Enters at |
|---|---|---|
| Prose (OMIA paper, then thesis) | passage / section | Step 1–2 |
| Metric tables (PSNR/SSIM/accuracy grids) | record (method, degradation, metric, value) | Step 3 |
| Code (notebooks, `.py`) | function / cell | Step 6 |
| Figures & plots (from paper, then Drive) | caption + image embedding | Step 7 (v2) |

## Steps

### Step 0 — Scaffold ✅
Repo, folder structure, research log, first commit, pushed to GitHub as Fovea.
*Done.*

### Step 1 — Ingest prose
- **Goal:** get one clean prose document in, chunk it by section, eyeball the chunks.
- **Input:** the OMIA paper `.tex` (smallest, cleanest source) → later swap in the thesis.
- **Do:** loader reads the file; split on section boundaries with overlap; print the chunks.
- **Learn:** why chunk *boundaries* and *overlap* decide retrieval quality; what a "unit of meaning" is.
- **Done when:** you can look at the printed chunks and each one is a coherent, self-contained passage.
- **Commit:** `feat: ingest and chunk thesis prose`

### Step 2 — Embed + retrieve
- **Goal:** turn chunks into vectors and build a `retrieve(query)` you can poke.
- **Do:** embed each chunk; store in a local vector index; write `retrieve(query, k)` returning top-k chunks.
- **Learn:** embeddings, similarity search, what "top-k" really returns; where retrieval already fails.
- **Done when:** asking a question returns visibly relevant chunks (and you've found a query where it *doesn't* — note it).
- **Commit:** `feat: add embedding + vector retrieval`

### Step 3 — Table lane
- **Goal:** make metric tables answerable as *exact lookups*, not fuzzy text.
- **Do:** parse your results tables into structured records; query by (method, degradation, metric).
- **Learn:** why structured data must bypass the text chunker; routing a query to the right store.
- **Done when:** "PSNR for Cold Diffusion on noise?" returns the exact cell value.
- **Commit:** `feat: add structured metric-table lookup`

### Step 4 — Grounded generation
- **Goal:** an LLM that answers **only** from retrieved context, with citations.
- **Do:** retrieve → stuff context into a prompt → generate; require a source reference per claim.
- **Learn:** prompt grounding, citation discipline, refusing when the answer isn't in context.
- **Done when:** answers cite which chunk/record they came from, and say "not in corpus" when appropriate.
- **Commit:** `feat: grounded answer generation with citations`

### Step 5 — Faithfulness eval ⭐ research core
- **Goal:** measure *when it hallucinates* — the actual research contribution.
- **Do:** hand-write a set of probe questions (incl. the SwinIR+GAN-is-unsafe case); score each answer for
  groundedness / faithfulness; log failure cases.
- **Learn:** faithfulness vs. relevance metrics (RAGAS-family as a starting reference), building an eval set,
  reading failures like a scientist.
- **Done when:** you have a small table of probes with pass/fail + notes on *why* failures happen.
- **Commit:** `feat: faithfulness eval harness + first results`

### Step 6 — Reranker + code lane
- **Goal:** sharpen retrieval, and bring in code as its own lane.
- **Do:** add a reranker over top-k results; chunk code by function/cell and index it separately.
- **Learn:** cross-encoder reranking; why code needs structure-aware splitting.
- **Commit:** `feat: add reranker` then `feat: add code ingestion lane`

### Step 7 — Figures / VLM lane (v2)
- **Goal:** let the system "see" figures and plots.
- **Do:** caption figures with a vision-language model; embed captions; retrieve the figure with its caption.
- **Learn:** multimodal retrieval; visual-faithfulness (does a claim match what the plot shows?).
- **Commit:** `feat: add figure captioning + multimodal retrieval`

## Where the research lives

The build (Steps 1–4, 6–7) is the *engineering*. The **research** is Step 5
and the visual-faithfulness extension in Step 7: a clean, bounded study of
groundedness across modalities. The through-line tying it to your thesis is
the theme of **trust signals for generative AI** — your diffusion work asks
"did the restorer invent a lesion?"; Fovea asks "did the assistant invent a
finding?" Same question, different layer.

## Status

- [x] Step 0 — scaffold
- [ ] Step 1 — ingest prose  ← *next*
- [ ] Step 2 — embed + retrieve
- [ ] Step 3 — table lane
- [ ] Step 4 — grounded generation
- [ ] Step 5 — faithfulness eval
- [ ] Step 6 — reranker + code lane
- [ ] Step 7 — figures / VLM lane
