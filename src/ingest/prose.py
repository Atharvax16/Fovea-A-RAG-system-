r"""
Step 1 — Ingest prose.

GOAL (from ROADMAP.md):
    Get one clean prose document in, chunk it by section, eyeball the chunks.
    Done when each printed chunk is a coherent, self-contained passage.

WHY THIS MATTERS (the learning bit):
    A RAG system never feeds the *whole* document to the language model. It
    retrieves a few small pieces ("chunks") that look relevant to the question.
    So the quality of your answers is capped by the quality of your chunks.
    Two knobs decide that quality:

      1. BOUNDARIES — where you cut. Cut in the middle of an idea and a chunk
         becomes half a thought that means nothing on its own. We cut on
         *section boundaries* because a section is a natural "unit of meaning":
         it's about one thing.

      2. OVERLAP — how much neighbouring text you repeat. If an idea straddles
         a cut, overlap means both chunks still contain enough context to be
         understood. Too little overlap loses context; too much wastes space
         and retrieves near-duplicates. We only need overlap when a single
         section is too big to be one chunk and we have to window it.

PIPELINE in this file:
    load_tex  ->  extract_body  ->  strip_non_prose  ->  clean_latex
              ->  split_into_sections  ->  window_long_sections  ->  [Chunk, ...]

Run it:
    python -m src.ingest.prose
    python -m src.ingest.prose --max-chars 1500 --overlap 200
    python -m src.ingest.prose --file "data/raw/paper (1).tex" --full
"""

from __future__ import annotations  # lets us write type hints like list[Chunk] on 3.11

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

# We only use the Python standard library in Step 1 — no pip installs needed.
# (Embeddings in Step 2 will be the first time we pull in a third-party package.)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
# A dataclass is just a lightweight struct: a named bag of fields. Using one
# (instead of a raw dict) means every chunk has the SAME shape, which makes the
# rest of the system predictable. Later steps will add an `embedding` field, a
# `source_type` field, etc. — this is the object that flows through the system.
@dataclass
class Chunk:
    chunk_id: int          # simple running index, handy for citing later ("chunk #4")
    section_title: str     # e.g. "Methods > Restoration Models" — gives the chunk context
    text: str              # the actual passage that will get embedded/retrieved
    source_file: str       # which file it came from (provenance = trust)
    char_count: int        # quick proxy for "how big"; ~4 chars ≈ 1 token

    def preview(self, width: int = 280) -> str:
        """A short one-glance view for eyeballing in the terminal."""
        body = self.text if len(self.text) <= width else self.text[:width] + " ..."
        return body.replace("\n", " ")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_tex(path: str | Path) -> str:
    """Read the raw .tex file into a single string.

    `encoding="utf-8"` matters: LaTeX papers contain en-dashes, accents, math
    symbols. The wrong encoding silently corrupts those characters, and corrupt
    text produces corrupt embeddings downstream.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Extract the body
# ---------------------------------------------------------------------------
def extract_body(raw: str) -> str:
    r"""Keep only the part of the document a human would actually read.

    A .tex file is mostly *machinery*: \usepackage lines, author blocks, the
    bibliography. None of that is prose worth retrieving. The real content lives
    between \begin{document} and \end{document}, and the bibliography after
    \bibliography{...} is just citation keys, so we cut there too.
    """
    # `re.search` finds the first match; `.start()`/`.end()` give its position.
    start = re.search(r"\\begin\{document\}", raw)
    end = re.search(r"\\end\{document\}", raw)

    # If the markers are missing (not a full LaTeX doc), fall back to the whole
    # string rather than crashing — defensive, so a weird file still does *something*.
    body = raw[start.end():end.start()] if (start and end) else raw

    # Drop everything from \bibliography{...} onward (citation list, not prose).
    body = re.split(r"\\bibliography\{", body)[0]
    return body


# ---------------------------------------------------------------------------
# 3. Strip non-prose (tables / figures)  —  the "one lane per step" rule
# ---------------------------------------------------------------------------
def strip_non_prose(body: str) -> str:
    r"""Remove table and figure environments.

    The ROADMAP is strict: ONE source type per step. Tables (the PSNR/QWK grids)
    are structured data and get their own lane in Step 3, where they'll be looked
    up *exactly* instead of fuzzily matched as text. If we leave a table dumped
    into a prose chunk now, the numbers turn into word-soup that pollutes
    retrieval. So we excise them and leave a breadcrumb in their place.

    Note the `*?` (non-greedy) and `re.DOTALL`: `.` normally stops at newlines,
    and `*` is greedy (would swallow from the first \begin to the LAST \end).
    `re.DOTALL` lets `.` span newlines; `*?` makes it stop at the *nearest* \end.
    """
    env_pattern = re.compile(
        r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}",  # \1 back-reference: same env name
        re.DOTALL,
    )
    body = env_pattern.sub("[NON-PROSE ENVIRONMENT OMITTED -- handled in Step 3 table lane]", body)
    return body


# ---------------------------------------------------------------------------
# 4. Clean LaTeX noise
# ---------------------------------------------------------------------------
def clean_latex(text: str) -> str:
    r"""Turn LaTeX markup into plain readable prose.

    Embedding models were trained on natural language, not on `\textbf{...}` and
    `\cite{he2016resnet}`. That markup is noise: it dilutes the meaning of a
    chunk and can make two unrelated passages look similar just because they
    share the same commands. Cleaning is cheap insurance for retrieval quality.

    Order matters here — we unwrap commands that KEEP their content before we
    delete commands that DON'T, so we never accidentally eat real words.
    """
    # 4a. Drop line comments: a "%" to end-of-line (but not an escaped "\%").
    text = re.sub(r"(?<!\\)%.*", "", text)

    # 4b. Unwrap formatting commands but KEEP the words inside the braces.
    #     \textbf{Methods} -> Methods,  \textit{x} -> x,  \emph{y} -> y
    text = re.sub(r"\\(?:textbf|textit|emph|texttt|textsuperscript)\{([^}]*)\}", r"\1", text)

    # 4c. Delete commands whose *argument* is metadata we don't want as words.
    #     \cite{...}, \ref{...}, \label{...}  -> gone entirely.
    text = re.sub(r"\\(?:cite|ref|label|caption)\{[^}]*\}", "", text)

    # 4d. Convert list scaffolding into readable bullets.
    text = re.sub(r"\\begin\{(itemize|enumerate)\}", "", text)
    text = re.sub(r"\\end\{(itemize|enumerate)\}", "", text)
    text = re.sub(r"\\item\s*", "- ", text)

    # 4e. Un-escape characters that LaTeX requires backslashed.
    for esc, plain in [(r"\%", "%"), (r"\&", "&"), (r"\#", "#"),
                       (r"\_", "_"), (r"\$", "$"), ("``", '"'), ("''", '"')]:
        text = text.replace(esc, plain)

    # 4f. Any remaining bare commands like \maketitle, \IEEEoverride... -> drop.
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # 4g. Collapse runs of whitespace. Embeddings don't care about blank lines,
    #     and tidy text is far easier to eyeball.
    text = re.sub(r"[ \t]+", " ", text)        # multiple spaces -> one
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)  # multiple blank lines -> one
    return text.strip()


# ---------------------------------------------------------------------------
# 5. Split into sections  —  cut on natural boundaries
# ---------------------------------------------------------------------------
def split_into_sections(body: str) -> list[tuple[str, str]]:
    r"""Slice the body at every \section / \subsection heading.

    Returns a list of (title, text) where title carries the hierarchy, e.g.
    "Methods > Restoration Models". Carrying the parent section into the title is
    a small but important trick: a chunk that says only "We evaluate six
    strategies..." is ambiguous, but "Methods > Restoration Models: We evaluate
    six strategies..." is self-contained — exactly the property the ROADMAP asks
    for ("each chunk a coherent, self-contained passage").
    """
    # Also treat the abstract (an environment, not a \section) as its own unit.
    body = re.sub(r"\\begin\{abstract\}", r"\\section{Abstract}", body)
    body = re.sub(r"\\end\{abstract\}", "", body)

    # Find every heading and remember: its level, its title, and WHERE it starts.
    heading_re = re.compile(r"\\(section|subsection)\{([^}]*)\}")
    matches = list(heading_re.finditer(body))
    if not matches:
        return [("(whole document)", body)]

    sections: list[tuple[str, str]] = []
    current_section_title = ""  # remembers the last top-level \section seen

    for i, m in enumerate(matches):
        level, title = m.group(1), m.group(2).strip()

        # The text of this unit runs from the END of this heading to the START
        # of the next heading (or end-of-body for the final unit).
        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[text_start:text_end]

        if level == "section":
            current_section_title = title
            full_title = title
        else:  # subsection -> prefix with its parent for self-containment
            full_title = f"{current_section_title} > {title}" if current_section_title else title

        sections.append((full_title, text))

    return sections


# ---------------------------------------------------------------------------
# 6. Window long sections  —  this is where OVERLAP earns its keep
# ---------------------------------------------------------------------------
def window_long_sections(
    title: str,
    text: str,
    max_chars: int,
    overlap: int,
) -> list[str]:
    r"""Split one section's text into <= max_chars windows that OVERLAP.

    WHY a size cap at all? Embedding models squeeze a whole chunk into ONE
    vector. Cram in too much and that vector becomes a blurry average — it
    matches everything weakly and nothing strongly. Small, focused chunks give
    sharp matches. (~4 characters ≈ 1 token, so max_chars=1200 ≈ ~300 tokens.)

    WHY overlap? If we cut at character 1200 and an idea spans 1150–1300, the
    first chunk gets the setup and the second gets the punchline — and neither
    is retrievable on its own. Repeating the last `overlap` characters at the
    start of the next window keeps that idea intact in at least one chunk.

    We slide a window of `max_chars`, then step forward by (max_chars - overlap)
    so consecutive windows share `overlap` characters. We also nudge each cut to
    the nearest sentence/space so we don't slice a word in half.
    """
    if len(text) <= max_chars:
        return [text]  # short section: no windowing needed, it's already one unit

    windows: list[str] = []
    step = max_chars - overlap          # how far the window advances each time
    if step <= 0:                        # guard against silly args (overlap >= size)
        step = max_chars

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))

        # Try to end on a clean boundary so chunks read naturally. Look back from
        # `end` for a sentence end (". "), then a newline, then a space.
        if end < len(text):
            window_slice = text[start:end]
            for sep in (". ", "\n", " "):
                cut = window_slice.rfind(sep)
                # only accept the boundary if it's reasonably far in (>60%),
                # otherwise we'd make a tiny chunk and re-process the rest forever
                if cut != -1 and cut > max_chars * 0.6:
                    end = start + cut + len(sep)
                    break

        windows.append(text[start:end].strip())
        if end >= len(text):
            break

        # Step back by `overlap` so the next window repeats the tail of this one...
        start = end - overlap
        # ...but snap forward to the next space so we begin on a WHOLE word, not
        # mid-word ("ster than accuracy"). Costs us a few chars of overlap; buys
        # us chunks that actually read like prose from their first character.
        next_space = text.find(" ", start)
        if next_space != -1 and next_space < end:
            start = next_space + 1

    return [w for w in windows if w]  # drop any empties


# ---------------------------------------------------------------------------
# Orchestration — tie the pipeline together
# ---------------------------------------------------------------------------
def chunk_document(
    path: str | Path,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """Run the full Step-1 pipeline and return a flat list of Chunk objects."""
    raw = load_tex(path)
    body = extract_body(raw)
    body = strip_non_prose(body)          # tables/figures -> Step 3, not here
    sections = split_into_sections(body)  # cut on natural boundaries FIRST

    chunks: list[Chunk] = []
    next_id = 0
    for title, section_text in sections:
        # Clean AFTER splitting so the \section markers survive long enough to
        # split on, but the chunk text itself ends up as clean prose.
        cleaned = clean_latex(section_text)
        if not cleaned:
            continue  # skip empty sections (e.g. a heading with no body yet)

        # Only NOW do we window — and only if the section is too big to be one chunk.
        for piece in window_long_sections(title, cleaned, max_chars, overlap):
            chunks.append(
                Chunk(
                    chunk_id=next_id,
                    section_title=title,
                    text=piece,
                    source_file=str(Path(path).name),
                    char_count=len(piece),
                )
            )
            next_id += 1

    return chunks


# ---------------------------------------------------------------------------
# CLI — "build one thing -> observe what it does"
# ---------------------------------------------------------------------------
def _print_chunks(chunks: list[Chunk], full: bool) -> None:
    """Pretty-print chunks so you can eyeball whether each is self-contained."""
    print(f"\n=== {len(chunks)} chunks ===\n")
    for c in chunks:
        print(f"[chunk {c.chunk_id:>2}]  {c.section_title}   ({c.char_count} chars)")
        print("-" * 78)
        print(c.text if full else c.preview())
        print()  # blank line between chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: ingest + chunk thesis prose.")
    parser.add_argument(
        "--file",
        default="data/raw/paper (1).tex",
        help="Path to the .tex source (default: the OMIA/IEEE paper).",
    )
    parser.add_argument("--max-chars", type=int, default=1200,
                        help="Max characters per chunk before windowing (~4 chars/token).")
    parser.add_argument("--overlap", type=int, default=150,
                        help="Characters repeated between consecutive windows.")
    parser.add_argument("--full", action="store_true",
                        help="Print full chunk text instead of a short preview.")
    args = parser.parse_args()

    chunks = chunk_document(args.file, max_chars=args.max_chars, overlap=args.overlap)
    _print_chunks(chunks, full=args.full)

    # A tiny summary makes the observation step quantitative, not just vibes.
    sizes = [c.char_count for c in chunks]
    if sizes:
        print("=== summary ===")
        print(f"file        : {args.file}")
        print(f"chunks      : {len(chunks)}")
        print(f"size (chars): min={min(sizes)}  max={max(sizes)}  avg={sum(sizes)//len(sizes)}")


if __name__ == "__main__":
    main()
