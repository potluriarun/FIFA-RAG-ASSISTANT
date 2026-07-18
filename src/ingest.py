"""Pipeline A: load PDFs, chunk by document structure, embed, and store in Chroma.

Run: python src/ingest.py
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "fifa_rules"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000       # target max characters per chunk
CHUNK_OVERLAP = 150     # overlap used only when a single paragraph must be sliced

# Three independent structural signals, since pypdf's per-page text doesn't
# reliably preserve blank lines and different rulebooks format headings
# (and even different Laws/sections within the same rulebook) inconsistently:
#
# 1. A section-divider page: short, non-numeric page text with no real body
#    content — a large stylised heading printed alone, e.g. "Law 2" or
#    "Video assistant referee (VAR) protocol".
DIVIDER_MAX_CHARS = 60
#
# 2. A running header repeated on some (not all) content pages, e.g.
#    "Laws of the Game 2025/26   |  Law 1  |  The Field of Play". Where
#    present, close to a divider, it gives us the section's title, which
#    the divider page alone doesn't have.
RUNNING_HEADER_RE = re.compile(r"Laws of the Game\s+\d{4}/\d{2}\s*\|\s*(.+)")
#
# 3. An ALL-CAPS heading starting its own line, e.g. "ARTICLE 5: ...".
#    Requires literal uppercase so it doesn't fire on lowercase/mixed-case
#    inline cross-references like "as set out in Article 5.1".
CAPS_HEADER_RE = re.compile(r"^(LAW|ARTICLE|SECTION)\s+\d+[A-Z]?\b.*")


def detect_divider(page_text: str) -> str | None:
    """A page is a divider/title page if its entire text is short and not
    just a bare page number."""
    normalized = " ".join(page_text.split())
    if normalized and not normalized.isdigit() and len(normalized) <= DIVIDER_MAX_CHARS:
        return normalized
    return None


def detect_page_header(page_text: str) -> str | None:
    """Check the first ~200 chars of a page for the repeated running-header
    pattern used by e.g. the Laws of the Game PDF."""
    match = RUNNING_HEADER_RE.search(page_text[:200])
    if not match:
        return None
    label = match.group(1).splitlines()[0]
    return " ".join(label.split())[:100]


def detect_line_header(line: str) -> str | None:
    """Check a single line for an ALL-CAPS 'ARTICLE N: ...' style heading."""
    if CAPS_HEADER_RE.match(line):
        return " ".join(line.split())[:100]
    return None


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, page_text), ...], 1-indexed pages, blank pages dropped."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def split_into_sections(pages: list[tuple[int, str]]) -> list[dict]:
    """Group page text into sections, breaking hard at each detected header
    (either a page-level running header or a line-level ALL-CAPS heading)."""
    if not pages:
        return []

    sections = []
    current = {"header": None, "page": pages[0][0], "text": ""}
    divider_seen = False

    def start_new_section(header: str, page_num: int):
        nonlocal current
        if current["text"].strip():
            sections.append(current)
        current = {"header": header, "page": page_num, "text": ""}

    for page_num, page_text in pages:
        divider = detect_divider(page_text)
        if divider:
            # Divider pages carry no body text worth keeping (they're just
            # a short stylised heading), so start the section on the next
            # page. Once we've seen one, running headers only ever enrich
            # the current section (see below) — they never start a new one,
            # since the same running header can resurface mid-section on a
            # later page without that being a real section change (e.g. the
            # VAR protocol chapter re-stamps its own header on a later page).
            divider_seen = True
            start_new_section(divider, page_num + 1)
            continue

        page_header = detect_page_header(page_text)
        if page_header:
            if not divider_seen:
                prefix = page_header.split("|")[0].strip()
                current_prefix = (current["header"] or "").split("|")[0].strip()
                if prefix != current_prefix:
                    start_new_section(page_header, page_num)
            elif page_num - current["page"] <= 2 and "|" not in (current["header"] or ""):
                # Upgrade a bare divider label with its title, but only
                # right after the divider, and only if we don't already
                # have a title (some pages' headers extract truncated).
                current["header"] = page_header

        for line in page_text.splitlines():
            line = line.strip()
            if not line:
                continue
            line_header = detect_line_header(line)
            if line_header:
                start_new_section(line_header, page_num)
            current["text"] += line + "\n"

    if current["text"].strip():
        sections.append(current)
    return sections


def chunk_section(text: str) -> list[str]:
    """Merge lines up to CHUNK_SIZE, never splitting a line (so never cuts a
    word in half). pypdf gives us wrapped text lines, not real paragraphs
    (blank lines aren't reliably preserved), so lines are the safest
    mergeable unit. When a chunk boundary is hit, trailing lines from the
    previous chunk are carried forward (up to CHUNK_OVERLAP chars) so
    context isn't lost at the seam."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for line in lines:
        if buf and buf_len + len(line) + 1 > CHUNK_SIZE:
            chunks.append("\n".join(buf))
            overlap_lines: list[str] = []
            overlap_len = 0
            for prev in reversed(buf):
                if overlap_len + len(prev) > CHUNK_OVERLAP:
                    break
                overlap_lines.insert(0, prev)
                overlap_len += len(prev) + 1
            buf, buf_len = overlap_lines, overlap_len

        buf.append(line)
        buf_len += len(line) + 1

    if buf:
        chunks.append("\n".join(buf))

    return chunks


def build_chunks(pdf_path: Path) -> list[dict]:
    pages = extract_pages(pdf_path)
    sections = split_into_sections(pages)

    records = []
    for section in sections:
        for i, chunk_text in enumerate(chunk_section(section["text"])):
            records.append({
                "text": chunk_text,
                "source": pdf_path.name,
                "page": section["page"],
                "law": section["header"] or "",
                "chunk_index": i,
            })
    return records


def main():
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {PDF_DIR}. Add rulebook PDFs there first.")

    print(f"Found {len(pdf_paths)} PDF(s): {[p.name for p in pdf_paths]}")

    all_records = []
    total_pages = 0
    for pdf_path in pdf_paths:
        page_count = len(PdfReader(str(pdf_path)).pages)
        records = build_chunks(pdf_path)
        print(f"  {pdf_path.name}: {page_count} pages -> {len(records)} chunks")
        all_records.extend(records)
        total_pages += page_count

    print(f"Total: {total_pages} pages across {len(pdf_paths)} documents -> {len(all_records)} chunks")
    print(f"Embedding with {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(
        [r["text"] for r in all_records],
        show_progress_bar=True,
        batch_size=32,
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids = [f"{r['source']}_p{r['page']}_c{r['chunk_index']}_{i}" for i, r in enumerate(all_records)]
    metadatas = [{"source": r["source"], "page": r["page"], "law": r["law"]} for r in all_records]
    documents = [r["text"] for r in all_records]

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print(f"Stored {len(ids)} chunks in Chroma at {CHROMA_DIR}")

    print("\n--- Sample chunks (checkpoint: verify source/page/law look right) ---")
    for r in all_records[:3]:
        print(f"[{r['source']} p.{r['page']} law={r['law'] or '(none)'}] {r['text'][:200]}...\n")


if __name__ == "__main__":
    main()
