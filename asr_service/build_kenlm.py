"""Build a word-level KenLM 4-gram on Russian legal text.

Run inside the asr_service container:
    docker compose run --rm asr_worker python /app/build_kenlm.py

Output:
    /app/kenlm/legal.bin       binary KenLM (loaded by pyctcdecode at runtime)
    /app/kenlm/unigrams.txt    top-N words by frequency (pyctcdecode vocab)

The LM is word-level, NOT SP-piece-level: pyctcdecode operates on BPE pieces
internally but queries kenlm with the words it assembles. Training kenlm on
words matches that contract; training on pieces would give a kenlm whose
vocabulary doesn't overlap pyctcdecode's unigrams.

Corpus: irlspbru/RusLawOD (HF dataset, CC BY-NC-SA 4.0, ~194M tokens).
We subsample to MAX_DOCS docs for speed; full corpus is overkill for a 4-gram.
Steps are idempotent — if corpus.txt already exists on disk we skip the
HF download.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

MAX_DOCS = int(os.getenv("KENLM_MAX_DOCS", "20000"))   # 20k docs ≈ 200–400 MB text
MIN_LINE_CHARS = 20
TOP_UNIGRAMS = int(os.getenv("KENLM_TOP_UNIGRAMS", "100000"))  # top-N words for pyctcdecode
DATASET = "irlspbru/RusLawOD"
OUT_DIR = Path("/app/kenlm")
RAW_TXT = OUT_DIR / "corpus.txt"
ARPA = OUT_DIR / "legal.arpa"
BIN = OUT_DIR / "legal.bin"
UNIGRAMS = OUT_DIR / "unigrams.txt"

# Strip article/section headers and tables-of-contents that pollute n-grams.
_SKIP_LINE_RE = re.compile(
    r"^(статья\s+\d+|часть\s+\d+|пункт\s+\d+|глава\s+\d+|раздел\s+\d+|приложение\s+\d+)\.?\s*$",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
# Word-tokenizer for unigrams: keep cyrillic letters + digits + hyphen; drop punct.
_WORD_RE = re.compile(r"[а-яё][а-яё0-9\-]*", re.IGNORECASE | re.UNICODE)


def clean_text(text: str) -> list[str]:
    """Return non-trivial lines from a document, lowercased and whitespace-normalized."""
    out: list[str] = []
    for line in text.split("\n"):
        line = _WS_RE.sub(" ", line).strip()
        if len(line) < MIN_LINE_CHARS:
            continue
        if _SKIP_LINE_RE.match(line):
            continue
        out.append(line.lower())
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_TXT.exists() and RAW_TXT.stat().st_size > 0:
        print(f"[kenlm] Reusing existing corpus: {RAW_TXT} ({RAW_TXT.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        from datasets import load_dataset

        print(f"[kenlm] Loading dataset {DATASET}...")
        ds = load_dataset(DATASET, split="train", streaming=True)

        print(f"[kenlm] Streaming up to {MAX_DOCS} docs -> {RAW_TXT}")
        n_lines = 0
        with RAW_TXT.open("w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                if i >= MAX_DOCS:
                    break
                text = row.get("textIPS") or ""
                for line in clean_text(text):
                    f.write(line + "\n")
                    n_lines += 1
                if i % 1000 == 0:
                    print(f"[kenlm]   docs={i}  lines={n_lines}")
        print(f"[kenlm] Wrote {n_lines} lines to {RAW_TXT} ({RAW_TXT.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"[kenlm] Running lmplz -o 4 (word-level) -> {ARPA}")
    with RAW_TXT.open("rb") as fin, ARPA.open("wb") as fout:
        subprocess.run(
            ["lmplz", "-o", "4", "--discount_fallback"],
            stdin=fin, stdout=fout, check=True,
        )
    print(f"[kenlm] ARPA size: {ARPA.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"[kenlm] Running build_binary -> {BIN}")
    subprocess.run(["build_binary", str(ARPA), str(BIN)], check=True)
    print(f"[kenlm] Binary size: {BIN.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"[kenlm] Extracting unigrams (top {TOP_UNIGRAMS} by frequency) -> {UNIGRAMS}")
    extract_unigrams()
    print(f"[kenlm] DONE. Set KENLM_PATH={BIN}, UNIGRAMS_PATH={UNIGRAMS}, LM_MODE=kenlm.")


def extract_unigrams() -> None:
    """Whitespace-tokenize the raw corpus, count word frequencies, keep top-N."""
    from collections import Counter

    counts: Counter[str] = Counter()
    with RAW_TXT.open("r", encoding="utf-8") as f:
        for line in f:
            for word in _WORD_RE.findall(line):
                counts[word] += 1

    print(f"[kenlm]   unique words: {len(counts)}")
    top = counts.most_common(TOP_UNIGRAMS)
    with UNIGRAMS.open("w", encoding="utf-8") as f:
        for word, _ in top:
            f.write(word + "\n")
    print(f"[kenlm]   kept top {len(top)} -> {UNIGRAMS} ({UNIGRAMS.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
