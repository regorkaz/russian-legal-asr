"""Parsers for ground-truth and predicted transcripts."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    speaker: str
    text: str
    start: float | None = None
    end: float | None = None


_GT_LINE = re.compile(r"^\s*\[(?P<speaker>[^\]]+)\]\s*:\s*(?P<text>.*)$")

_PRED_LINE = re.compile(
    r"^\s*\[(?P<speaker>[^\]]+)\]\s+"
    r"\(chunk_(?P<start>[\d.]+)s_(?P<end>[\d.]+)s\)\s*:\s*"
    r"(?P<text>.*)$"
)


def parse_ground_truth(path: str) -> list[Segment]:
    segments: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = _GT_LINE.match(line)
            if not m:
                continue
            segments.append(
                Segment(
                    speaker=m.group("speaker").strip(),
                    text=m.group("text").strip(),
                )
            )
    return segments


def parse_predicted(path: str) -> list[Segment]:
    segments: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = _PRED_LINE.match(line)
            if not m:
                continue
            segments.append(
                Segment(
                    speaker=m.group("speaker").strip(),
                    text=m.group("text").strip(),
                    start=float(m.group("start")),
                    end=float(m.group("end")),
                )
            )
    return segments
