"""WER, CER, speaker accuracy via word-level alignment."""

import re
from collections import Counter, defaultdict
from statistics import mean

import jiwer

from metrics.parse import Segment

_OPAQUE_LABEL = re.compile(r"^SPEAKER_\d+$")

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str, fold_yo: bool = True) -> str:
    text = text.lower()
    if fold_yo:
        text = text.replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def _to_tagged_words(segments: list[Segment]) -> list[tuple[str, str]]:
    """Flatten segments into a list of (word, speaker) pairs, preserving order."""
    tagged: list[tuple[str, str]] = []
    for seg in segments:
        for w in normalize(seg.text).split():
            if w:
                tagged.append((w, seg.speaker))
    return tagged


def has_opaque_diarization_labels(hyp: list[Segment]) -> bool:
    """True if hyp uses unsupervised diarization labels (SPEAKER_0, SPEAKER_1, …)."""
    return any(_OPAQUE_LABEL.match(s.speaker) for s in hyp)


def relabel_diarization_speakers(hyp: list[Segment], ref: list[Segment]) -> tuple[list[Segment], dict]:
    """Map opaque SPEAKER_N labels to ground-truth labels via best word-overlap.

    Greedy assignment over a (hyp_spk × ref_spk) confusion matrix derived from
    jiwer's word-level alignment of the hypothesis and reference texts. Mirrors
    the standard "optimal label assignment" step in diarization evaluation.
    Hyp clusters that don't end up in the assignment are mapped to "Unknown".
    """
    ref_tagged = _to_tagged_words(ref)
    hyp_tagged = _to_tagged_words(hyp)
    if not ref_tagged or not hyp_tagged:
        return hyp, {}

    out = jiwer.process_words(
        " ".join(w for w, _ in ref_tagged),
        " ".join(w for w, _ in hyp_tagged),
    )
    chunks = out.alignments[0] if out.alignments else []

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for ch in chunks:
        if ch.type not in ("equal", "substitute"):
            continue
        n = ch.ref_end_idx - ch.ref_start_idx
        for i in range(n):
            r_spk = ref_tagged[ch.ref_start_idx + i][1]
            h_spk = hyp_tagged[ch.hyp_start_idx + i][1]
            confusion[h_spk][r_spk] += 1

    pairs = sorted(
        ((cnt, h, r) for h, rs in confusion.items() for r, cnt in rs.items()),
        reverse=True,
    )
    used_ref: set[str] = set()
    mapping: dict[str, str] = {}
    for _, h, r in pairs:
        if h in mapping or r in used_ref:
            continue
        mapping[h] = r
        used_ref.add(r)
    for h in confusion:
        mapping.setdefault(h, "Unknown")

    relabeled = [
        Segment(
            speaker=mapping.get(s.speaker, s.speaker),
            text=s.text,
            start=s.start,
            end=s.end,
        )
        for s in hyp
    ]
    return relabeled, mapping


def compute_asr_and_speaker(hyp: list[Segment], ref: list[Segment]) -> dict:
    ref_tagged = _to_tagged_words(ref)
    hyp_tagged = _to_tagged_words(hyp)

    result: dict = {
        "ref_words": len(ref_tagged),
        "hyp_words": len(hyp_tagged),
    }

    if not ref_tagged:
        result.update({"wer": None, "cer": None, "speaker_accuracy": None, "per_speaker": {}})
        return result

    ref_str = " ".join(w for w, _ in ref_tagged)
    hyp_str = " ".join(w for w, _ in hyp_tagged)

    out = jiwer.process_words(ref_str, hyp_str)
    cer = jiwer.cer(ref_str, hyp_str)

    correct_speaker = 0
    compared = 0
    subs = dels = ins = eq = 0
    per_speaker: dict[str, dict[str, int]] = {}

    # jiwer.process_words returns alignments as a list[list[AlignmentChunk]] (one per utterance)
    alignment_lists = out.alignments
    chunks = alignment_lists[0] if alignment_lists else []

    for ch in chunks:
        n_ref = ch.ref_end_idx - ch.ref_start_idx
        n_hyp = ch.hyp_end_idx - ch.hyp_start_idx
        if ch.type == "equal":
            eq += n_ref
            for i in range(n_ref):
                r_spk = ref_tagged[ch.ref_start_idx + i][1]
                h_spk = hyp_tagged[ch.hyp_start_idx + i][1]
                stats = per_speaker.setdefault(r_spk, {"total": 0, "correct": 0})
                stats["total"] += 1
                compared += 1
                if r_spk == h_spk:
                    correct_speaker += 1
                    stats["correct"] += 1
        elif ch.type == "substitute":
            subs += n_ref
            for i in range(n_ref):
                r_spk = ref_tagged[ch.ref_start_idx + i][1]
                h_spk = hyp_tagged[ch.hyp_start_idx + i][1]
                stats = per_speaker.setdefault(r_spk, {"total": 0, "correct": 0})
                stats["total"] += 1
                compared += 1
                if r_spk == h_spk:
                    correct_speaker += 1
                    stats["correct"] += 1
        elif ch.type == "delete":
            dels += n_ref
        elif ch.type == "insert":
            ins += n_hyp

    result.update({
        "wer": out.wer,
        "cer": cer,
        "equal": eq,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "speaker_accuracy": correct_speaker / compared if compared else None,
        "speaker_compared_words": compared,
        "speaker_correct_words": correct_speaker,
        "per_speaker": {
            spk: {
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "ref_words": v["total"],
            }
            for spk, v in per_speaker.items()
        },
    })
    return result


def simulate_streaming_latency(chunks: list[dict]) -> None:
    """Annotate chunks with sim_{asr,id,final}_latency_s under a virtual real-time stream.

    Models a system where chunks arrive in audio time at ``end_s`` of each chunk,
    and ASR + Speaker workers each process their queues serially in parallel.
    Final latency also enforces seq-ordered flushing.

    Mutates ``chunks`` in place; assumes they are seq-ordered.
    """
    asr_busy = 0.0
    spk_busy = 0.0
    flush_busy = 0.0

    for c in chunks:
        end_s = c.get("end_s")
        asr_proc = c.get("asr_processing_s")
        spk_proc = c.get("speaker_processing_s")
        if end_s is None or asr_proc is None or spk_proc is None:
            c["sim_asr_latency_s"] = None
            c["sim_id_latency_s"] = None
            c["sim_final_latency_s"] = None
            continue

        arrival = end_s
        asr_done = max(arrival, asr_busy) + asr_proc
        spk_done = max(arrival, spk_busy) + spk_proc
        both_done = max(asr_done, spk_done)
        flush_done = max(both_done, flush_busy)

        c["sim_asr_latency_s"] = asr_done - arrival
        c["sim_id_latency_s"] = spk_done - arrival
        c["sim_final_latency_s"] = flush_done - arrival

        asr_busy = asr_done
        spk_busy = spk_done
        flush_busy = flush_done


def summarize_timings(timings: dict) -> dict:
    chunks = timings.get("chunks", [])
    if not chunks:
        return {}

    chunks = sorted(chunks, key=lambda c: c.get("seq", 0))
    realtime_factor = timings.get("realtime_factor")
    use_simulated = realtime_factor is not None and float(realtime_factor) == 0.0
    warmup_trim = int(timings.get("warmup_trim") or 0)

    if use_simulated:
        # Simulation needs the full ordered sequence — earlier chunks set the queue
        # state seen by later ones. Trim happens only at the aggregation step.
        simulate_streaming_latency(chunks)
        asr_field, id_field, final_field = "sim_asr_latency_s", "sim_id_latency_s", "sim_final_latency_s"
        latency_mode = "simulated"
    else:
        asr_field, id_field, final_field = "asr_latency_s", "id_latency_s", "final_latency_s"
        latency_mode = "real"

    stats_chunks = chunks[warmup_trim:] if warmup_trim < len(chunks) else []

    def _stats(field: str) -> tuple[float | None, float | None]:
        vals = [c[field] for c in stats_chunks if c.get(field) is not None]
        return (mean(vals), max(vals)) if vals else (None, None)

    asr_mean, asr_max = _stats(asr_field)
    id_mean, id_max = _stats(id_field)
    final_mean, final_max = _stats(final_field)
    durations = [c["duration_s"] for c in chunks if c.get("duration_s") is not None]

    audio = timings.get("audio_duration_s") or 0.0
    wall = timings.get("wall_clock_s") or 0.0

    return {
        "audio_duration_s": round(audio, 2),
        "wall_clock_s": round(wall, 2),
        "rtf": (wall / audio) if audio else None,
        "n_chunks": len(chunks),
        "n_chunks_used_for_latency": len(stats_chunks),
        "warmup_trim": warmup_trim,
        "realtime_factor": realtime_factor,
        "latency_mode": latency_mode,
        "asr_latency_mean_s": asr_mean,
        "asr_latency_max_s": asr_max,
        "id_latency_mean_s": id_mean,
        "id_latency_max_s": id_max,
        "final_latency_mean_s": final_mean,
        "final_latency_max_s": final_max,
        "chunk_duration_mean_s": mean(durations) if durations else None,
    }
