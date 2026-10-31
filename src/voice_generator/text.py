"""Script chunking: TTS models degrade on long inputs, so a script is split
into sentence-sized chunks, synthesized one at a time, then concatenated."""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def chunk_script(text: str, *, max_chars: int = 300) -> list[str]:
    """Split text into chunks of at most ~max_chars, on sentence boundaries first.

    Sentences are packed greedily into chunks; a single sentence longer
    than ``max_chars`` is hard-split on commas, then spaces, then
    arbitrarily. ``max_chars`` must be positive: zero or negative would
    make the hard-split loop spin forever (caught here, not in the CLI).
    Known limitation, by design: the sentence regex also splits after
    abbreviations ("Dr. Smith") and decimals — fixing that properly needs
    a sentence tokenizer, which the stdlib does not provide.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:  # one very long sentence: hard-split on commas / spaces
            if current:
                chunks.append(current)
                current = ""
            rest = sentence
            while len(rest) > max_chars:
                cut = rest.rfind(", ", 0, max_chars)
                if cut < max_chars // 2:
                    cut = rest.rfind(" ", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                chunks.append(rest[:cut].strip())
                rest = rest[cut:].strip(" ,")
            if rest:
                current = rest
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks
