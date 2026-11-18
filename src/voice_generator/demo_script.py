"""The project's demo script — the one canonical passage the A/B bench
renders and the listen page compares across backends.

Formerly ``recording-script.txt`` in the project root (the project
introduction, 133 words); that file was removed 2026-11-18 and the
emotional-range paragraph — already the Setup page's default read-aloud
text — took its place as the demo text, so there is one text, not two.
The paragraph is one long line on purpose: the stored transcript must
be the exact text, with no line breaks the reader did not read. It is
split across adjacent literals (which concatenate with no separator)
only to keep ruff's line-length rule happy.
"""

from __future__ import annotations

DEMO_SCRIPT = (
    "Honestly, I never expected to find myself here at 6:45 AM, holding "
    "a cup of lukewarm coffee and staring at this screen! Do you have any "
    "idea how long this took? It's been—what, three years?—since the V-12 "
    "project launched in 2023. We saw a massive 87.5% increase in output, "
    "which is just... mind-blowing. But hey, let's not get ahead of "
    "ourselves. If we can maintain this momentum, I genuinely believe the "
    "next phase will change everything."
)
