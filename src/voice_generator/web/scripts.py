"""Recording scripts for the Setup page.

The reference transcript must be the exact text the user read, so each
script is a complete, self-contained passage: the user reads the whole
thing, and the setup flow stores that passage's text as
``refs/<name>.txt``.

Why two scripts: the emotional-range paragraph forces pitch, energy
and pacing shifts plus numbers and acronyms; the Harvard sentences are
phonetically balanced (IEEE List 5, 1965 revised list — verified
against the Columbia/McGill copies). The quick-brown-fox line is a
pangram, not a Harvard sentence; it is included as a useful read-aloud
line and labeled as such. The project introduction used to be a third
choice; it was removed in v0.5.3, and its text lived on as
``recording-script.txt``, the demo text the synth page and the A/B
bench rendered. That file was removed in v0.5.22 — the range paragraph
below is now the demo text too (see ``voice_generator.demo_script``).
"""

from __future__ import annotations

from dataclasses import dataclass

from voice_generator.demo_script import DEMO_SCRIPT

# The range paragraph is the demo script as well as a read-aloud text,
# so it lives in the core (``demo_script``) and is reused here, not
# copied: one text, one source of truth.
RANGE_TEXT = DEMO_SCRIPT

HARVARD_TEXT = """The quick brown fox jumps over the lazy dog.
A king ruled the state in the early days.
The ship was torn apart on the sharp reef.
Sickness kept him home the third week.
The wide road shimmered in the hot sun."""


@dataclass(frozen=True)
class Script:
    """One read-aloud text for the Setup page: id, display title, blurb,
    the "why this works" bullets, and the full passage stored as the
    clip's transcript."""

    id: str
    title: str
    blurb: str
    why: list[str]
    text: str


SCRIPTS: list[Script] = [
    Script(
        id="range",
        title="Emotional range",
        blurb=(
            "One paragraph that pushes your voice through questions, "
            "excitement, pauses, and numbers."
        ),
        why=[
            "Questions (upward inflection): “Do you have any idea how "
            "long this took?” forces your pitch to rise at the end of "
            "the sentence.",
            "Excitement (volume and energy): the opening builds to an "
            "exclamation point, capturing your high-energy output.",
            "Pauses and hesitations: “It’s been—what, three years?—” "
            "and “just... mind-blowing” train the model on your cadence, "
            "breathing, and natural conversational delays.",
            "Alphanumeric data: 6:45 AM, V-12, 2023, and 87.5% teach "
            "the model how you pronounce numbers, acronyms, and symbols.",
            "Intimacy (lower register): “But hey, let’s not get ahead of "
            "ourselves” drops your voice into a lower, calmer register.",
        ],
        text=RANGE_TEXT,
    ),
    Script(
        id="harvard",
        title="Harvard sentences",
        blurb="Phonetically balanced read-aloud lines (IEEE List 5) — short, one at a time.",
        why=[
            "Phonetically balanced: the sounds of English in roughly the "
            "proportions they appear in daily speech (IEEE Harvard "
            "Sentences, List 5, 1965 revised list).",
            "Short standalone sentences — easy to read one at a time, "
            "naturally, without slipping into a “reading” voice.",
            "“The quick brown fox…” is a pangram (every letter of the "
            "alphabet), not a Harvard sentence — included as a useful "
            "read-aloud line.",
            "Stand up, use your hands to talk, and smile when the text "
            "calls for it — the model picks up the micro-changes in your "
            "vocal cords.",
        ],
        text=HARVARD_TEXT,
    ),
]


def get_script(script_id: str | None) -> Script:
    """The script for ``script_id``; unknown ids fall back to the range."""
    for script in SCRIPTS:
        if script.id == script_id:
            return script
    return SCRIPTS[0]
