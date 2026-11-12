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
choice; it was removed — it stays in the repo as
``recording-script.txt``, the demo text the synth page and the A/B
bench render.
"""

from __future__ import annotations

from dataclasses import dataclass

# The paragraph is one long line on purpose: the stored transcript must
# be the exact text, with no line breaks the user did not read. It is
# split across adjacent literals (which concatenate with no separator)
# only to keep ruff's line-length rule happy.
RANGE_TEXT = (
    "Honestly, I never expected to find myself here at 6:45 AM, holding "
    "a cup of lukewarm coffee and staring at this screen! Do you have any "
    "idea how long this took? It's been—what, three years?—since the V-12 "
    "project launched in 2023. We saw a massive 87.5% increase in output, "
    "which is just... mind-blowing. But hey, let's not get ahead of "
    "ourselves. If we can maintain this momentum, I genuinely believe the "
    "next phase will change everything."
)

HARVARD_TEXT = """The quick brown fox jumps over the lazy dog.
A king ruled the state in the early days.
The ship was torn apart on the sharp reef.
Sickness kept him home the third week.
The wide road shimmered in the hot sun."""


@dataclass(frozen=True)
class Script:
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
