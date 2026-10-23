"""Enables `python -m voice_generator ...` — used by the A/B compare tool,
which drives each backend through its own venv's interpreter."""

from voice_generator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
