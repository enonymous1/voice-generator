"""Backend registry. Backends are imported lazily so the core CLI and tests
run on a machine with no torch / no GPU at all."""

from __future__ import annotations

from voice_generator.backends.base import BaseBackend

BACKENDS: dict[str, str] = {
    "mock": "voice_generator.backends.mock_backend:MockBackend",
    "chatterbox": "voice_generator.backends.chatterbox_backend:ChatterboxBackend",
    # Planned, deliberately not stubbed as if they work:
    # "cosyvoice": separate venv/repo (CosyVoice 3, Apache-2.0, ~3 s clone)
    # "qwen3":     separate venv (Qwen3-TTS, Apache-2.0, ~3 s clone)
}


def available_backends() -> list[str]:
    return sorted(BACKENDS)


def get_backend(name: str, **kwargs: object) -> BaseBackend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r} — available: {', '.join(available_backends())}")
    module_path, class_name = BACKENDS[name].split(":")
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)  # type: ignore[return-value]
