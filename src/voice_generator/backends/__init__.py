"""Backend registry. Backends are imported lazily so the core CLI and tests
run on a machine with no torch / no GPU at all."""

from __future__ import annotations

from voice_generator.backends.base import BaseBackend

BACKENDS: dict[str, str] = {
    "mock": "voice_generator.backends.mock_backend:MockBackend",
    "chatterbox": "voice_generator.backends.chatterbox_backend:ChatterboxBackend",
    "qwen3": "voice_generator.backends.qwen_backend:Qwen3Backend",
    "cosyvoice": "voice_generator.backends.cosy_backend:CosyBackend",
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
