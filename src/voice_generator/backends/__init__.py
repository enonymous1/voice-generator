"""Backend registry. Backends are imported lazily so the core CLI and tests
run on a machine with no torch / no GPU at all."""

from __future__ import annotations

from voice_generator.backends.base import BaseBackend

BACKENDS: dict[str, str] = {
    "mock": "voice_generator.backends.mock_backend:MockBackend",
    "chatterbox": "voice_generator.backends.chatterbox_backend:ChatterboxBackend",
    "qwen3": "voice_generator.backends.qwen_backend:Qwen3Backend",
    "cosyvoice": "voice_generator.backends.cosy_backend:CosyBackend",
    "voxcpm": "voice_generator.backends.voxcpm_backend:VoxCPM2Backend",
}


# Import name each backend's engine package is known by, for `doctor`
# probes. Kept separate from BACKENDS because the registry keys are CLI
# names ("qwen3") while the import names are upstream's ("qwen_tts").
BACKEND_IMPORTS: dict[str, str] = {
    "mock": "voice_generator.backends.mock_backend",
    "chatterbox": "chatterbox",
    "qwen3": "qwen_tts",
    "cosyvoice": "cosyvoice",
    "voxcpm": "voxcpm",
}


def available_backends() -> list[str]:
    return sorted(BACKENDS)


def get_backend(name: str, **kwargs: object) -> BaseBackend:
    """Instantiate a backend by CLI name. kwargs go to its constructor.

    Import is lazy (module import happens here, not at package import),
    which is what keeps the core CLI torch-free. Raises ``ValueError``
    for an unknown name and lets the backend's own ``RuntimeError``
    surface when its engine package is missing.
    """
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r} — available: {', '.join(available_backends())}")
    module_path, class_name = BACKENDS[name].split(":")
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)  # type: ignore[return-value]
