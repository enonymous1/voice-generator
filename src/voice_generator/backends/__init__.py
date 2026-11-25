"""Backend registry. Backends are imported lazily so the core CLI and tests
run on a machine with no torch / no GPU at all."""

from __future__ import annotations

import importlib
import inspect

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
    """CLI names of all registered backends, sorted."""
    return sorted(BACKENDS)


def get_backend(name: str, **kwargs: object) -> BaseBackend:
    """Instantiate a backend by CLI name. kwargs go to its constructor.

    Import is lazy (module import happens here, not at package import),
    which is what keeps the core CLI torch-free. Raises ``ValueError``
    for an unknown name, ``RuntimeError`` for a misregistered backend
    (class not found, or an option the constructor does not accept —
    both name the backend and the valid options, instead of surfacing a
    bare ``AttributeError``/``TypeError`` from reflection), and lets the
    backend's own ``BackendNotInstalledError`` surface when its engine
    package is missing.
    """
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r} — available: {', '.join(available_backends())}")
    module_path, class_name = BACKENDS[name].split(":")
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"backend {name!r} is misregistered: {class_name} not found in {module_path}"
        ) from exc
    # Validate options against the constructor signature before calling
    # it: a TypeError raised inside __init__ is a backend bug and must
    # propagate untouched, while a bad option is a caller error that
    # deserves the backend name and the valid options in the message.
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not accepts_var:
        valid = {p for p in params if p != "self"}
        unknown = sorted(set(kwargs) - valid)
        if unknown:
            raise RuntimeError(
                f"backend {name!r} ({class_name}) does not accept option(s): "
                f"{', '.join(repr(k) for k in unknown)} — valid options: "
                f"{', '.join(sorted(valid)) or 'none'}"
            )
    return cls(**kwargs)  # type: ignore[return-value]
