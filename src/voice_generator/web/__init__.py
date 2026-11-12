"""Web UI package: the local Flask front end (``voice-gen serve``)."""

from voice_generator.web.app import WebConfig, create_app

__all__ = ["WebConfig", "create_app"]
