"""Audio detection package.

This module contains a simple speech listener that detects the spoken word
"SOS" and forwards an event to the integration API (`/event`).
"""

__all__ = ["listener"]
