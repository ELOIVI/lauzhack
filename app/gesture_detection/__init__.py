"""
Módulo de detección de gestos con MediaPipe
"""

from .detector import GestureDetector
from . import config

__all__ = ['GestureDetector', 'config']