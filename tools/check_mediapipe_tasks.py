"""
Small script to check if mediapipe Tasks API is available in the current environment.
Run this inside your virtualenv before enabling USE_MEDIAPIPE_RECOGNIZER.
"""
import sys

try:
    import mediapipe as mp
    print(f"mediapipe version: {mp.__version__}")
except Exception as e:
    print(f"mediapipe import failed: {e}")

try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks import python
    print("mediapipe.tasks available")
    # try to access GestureRecognizer class
    try:
        from mediapipe.tasks.python.vision import GestureRecognizer
        print("GestureRecognizer class present")
    except Exception as e:
        print(f"GestureRecognizer not found in mediapipe.tasks.python.vision: {e}")
except Exception as e:
    print(f"mediapipe.tasks import failed: {e}")

print('\nPython executable:', sys.executable)
