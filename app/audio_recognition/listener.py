"""Simple microphone listener that detects the word 'SOS' and posts an event.

Notes:
- Uses the `SpeechRecognition` package and the Google Web Speech API by default.
- On Windows you may need to install PyAudio (or use an alternative audio backend).
- When 'sos' is detected the listener sends a POST to the integration API at
  http://localhost:8000/event with JSON {gesture, confidence, timestamp}.
"""
from __future__ import annotations

import time
import logging
import os
from typing import Optional

import requests
import speech_recognition as sr

API_URL = os.environ.get("GESTURE_API_URL", "http://localhost:8000/event")
LOG = logging.getLogger("audio_listener")
logging.basicConfig(level=logging.INFO)


def send_event(gesture: str, confidence: float = 0.95, timestamp: Optional[float] = None) -> None:
    payload = {"gesture": gesture, "confidence": confidence, "timestamp": timestamp or time.time()}
    try:
        resp = requests.post(API_URL, json=payload, timeout=3)
        if resp.status_code >= 400:
            LOG.warning("Event POST returned status %s: %s", resp.status_code, resp.text)
        else:
            LOG.info("Sent event: %s", payload)
    except Exception:
        LOG.exception("Failed to send event to %s", API_URL)


def listen_forever(language: Optional[str] = None) -> None:
    """Start listening on the default microphone and send an event when 'sos' is heard.

    The default language is English ("en-US"). You can override the language by
    passing the `language` parameter or setting the environment variable
    `SPEECH_LANGUAGE` (for example: SPEECH_LANGUAGE="en-GB").

    This is intentionally simple: it uses the Google Web Speech API via
    SpeechRecognition. It will work when the machine has internet access. For an
    offline solution consider VOSK or an on-device model.
    """
    # Resolve language preference: function param > env var > default en-US
    language = language or os.environ.get("SPEECH_LANGUAGE", "en-US")
    recognizer = sr.Recognizer()

    try:
        mic = sr.Microphone()
    except Exception:
        LOG.exception("No microphone found or audio backend not available")
        return

    # Calibrate for ambient noise
    with mic as source:
        LOG.info("Calibrating microphone for ambient noise (1s)...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    LOG.info("Listening for 'SOS' (press Ctrl+C to stop)...")

    try:
        while True:
            with mic as source:
                # Listen with a short phrase time limit to keep responsiveness
                audio = recognizer.listen(source, phrase_time_limit=5)

            try:
                text = recognizer.recognize_google(audio, language=language)
                LOG.info("Recognized: %s", text)
                # Keywords to detect (case-insensitive). The payload 'gesture'
                # will be set to the uppercase form of the matched word.
                keywords = {
                    "sos": "SOS",
                    "emergency": "EMERGENCY",
                    "hospital": "HOSPITAL",
                    "help": "HELP",
                }
                lower = text.lower()
                for key, gesture_name in keywords.items():
                    if key in lower:
                        LOG.info("Keyword '%s' detected in '%s' -> sending %s", key, text, gesture_name)
                        send_event(gesture_name, confidence=0.98)
                        break
            except sr.UnknownValueError:
                # Nothing understandable detected — silent skip
                continue
            except sr.RequestError:
                LOG.exception("Speech recognition backend error (network or API)")
                # Sleep briefly to avoid tight retry loop if network fails
                time.sleep(2)
    except KeyboardInterrupt:
        LOG.info("Listener stopped by user")


if __name__ == "__main__":
    listen_forever()
