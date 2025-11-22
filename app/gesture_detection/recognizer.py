"""
Wrapper for MediaPipe Gesture Recognizer (canned gestures).
If MediaPipe Tasks API is not available, this wrapper exposes a stub that raises
an informative ImportError when used.
"""
import cv2

try:
    # Import mediapipe tasks (may not be present in older mediapipe installs)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    _HAS_MP_TASKS = True
except Exception:
    _HAS_MP_TASKS = False


class MPGestureRecognizer:
    def __init__(self, model_path: str | None = None, gestures_to_include: list | None = None):
        if not _HAS_MP_TASKS:
            raise ImportError("MediaPipe Tasks API not available. Install a mediapipe version with mediapipe.tasks (see project README).")

        base_options = vision.BaseOptions(model_asset_path=model_path) if model_path else vision.BaseOptions()

        canned_opts = None
        if gestures_to_include:
            try:
                canned_opts = vision.CannedGestureClassifierOptions(include_gesture_names=list(gestures_to_include))
            except Exception:
                # Some versions may not expose this option; ignore and use defaults
                canned_opts = None

        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=canned_opts
        )

        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def recognize_for_video(self, frame_bgr, timestamp_ms: int):
        """Run recognition on a BGR frame and return (gesture_name, score) or (None, None).

        Returns the top-scoring canned gesture name and its score.
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor_image = vision.TensorImage.create_from_array(img_rgb)
        result = self.recognizer.recognize_for_video(tensor_image, timestamp_ms)

        # result.gestures is a list[List[Category]] (one list per hand)
        try:
            if result.gestures and len(result.gestures) > 0 and len(result.gestures[0]) > 0:
                top = result.gestures[0][0]
                # attribute names may vary across versions
                name = getattr(top, 'category_name', None) or getattr(top, 'label', None)
                score = getattr(top, 'score', None) or getattr(top, 'confidence', None)
                return name, score
        except Exception:
            pass
        return None, None


# Fallback stub class for environments without mediapipe.tasks
class MPGestureRecognizerStub:
    def __init__(self, *args, **kwargs):
        raise ImportError("MediaPipe Tasks API not available. Install mediapipe with the Tasks API or set USE_MEDIAPIPE_RECOGNIZER=False in config.")


# choose exported class depending on availability
MPGestureRecognizerOrStub = MPGestureRecognizer if _HAS_MP_TASKS else MPGestureRecognizerStub
