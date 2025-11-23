"""
Configuración del detector de gestos
"""

# MediaPipe
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Detector
SMOOTHING_FRAMES = 5
MOVEMENT_THRESHOLD = 0.03

GESTURES = {
    # Gestos dinámicos (ML)
    "SWIPE_LEFT": "Mano hacia la izquierda",
    "SWIPE_RIGHT": "Mano hacia la derecha",
    "SWIPE_UP": "Mano hacia arriba",
    "SWIPE_DOWN": "Mano hacia abajo",
    
    # Gestos estáticos (reglas)
    "OPEN": "Mano abierta (5 dedos)",
    "CLOSED": "Puño cerrado"
}

# Parámetros de detección
SEQUENCE_LENGTH = 15
CONFIDENCE_THRESHOLD = 0.75
MIN_FINGERS_OPEN = 4.5
MAX_FINGERS_CLOSED = 1.5

# Cámara
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480