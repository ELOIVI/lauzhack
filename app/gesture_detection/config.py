"""
Configuración del detector de gestos
"""

# MediaPipe
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Detector
SMOOTHING_FRAMES = 5
MOVEMENT_THRESHOLD = 0.03

# Gestos disponibles
GESTURES = {
    "LEFT": "Mano hacia la izquierda",
    "RIGHT": "Mano hacia la derecha",
    "UP": "Mano hacia arriba",
    "DOWN": "Mano hacia abajo",
    "OPEN": "Mano abierta (5 dedos)",
    "CLOSED": "Puño cerrado"
}

# Cámara
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480