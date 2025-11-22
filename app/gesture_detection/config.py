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
    "CLOSED": "Puño cerrado",
    "SLIDE_LEFT": "Deslizamiento rápido a la izquierda",
    "SLIDE_RIGHT": "Deslizamiento rápido a la derecha"
}

# Parametros para detectar deslizamientos (slides)
SLIDE_WINDOW = 4  # frames usados para calcular el deslizamiento
SLIDE_THRESHOLD = 0.15  # desplazamiento horizontal normalizado mínimo para considerar slide
SLIDE_VERTICAL_TOLERANCE = 0.08  # tolerancia vertical para que el movimiento sea horizontal

# Parámetros para detectar pointing (apuntar con el índice)
POINTING_ANGLE_THRESHOLD = 30.0  # grados: máxima desviación para considerar el índice recto
POINTING_PERSISTENCE = 3  # frames mínimos que debe mantenerse pointing
POINTING_MIN_EXTENSION = 0.04  # distancia normalizada mínima tip->pip para considerar dedo extendido
# Umbral para considerar mano abierta (número de dedos)
OPEN_FINGER_THRESHOLD = 5
# Permite activar/desactivar la detección de mano abierta
ENABLE_OPEN = False

# MediaPipe GestureRecognizer integration
# Si True, el servicio intentará usar el GestureRecognizer de MediaPipe (canned gestures).
# Requiere una versión de mediapipe con la API 'mediapipe.tasks'. Si no está disponible,
# el sistema volverá al detector heurístico.
USE_MEDIAPIPE_RECOGNIZER = True

# Lista opcional de gestos canned a incluir (None = usar los por defecto del modelo)
MEDIAPIPE_CANNED_GESTURES = None

# Cámara
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480