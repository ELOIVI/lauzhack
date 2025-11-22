"""
Detector de gestos usando MediaPipe Hands
"""

import cv2
import mediapipe as mp
import math
from collections import deque
from . import config


class GestureDetector:
    def __init__(self, smoothing_frames=None, movement_threshold=None):
        """
        Inicializa el detector de gestos
        
        Args:
            smoothing_frames: Frames para suavizar
            movement_threshold: Sensibilidad de movimiento
        """
        self.smoothing_frames = smoothing_frames or config.SMOOTHING_FRAMES
        self.movement_threshold = movement_threshold or config.MOVEMENT_THRESHOLD
        
        self.prev_wrist_x = None
        self.prev_wrist_y = None
        self.gesture_history = deque(maxlen=self.smoothing_frames)
        # Último recuento de dedos detectado
        self.last_finger_count = 0
        # Historial corto de posiciones de muñeca para detectar slides (dx rápido)
        self.wrist_history = deque(maxlen=getattr(config, 'SLIDE_WINDOW', 4))
        # Historial de pointing (True/False) para exigir persistencia
        self.pointing_history = deque(maxlen=getattr(config, 'POINTING_PERSISTENCE', 3))

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
    def count_fingers(self, landmarks):
        """Cuenta dedos extendidos"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 6, 10, 14, 18]
        
        fingers_up = 0
        
        # Pulgar
        if landmarks[finger_tips[0]].x < landmarks[finger_bases[0]].x:
            fingers_up += 1
        
        # Otros dedos
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_bases[i]].y:
                fingers_up += 1
                
        return fingers_up

    def is_pointing(self, landmarks) -> bool:
        """Heurística para determinar si se está apuntando con el índice.

        Requisitos:
        - Índice recto (ángulo cercano) según POINTING_ANGLE_THRESHOLD
        - Extensión mínima tip->pip según POINTING_MIN_EXTENSION
        - Otros dedos (middle, ring, pinky) doblados
        """
        try:
            # Índice
            tip = landmarks[8]
            pip = landmarks[6]
            mcp = landmarks[5]

            # Vectores en 2D
            v1x = tip.x - pip.x
            v1y = tip.y - pip.y
            v2x = mcp.x - pip.x
            v2y = mcp.y - pip.y

            # ángulo entre v1 y v2
            dot = v1x * v2x + v1y * v2y
            n1 = math.hypot(v1x, v1y)
            n2 = math.hypot(v2x, v2y)
            if n1 == 0 or n2 == 0:
                return False
            cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
            angle = math.degrees(math.acos(cos_a))

            if angle > getattr(config, 'POINTING_ANGLE_THRESHOLD', 30.0):
                return False

            # extensión mínima tip->pip
            extension = math.hypot(v1x, v1y)
            if extension < getattr(config, 'POINTING_MIN_EXTENSION', 0.04):
                return False

            # comprobar dedos restantes doblados (middle, ring, pinky)
            folded = True
            other_tips = [12, 16, 20]
            other_bases = [10, 14, 18]
            for t, b in zip(other_tips, other_bases):
                if landmarks[t].y < landmarks[b].y:
                    # si tip está por encima de la base => dedo extendido
                    folded = False
                    break

            return folded
        except Exception:
            return False
    
    def detect_gesture(self, landmarks):
        """
        Detecta gesto
        
        Args:
            landmarks: Landmarks de MediaPipe
            
        Returns:
            str: Gesto detectado o None
        """
        if not landmarks:
            return None
        
        wrist = landmarks[0]
        current_x = wrist.x
        current_y = wrist.y
        
        # Inicializar prev si es la primera vez
        if self.prev_wrist_x is None:
            self.prev_wrist_x = current_x
            self.prev_wrist_y = current_y
            # registrar posición inicial en historial de muñeca
            try:
                self.wrist_history.append((current_x, current_y))
            except Exception:
                pass
            # contar dedos y actualizar estado
            fingers_init = self.count_fingers(landmarks)
            self.last_finger_count = fingers_init
            return None

        # calcular deltas
        delta_x = current_x - self.prev_wrist_x
        delta_y = current_y - self.prev_wrist_y

        # actualizar prev
        self.prev_wrist_x = current_x
        self.prev_wrist_y = current_y

        # registrar posición en el historial de muñeca (siempre que haya mano)
        try:
            self.wrist_history.append((current_x, current_y))
        except Exception:
            pass

        # contar dedos y exponerlo
        fingers = self.count_fingers(landmarks)
        self.last_finger_count = fingers
        # comprobar pointing y guardarlo para persistencia
        pointing = self.is_pointing(landmarks)
        try:
            self.pointing_history.append(1 if pointing else 0)
        except Exception:
            pass

        gesture = None

        # Si el usuario está señalando (persistencia) priorizar SLIDE basado en el movimiento horizontal acumulado
        pointing_majority = False
        try:
            if len(self.pointing_history) == self.pointing_history.maxlen:
                pointing_majority = sum(self.pointing_history) >= (self.pointing_history.maxlen // 2 + 1)
        except Exception:
            pointing_majority = False

        if pointing_majority and fingers == 1 and len(self.wrist_history) == self.wrist_history.maxlen:
            x0, y0 = self.wrist_history[0]
            x1, y1 = self.wrist_history[-1]
            total_dx = x1 - x0
            total_dy = y1 - y0
            if abs(total_dx) >= getattr(config, 'SLIDE_THRESHOLD', 0.15) and abs(total_dy) <= getattr(config, 'SLIDE_VERTICAL_TOLERANCE', 0.08):
                gesture = "SLIDE_RIGHT" if total_dx > 0 else "SLIDE_LEFT"
                # devolver el slide inmediatamente
                return gesture

        # Si no es un slide señalando, usar la lógica anterior para movimientos y estado de mano
        if abs(delta_x) > self.movement_threshold or abs(delta_y) > self.movement_threshold:
            if abs(delta_x) > abs(delta_y):
                gesture = "RIGHT" if delta_x > 0 else "LEFT"
            else:
                gesture = "DOWN" if delta_y > 0 else "UP"
        else:
            # Small/no movement: decidir entre OPEN/CLOSED
            # Para evitar falsos positivos de OPEN al reconocer slides/left/right,
            # usamos un umbral más estricto configurable (por defecto 5 dedos).
            if getattr(config, 'ENABLE_OPEN', False) and fingers >= getattr(config, 'OPEN_FINGER_THRESHOLD', 5):
                gesture = "OPEN"
            elif fingers <= 1:
                # mantener CLOSED solo si realmente está en puño completo (<=1) y no señalando movimiento
                gesture = "CLOSED"
        
        if gesture:
            # Si es un slide (movimiento rápido), devolver inmediatamente sin esperar suavizado
            if gesture.startswith("SLIDE_"):
                # añadir al historial para trazabilidad y devolver ya
                try:
                    self.gesture_history.append(gesture)
                except Exception:
                    pass
                return gesture

            self.gesture_history.append(gesture)
            if len(self.gesture_history) >= 3:
                most_common = max(set(self.gesture_history), 
                                 key=self.gesture_history.count)
                if self.gesture_history.count(most_common) >= 2:
                    return most_common
        
        return None
    
    def process_frame(self, frame):
        """
        Procesa frame y detecta gesto
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            tuple: (gesture, hand_detected, results)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture = None
        # almacenar los landmarks procesados recientemente para uso externo (debug/pointing)
        self.last_landmarks = None
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list = hand_landmarks.landmark
                # guardar landmarks para que el servicio pueda consultarlos
                self.last_landmarks = landmarks_list
                gesture = self.detect_gesture(landmarks_list)
        else:
            self.reset()

        return gesture, hand_detected, results
    
    def draw_landmarks(self, frame, results):
        """Dibuja landmarks en el frame"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )
        return frame
    
    def reset(self):
        """Resetea el detector"""
        self.prev_wrist_x = None
        self.prev_wrist_y = None
        self.gesture_history.clear()
        self.last_finger_count = 0
        try:
            self.wrist_history.clear()
        except Exception:
            pass
    
    def close(self):
        """Cierra MediaPipe"""
        self.hands.close()