"""
Detector de gestos usando MediaPipe Hands
"""

import cv2
import mediapipe as mp
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
        
        if self.prev_wrist_x is None:
            self.prev_wrist_x = current_x
            self.prev_wrist_y = current_y
            return None
        
        delta_x = current_x - self.prev_wrist_x
        delta_y = current_y - self.prev_wrist_y
        
        self.prev_wrist_x = current_x
        self.prev_wrist_y = current_y
        
        gesture = None
        
        if abs(delta_x) > self.movement_threshold or abs(delta_y) > self.movement_threshold:
            if abs(delta_x) > abs(delta_y):
                gesture = "RIGHT" if delta_x > 0 else "LEFT"
            else:
                gesture = "DOWN" if delta_y > 0 else "UP"
        else:
            fingers = self.count_fingers(landmarks)
            if fingers >= 4:
                gesture = "OPEN"
            elif fingers <= 1:
                gesture = "CLOSED"
        
        if gesture:
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
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list = hand_landmarks.landmark
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
    
    def close(self):
        """Cierra MediaPipe"""
        self.hands.close()