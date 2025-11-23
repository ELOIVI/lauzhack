"""
Detector de gestos v3 - Ventana deslizante + cooldown
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque
import warnings
import time

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

SEQUENCE_LENGTH = 15
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCP = [2, 5, 9, 13, 17]
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]
EPSILON = 1e-6

class GestureDetector:
    """Detector con ventana deslizante y cooldown"""
    
    def __init__(self, model_path=None, scaler_path=None):
        base_path = os.path.join(os.path.dirname(__file__), '..', '..')
        
        if model_path is None:
            model_path = os.path.join(base_path, 'models', 'gesture_model_v3.pkl')
        if scaler_path is None:
            scaler_path = os.path.join(base_path, 'models', 'feature_scaler_v3.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✅ Modelo: {os.path.basename(model_path)}")
        print(f"✅ Gestos: {list(self.model.classes_)}")
        
        # Buffer deslizante
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # Cooldown
        self.last_detection_time = 0
        self.cooldown_duration = 1.5  # 1.5 segundos entre detecciones
        
        # Parámetros de detección
        self.confidence_threshold = 0.55  # Bajado aún más para detectar mejor
        self.min_predictions = 2  # Necesita menos predicciones seguidas
        self.prediction_history = deque(maxlen=5)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
    
    def extract_motion_features(self, sequence):
        """Extraer features de movimiento"""
        if len(sequence) < 2:
            return None
        
        features = []
        first_wrist = sequence[0][0]
        
        for frame_landmarks in sequence:
            normalized = frame_landmarks - first_wrist
            features.extend(normalized.flatten())
        
        velocities = []
        for i in range(1, len(sequence)):
            velocity = sequence[i] - sequence[i-1]
            velocities.extend(velocity.flatten())
        features.extend(velocities)
        
        accelerations = []
        for i in range(2, len(sequence)):
            accel = (sequence[i] - sequence[i-1]) - (sequence[i-1] - sequence[i-2])
            accelerations.extend(accel.flatten())
        features.extend(accelerations)
        
        wrist_trajectory = np.array([frame[0] for frame in sequence])
        total_displacement = wrist_trajectory[-1] - wrist_trajectory[0]
        features.extend(total_displacement.flatten())
        
        x_disp = total_displacement[0]
        y_disp = total_displacement[1]
        features.extend([x_disp, y_disp, np.sqrt(x_disp**2 + y_disp**2)])
        
        mean_velocity = np.mean([
            np.linalg.norm(wrist_trajectory[i] - wrist_trajectory[i-1])
            for i in range(1, len(wrist_trajectory))
        ])
        features.append(mean_velocity)

        last_frame = sequence[-1] - first_wrist
        wrist = last_frame[0]
        palm_points = last_frame[PALM_LANDMARKS]
        palm_center = np.mean(palm_points, axis=0)
        palm_vec = palm_center - wrist
        palm_norm = np.linalg.norm(palm_vec[:2]) + EPSILON

        posture_features = []

        for tip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_MCP):
            tip_vec = last_frame[tip_idx] - wrist
            tip_dist = np.linalg.norm(tip_vec)
            finger_length = np.linalg.norm(last_frame[tip_idx] - last_frame[mcp_idx])
            extension_ratio = tip_dist / palm_norm
            angle = np.arctan2(tip_vec[1], tip_vec[0])

            posture_features.extend([
                tip_dist,
                finger_length,
                extension_ratio,
                np.sin(angle),
                np.cos(angle)
            ])

        for first_idx, second_idx in zip(FINGER_TIPS[:-1], FINGER_TIPS[1:]):
            spread = np.linalg.norm(last_frame[first_idx] - last_frame[second_idx])
            posture_features.append(spread)

        palm_spread = np.mean(np.linalg.norm(palm_points - palm_center, axis=1))
        palm_depth = np.mean(np.abs(palm_points[:, 2]))

        posture_features.extend([palm_norm, palm_spread, palm_depth])

        features.extend(posture_features)

        return np.array(features, dtype=np.float32)
    
    def is_in_cooldown(self):
        """Verificar si está en cooldown"""
        return (time.time() - self.last_detection_time) < self.cooldown_duration
    
    def process_frame(self, frame):
        """Procesar frame con ventana deslizante"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = None
        confidence = 0.0
        hand_detected = False
        
        # Si está en cooldown, no procesar
        if self.is_in_cooldown():
            return None, 0.0, results.multi_hand_landmarks is not None, results
        
        if results.multi_hand_landmarks:
            hand_detected = True
            
            landmarks = results.multi_hand_landmarks[0].landmark
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            self.sequence_buffer.append(landmark_array)
            
            # Predecir cuando el buffer está lleno (ventana deslizante)
            if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                features = self.extract_motion_features(list(self.sequence_buffer))
                
                if features is not None:
                    features_scaled = self.scaler.transform([features])
                    prediction = self.model.predict(features_scaled)[0]
                    probabilities = self.model.predict_proba(features_scaled)[0]
                    pred_confidence = np.max(probabilities)
                    
                    # Solo considerar predicciones con alta confianza
                    if pred_confidence >= self.confidence_threshold:
                        self.prediction_history.append(prediction)
                    else:
                        self.prediction_history.append(None)
                    
                    # Verificar consenso (filtrando None)
                    valid_predictions = [p for p in self.prediction_history if p is not None]
                    
                    if len(valid_predictions) >= self.min_predictions:
                        most_common = max(set(valid_predictions),
                                        key=valid_predictions.count)
                        
                        # Si hay consenso
                        if valid_predictions.count(most_common) >= self.min_predictions:
                            gesture = most_common
                            confidence = pred_confidence
                            
                            # Activar cooldown
                            self.last_detection_time = time.time()
                            
                            # Limpiar historia
                            self.prediction_history.clear()
                            self.sequence_buffer.clear()
        else:
            # Sin mano: limpiar buffer
            self.sequence_buffer.clear()
            self.prediction_history.clear()
        
        return gesture, confidence, hand_detected, results
    
    def get_buffer_progress(self):
        return len(self.sequence_buffer), SEQUENCE_LENGTH
    
    def get_cooldown_remaining(self):
        """Obtener tiempo restante de cooldown"""
        if self.is_in_cooldown():
            return self.cooldown_duration - (time.time() - self.last_detection_time)
        return 0
    
    def reset(self):
        self.sequence_buffer.clear()
        self.prediction_history.clear()
        self.last_detection_time = 0
    
    def close(self):
        self.hands.close()