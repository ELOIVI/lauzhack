"""
Detector de gestos v2 - Usando secuencias temporales
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque
from . import config

SEQUENCE_LENGTH = 15  # Debe coincidir con el entrenamiento

class GestureDetector:
    """
    Detector de gestos usando secuencias temporales
    """
    def __init__(self, model_path=None, scaler_path=None):
        """
        Inicializa el detector
        
        Args:
            model_path: Ruta al modelo .pkl
            scaler_path: Ruta al scaler .pkl
        """
        # Rutas por defecto
        base_path = os.path.join(os.path.dirname(__file__), '..', '..')
        
        if model_path is None:
            model_path = os.path.join(base_path, 'gesture_model_v2.pkl')
        if scaler_path is None:
            scaler_path = os.path.join(base_path, 'feature_scaler.pkl')
        
        # Cargar modelo
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo no encontrado en {model_path}\n"
                "Entrena el modelo con train_sequence_model.ipynb"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Cargar scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler no encontrado en {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✅ Modelo cargado: {model_path}")
        print(f"✅ Scaler cargado: {scaler_path}")
        print(f"📋 Gestos: {list(self.model.classes_)}")
        
        # Buffer para secuencia en vivo
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.confidence_threshold = 0.75
        
        # Historial para suavizar
        self.gesture_history = deque(maxlen=3)
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
    def extract_motion_features(self, sequence):
        """Extrae features de movimiento (misma lógica que en entrenamiento)"""
        if len(sequence) < 2:
            return None
        
        features = []
        
        # Normalizar por primera posición
        first_wrist = sequence[0][0]
        
        for frame_landmarks in sequence:
            normalized = frame_landmarks - first_wrist
            features.extend(normalized.flatten())
        
        # Velocidades
        velocities = []
        for i in range(1, len(sequence)):
            velocity = sequence[i] - sequence[i-1]
            velocities.extend(velocity.flatten())
        features.extend(velocities)
        
        # Aceleraciones
        accelerations = []
        for i in range(2, len(sequence)):
            accel = (sequence[i] - sequence[i-1]) - (sequence[i-1] - sequence[i-2])
            accelerations.extend(accel.flatten())
        features.extend(accelerations)
        
        # Features globales
        wrist_trajectory = np.array([frame[0] for frame in sequence])
        total_displacement = wrist_trajectory[-1] - wrist_trajectory[0]
        features.extend(total_displacement.flatten())
        
        x_disp = total_displacement[0]
        y_disp = total_displacement[1]
        features.extend([x_disp, y_disp, np.sqrt(x_disp**2 + y_disp**2)])
        
        mean_velocity = np.mean([np.linalg.norm(wrist_trajectory[i] - wrist_trajectory[i-1]) 
                                for i in range(1, len(wrist_trajectory))])
        features.append(mean_velocity)
        
        return np.array(features)
    
    def process_frame(self, frame):
        """
        Procesa frame y actualiza buffer de secuencia
        
        Returns:
            tuple: (gesture, confidence, hand_detected, results)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = None
        confidence = 0.0
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            
            # Añadir frame al buffer
            landmarks = results.multi_hand_landmarks[0].landmark
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            self.sequence_buffer.append(landmark_array)
            
            # Si tenemos secuencia completa, predecir
            if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                features = self.extract_motion_features(list(self.sequence_buffer))
                
                if features is not None:
                    # Normalizar
                    features_scaled = self.scaler.transform([features])
                    
                    # Predecir
                    prediction = self.model.predict(features_scaled)[0]
                    probabilities = self.model.predict_proba(features_scaled)[0]
                    pred_confidence = np.max(probabilities)
                    
                    # Solo aceptar si confianza > umbral
                    if pred_confidence >= self.confidence_threshold:
                        # Suavizar con historial
                        self.gesture_history.append(prediction)
                        
                        if len(self.gesture_history) >= 2:
                            most_common = max(set(self.gesture_history),
                                            key=self.gesture_history.count)
                            if self.gesture_history.count(most_common) >= 2:
                                gesture = most_common
                                confidence = pred_confidence
        else:
            # Sin mano, resetear buffer
            self.sequence_buffer.clear()
            self.gesture_history.clear()
        
        return gesture, confidence, hand_detected, results
    
    def get_buffer_progress(self):
        """Retorna el progreso del buffer (para UI)"""
        return len(self.sequence_buffer), SEQUENCE_LENGTH
    
    def reset(self):
        """Resetea buffers"""
        self.sequence_buffer.clear()
        self.gesture_history.clear()
    
    def close(self):
        """Cierra MediaPipe"""
        self.hands.close()