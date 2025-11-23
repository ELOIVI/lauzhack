"""
Aplicación principal - Con cooldown visual
"""

import cv2
import sys
import os
import pickle
import time
from collections import deque
import warnings

import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from gesture_detection.detector import GestureDetector, SEQUENCE_LENGTH
import mediapipe as mp

GESTURE_DEFINITIONS = [
    ('1', 'SWIPE_LEFT', 'Swipe izquierda'),
    ('2', 'SWIPE_RIGHT', 'Swipe derecha'),
    ('3', 'SWIPE_UP', 'Swipe arriba'),
    ('4', 'SWIPE_DOWN', 'Swipe abajo'),
    ('5', 'PINCH_OPEN', 'Pinch abrir'),
    ('6', 'PINCH_CLOSE', 'Pinch cerrar'),
    ('7', 'FIST_CLOSE', 'Puño cerrado'),
    ('8', 'OPEN_STATIC', 'Mano abierta'),
]

GESTURE_SHORTCUTS = {ord(key): gesture for key, gesture, _ in GESTURE_DEFINITIONS}
GESTURE_DESCRIPTIONS = {gesture: desc for _, gesture, desc in GESTURE_DEFINITIONS}


def append_sequence_to_dataset(data_dir, label, sequence_frames):
    """Guardar una secuencia cruda de landmarks en el dataset"""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{label}.pkl")

    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            sequences = pickle.load(fh)
    else:
        sequences = []

    stacked = np.stack([np.array(frame, dtype=np.float32) for frame in sequence_frames])
    sequences.append(stacked)

    with open(file_path, 'wb') as fh:
        pickle.dump(sequences, fh)

    return len(sequences)


def main():
    print("="*70)
    print("  GESTURE DETECTION AI - LauzHack 2025")
    print("="*70)
    print("\n🚀 Iniciando detector...\n")
    
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(app_dir, 'models', 'gesture_model_v3.pkl')
        scaler_path = os.path.join(app_dir, 'models', 'feature_scaler_v3.pkl')
        
        detector = GestureDetector(
            model_path=model_path,
            scaler_path=scaler_path
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    if not cap.isOpened():
        print("❌ No se puede abrir la webcam")
        return
    
    gesture_count = {}
    last_gesture = None

    data_dir = os.path.join(app_dir, 'gesture_data')
    record_label = None
    recording = False
    record_buffer = deque(maxlen=SEQUENCE_LENGTH)
    record_session_counts = {}
    record_progress = 0
    feedback_message = ""
    feedback_until = 0.0
    
    print("\n[OK] Sistema listo")
    print("Cooldown: 1.5s entre gestos")
    print("Confianza: 55%")
    print("Q = Salir | R = Reset")
    print("Captura: 1-8 gesto | G grabar | C cancelar | N reset contador")
    print("Reentreno: python train_model.py\n")
    print("="*70)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gesture, confidence, hand_detected, results = detector.process_frame(frame)
        
        # Contar gestos
        if gesture and gesture != last_gesture:
            gesture_count[gesture] = gesture_count.get(gesture, 0) + 1
            last_gesture = gesture
            print(f"✅ {gesture:15} ({confidence*100:.1f}%) - Total: {gesture_count[gesture]}")
        
        # Landmarks y datos crudos para captura
        landmark_array = None
        if results.multi_hand_landmarks:
            first_hand = results.multi_hand_landmarks[0]
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in first_hand.landmark], dtype=np.float32)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )

        if recording:
            if landmark_array is not None:
                record_buffer.append(landmark_array.copy())
            record_progress = len(record_buffer)

            if len(record_buffer) == SEQUENCE_LENGTH:
                sequence_frames = list(record_buffer)
                dataset_total = append_sequence_to_dataset(data_dir, record_label, sequence_frames)
                record_session_counts[record_label] = record_session_counts.get(record_label, 0) + 1
                print(f"💾 {record_label}: muestra añadida (sesión {record_session_counts[record_label]}, dataset {dataset_total})")
                feedback_message = f"Guardado {record_label}: sesión {record_session_counts[record_label]} · dataset {dataset_total}"
                feedback_until = time.time() + 3.0

                recording = False
                record_buffer.clear()
                record_progress = 0
                detector.reset()
        else:
            record_progress = 0
            if record_buffer:
                record_buffer.clear()
        
        # UI
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 200), (0, 0, 0), -1)
        
        # Buffer
        progress, total = detector.get_buffer_progress()
        bar_width = int((progress / total) * 620)
        cv2.rectangle(frame, (10, 10), (630, 30), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 10), (10 + bar_width, 30), (0, 255, 255), -1)
        cv2.putText(frame, f"Buffer: {progress}/{total}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Cooldown
        cooldown = detector.get_cooldown_remaining()
        if cooldown > 0:
            cooldown_text = f"Cooldown: {cooldown:.1f}s"
            cv2.putText(frame, cooldown_text, (400, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Estado
        if gesture:
            text = f"GESTO: {gesture}"
            conf_text = f"Confianza: {confidence*100:.1f}%"
            color = (0, 255, 0)
        elif cooldown > 0:
            text = f"Ultimo: {last_gesture}" if last_gesture else "En cooldown"
            conf_text = ""
            color = (0, 165, 255)
        elif hand_detected and progress < total:
            text = "Acumulando frames..."
            conf_text = ""
            color = (255, 255, 0)
        elif hand_detected:
            text = "Procesando..."
            conf_text = ""
            color = (255, 255, 0)
        else:
            text = "Muestra tu mano"
            conf_text = ""
            color = (100, 100, 100)
        
        cv2.putText(frame, text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if conf_text:
            cv2.putText(frame, conf_text, (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if record_label:
            desc = GESTURE_DESCRIPTIONS.get(record_label, record_label)
            capture_text = f"[{record_label}] {desc}"
        else:
            capture_text = "Selecciona gesto (1-8)"

        if recording:
            capture_status = f"Capturando {capture_text} · {record_progress}/{SEQUENCE_LENGTH}"
            capture_color = (0, 165, 255)
        elif record_label:
            capture_status = f"Listo para grabar {capture_text} · G para capturar"
            capture_color = (0, 255, 255)
        else:
            capture_status = f"Captura: {capture_text}"
            capture_color = (150, 150, 150)

        cv2.putText(frame, capture_status, (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, capture_color, 2)

        now = time.time()
        if feedback_message and now < feedback_until:
            cv2.putText(frame, feedback_message, (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif feedback_message and now >= feedback_until:
            feedback_message = ""
        
        abbrev = {
            'SWIPE_LEFT': 'L', 'SWIPE_RIGHT': 'R',
            'SWIPE_UP': 'U', 'SWIPE_DOWN': 'D',
            'PINCH_OPEN': 'PO', 'PINCH_CLOSE': 'PC',
            'FIST_CLOSE': 'FC', 'OPEN_STATIC': 'OS'
        }

        # Estadísticas
        if record_session_counts:
            capture_stats = " | ".join([
                f"{abbrev.get(g, g[:2])}:{c}"
                for g, c in sorted(record_session_counts.items())
            ])
            cv2.putText(frame, f"Capturas: {capture_stats}", (10, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if gesture_count:
            stats = " | ".join([
                f"{abbrev.get(g, g[:2])}:{c}" 
                for g, c in sorted(gesture_count.items())
            ])
            
            cv2.putText(frame, stats, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Gesture Detection AI', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            gesture_count = {}
            last_gesture = None
            print("\n🔄 Reset\n")
        elif key in GESTURE_SHORTCUTS:
            record_label = GESTURE_SHORTCUTS[key]
            recording = False
            record_buffer.clear()
            record_progress = 0
            detector.reset()
            desc = GESTURE_DESCRIPTIONS.get(record_label, record_label)
            feedback_message = f"Gesto seleccionado: {record_label} ({desc})"
            feedback_until = time.time() + 2.5
        elif key == ord('g'):
            if record_label:
                recording = True
                record_buffer.clear()
                record_progress = 0
                detector.reset()
                feedback_message = f"Grabando {record_label}..."
            else:
                feedback_message = "Selecciona un gesto con 1-8"
            feedback_until = time.time() + 2.0
        elif key == ord('c'):
            if recording or record_buffer:
                recording = False
                record_buffer.clear()
                record_progress = 0
                detector.reset()
                feedback_message = "Captura cancelada"
                feedback_until = time.time() + 1.5
        elif key == ord('n'):
            if record_session_counts:
                record_session_counts.clear()
                feedback_message = "Contadores de captura reiniciados"
                feedback_until = time.time() + 1.5
        elif key == ord('t'):
            feedback_message = "Reentrena: python train_model.py"
            feedback_until = time.time() + 2.5
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    if gesture_count:
        print("\n" + "="*70)
        print("ESTADÍSTICAS")
        print("="*70)
        for gesture, count in sorted(gesture_count.items()):
            print(f"  {gesture:15} : {count}")
        print("="*70)
    
    print("\n✅ Cerrado\n")

if __name__ == "__main__":
    main()