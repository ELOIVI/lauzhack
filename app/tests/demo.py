"""
Demo del detector v2 con secuencias
"""

import cv2
import sys
import os

# Añadir el directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gesture_detection.detector import GestureDetector
from gesture_detection import config
import mediapipe as mp

def main():
    print("🚀 Detector de gestos V2 (secuencias temporales)")
    print("📹 Presiona 'q' para salir, 'r' para resetear\n")
    
    try:
        # Calcular rutas correctas
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(script_dir)
        
        model_path = os.path.join(app_dir, 'gesture_model_v2.pkl')
        scaler_path = os.path.join(app_dir, 'feature_scaler.pkl')
        
        print(f"🔍 Modelo: {model_path}")
        print(f"🔍 Scaler: {scaler_path}")
        print(f"🔍 ¿Existe modelo? {os.path.exists(model_path)}")
        print(f"🔍 ¿Existe scaler? {os.path.exists(scaler_path)}\n")
        
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
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gesture, confidence, hand_detected, results = detector.process_frame(frame)
        
        # Contar gestos
        if gesture:
            gesture_count[gesture] = gesture_count.get(gesture, 0) + 1
            print(f"✅ {gesture} ({confidence*100:.1f}%) - Total: {gesture_count[gesture]}")
        
        # Dibujar landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )
        
        # UI
        cv2.rectangle(frame, (0, 0), (640, 150), (0, 0, 0), -1)
        
        # Progress bar del buffer
        progress, total = detector.get_buffer_progress()
        bar_width = int((progress / total) * 620)
        cv2.rectangle(frame, (10, 10), (630, 30), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 10), (10 + bar_width, 30), (0, 255, 255), -1)
        cv2.putText(frame, f"Buffer: {progress}/{total}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Gesto detectado
        if gesture:
            text = f"GESTO: {gesture}"
            conf_text = f"Confianza: {confidence*100:.1f}%"
            color = (0, 255, 0)
        elif hand_detected and progress < total:
            text = "Acumulando frames..."
            conf_text = f"Movimiento detectado"
            color = (255, 255, 0)
        elif hand_detected:
            text = "Gesto no reconocido"
            conf_text = "Intenta de nuevo"
            color = (0, 0, 255)
        else:
            text = "Muestra tu mano"
            conf_text = ""
            color = (0, 0, 255)
        
        cv2.putText(frame, text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if conf_text:
            cv2.putText(frame, conf_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Estadísticas
        if gesture_count:
            stats = " | ".join([f"{g.split('_')[1][0]}:{c}" for g, c in sorted(gesture_count.items())])
            cv2.putText(frame, stats, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Gesture Detector V2', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("🔄 Detector reseteado")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    if gesture_count:
        print("\n📊 Estadísticas:")
        for gesture, count in sorted(gesture_count.items()):
            print(f"  {gesture}: {count}")
    print("\n✅ Demo finalizado")

if __name__ == "__main__":
    main()