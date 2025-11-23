"""
Aplicación principal - Con cooldown visual
"""

import cv2
import sys
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from gesture_detection.detector import GestureDetector
import mediapipe as mp

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
    
    print("\n[OK] Sistema listo")
    print("Cooldown: 1.5s entre gestos")
    print("Confianza: 55%")
    print("Q = Salir | R = Reset\n")
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
        
        # Landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )
        
        # UI
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 150), (0, 0, 0), -1)
        
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
        
        cv2.putText(frame, text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if conf_text:
            cv2.putText(frame, conf_text, (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Estadísticas
        if gesture_count:
            abbrev = {
                'SWIPE_LEFT': 'L', 'SWIPE_RIGHT': 'R',
                'SWIPE_UP': 'U', 'SWIPE_DOWN': 'D',
                'PINCH_OPEN': 'PO', 'PINCH_CLOSE': 'PC',
                'FIST_CLOSE': 'FC', 'OPEN_STATIC': 'OS'
            }
            
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