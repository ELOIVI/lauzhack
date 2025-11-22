"""
Demo del detector de gestos
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gesture_detection.detector import GestureDetector
from gesture_detection.utils import draw_gesture_info
from gesture_detection import config


def main():
    print("🚀 Demo de detección de gestos")
    print(f"📹 Gestos: {list(config.GESTURES.keys())}")
    print("📹 Presiona 'q' para salir, 'r' para resetear\n")
    
    detector = GestureDetector()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print("❌ Error: No se puede abrir la webcam")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gesture, hand_detected, results = detector.process_frame(frame)
        
        # Dibujar landmarks
        frame = detector.draw_landmarks(frame, results)
        
        # Dibujar info
        frame = draw_gesture_info(frame, gesture, hand_detected)
        
        # Mostrar gesto en consola
        if gesture:
            print(f"✅ {gesture}")
        
        cv2.imshow('Gesture Detection Demo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("🔄 Reset")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\n✅ Demo finalizado")


if __name__ == "__main__":
    main()