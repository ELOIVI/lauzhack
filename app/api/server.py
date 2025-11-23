"""
FastAPI server para la demo web (OPTIMIZADO)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import cv2
from datetime import datetime
import sys
import os
import base64

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gesture_detection.detector import GestureDetector
import mediapipe as mp

app = FastAPI(title="Gesture Detection Demo API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# IMPORTANTE: Cargar el modelo UNA SOLA VEZ al iniciar
DETECTOR = None
CAP = None

@app.on_event("startup")
async def startup_event():
    global DETECTOR
    print("🚀 Iniciando servidor de demo...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(script_dir)
    
    model_path = os.path.join(app_dir, 'gesture_model_v2.pkl')
    scaler_path = os.path.join(app_dir, 'feature_scaler.pkl')
    
    try:
        DETECTOR = GestureDetector(model_path=model_path, scaler_path=scaler_path)
        print("✅ Detector cargado globalmente")
    except Exception as e:
        print(f"❌ Error cargando detector: {e}")
    
    print("📡 WebSocket: ws://localhost:8001/ws/gestures")


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Gesture Detection Demo",
        "websocket": "ws://localhost:8001/ws/gestures"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.websocket("/ws/gestures")
async def websocket_gestures(websocket: WebSocket):
    await websocket.accept()
    print("✅ Cliente conectado")
    
    if DETECTOR is None:
        await websocket.send_json({"error": "Detector no inicializado"})
        await websocket.close()
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_json({"error": "No se puede abrir la cámara"})
        await websocket.close()
        return
    
    # MediaPipe para dibujar
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error leyendo frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Procesar
            gesture, confidence, hand_detected, results = DETECTOR.process_frame(frame)
            
            # Dibujar landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Info visual
            progress, total = DETECTOR.get_buffer_progress()
            
            # Overlay semitransparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            if gesture:
                cv2.putText(frame, f"{gesture}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, f"{confidence*100:.1f}%", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            elif hand_detected:
                cv2.putText(frame, f"Buffer: {progress}/{total}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Muestra tu mano", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 2)
            
            # OPTIMIZAR: Reducir resolución y calidad
            frame_small = cv2.resize(frame, (480, 360))
            _, buffer = cv2.imencode('.jpg', frame_small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Enviar
            data = {
                "gesture": gesture,
                "hand_detected": hand_detected,
                "confidence": confidence,
                "buffer_progress": progress,
                "buffer_total": total,
                "frame": frame_b64,
                "frame_count": frame_count
            }
            
            await websocket.send_json(data)
            frame_count += 1
            
            # Log cada 100 frames
            if frame_count % 100 == 0:
                print(f"📊 Frames enviados: {frame_count}")
            
            await asyncio.sleep(0.05)  # 20 FPS (más estable)
            
    except WebSocketDisconnect:
        print("❌ Cliente desconectado")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cap.release()
        print("🔒 Cámara liberada")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🎥 SERVIDOR DE DEMO WEB")
    print("=" * 60)
    print("Puerto: 8001")
    print("WebSocket: ws://localhost:8001/ws/gestures")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")