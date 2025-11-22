"""
Servicio que conecta detección de gestos con API
"""
import cv2
import requests
import threading
import time
import sys
import os
from datetime import datetime

# Añadir el directorio app al path para imports (igual que en demo.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

try:
    from gesture_detection.detector import GestureDetector
    from gesture_detection.utils import draw_gesture_info, draw_debug_info
    from gesture_detection import config
    # recognizer wrapper (optional)
    try:
        from gesture_detection.recognizer import MPGestureRecognizerOrStub
    except Exception:
        MPGestureRecognizerOrStub = None
except ImportError as e:
    print(f"❌ Error importando módulos de detección: {e}")
    print("Verificando estructura de carpetas...")
    print(f"Directorio actual: {os.getcwd()}")
    print(f"Contenido de app/: {os.listdir('app') if os.path.exists('app') else 'No existe'}")
    print(f"Contenido de app/gesture_detection/: {os.listdir('app/gesture_detection') if os.path.exists('app/gesture_detection') else 'No existe'}")
    sys.exit(1)

class GestureService:
    def __init__(self, api_url="http://localhost:8000/api/v1"):
        self.api_url = api_url
        try:
            self.detector = GestureDetector()
        except Exception as e:
            print(f"❌ Error inicializando detector: {e}")
            raise
        
        # Intentar inicializar MediaPipe GestureRecognizer si está habilitado en config
        self.mp_recognizer = None
        self.use_mediapipe_recognizer = False
        try:
            if getattr(config, 'USE_MEDIAPIPE_RECOGNIZER', False) and MPGestureRecognizerOrStub:
                try:
                    self.mp_recognizer = MPGestureRecognizerOrStub(gestures_to_include=getattr(config, 'MEDIAPIPE_CANNED_GESTURES', None))
                    self.use_mediapipe_recognizer = True
                    print("ℹ️ MediaPipe GestureRecognizer inicializado correctamente (usando canned gestures).")
                except Exception as e:
                    print(f"⚠️ No se pudo inicializar MediaPipe recognizer: {e}. Se usará el detector heurístico.")
                    self.mp_recognizer = None
                    self.use_mediapipe_recognizer = False
        except Exception:
            self.mp_recognizer = None
            self.use_mediapipe_recognizer = False
        self.last_gesture_time = 0
        self.min_gesture_interval = 2.0  # Segundos entre gestos
        
    def send_gesture_to_api(self, gesture: str, confidence: float | None = None):
        """Envía gesto detectado a la API"""
        try:
            # Agregar información adicional sobre el estado de la mano si está disponible
            hand_state = None
            finger_count = None
            try:
                finger_count = getattr(self.detector, 'last_finger_count', None)
                if finger_count is not None:
                    if finger_count >= 4:
                        hand_state = 'OPEN'
                    elif finger_count <= 1:
                        hand_state = 'CLOSED'
            except Exception:
                # no bloquear el envío si no está disponible
                finger_count = None
                hand_state = None

            payload = {
                "gesture": gesture,
                "confidence": float(confidence) if confidence is not None else 1.0,
                "timestamp": datetime.now().isoformat(),
                "device_id": "webcam_detector",
                "hand_state": hand_state,
                "finger_count": finger_count
            }
            
            response = requests.post(
                f"{self.api_url}/gesture",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {gesture} -> {result['message']}")
            else:
                print(f"❌ Error API: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error conexión API: {e}")
        except Exception as e:
            print(f"❌ Error inesperado enviando gesto: {e}")
    
    def run_detection(self):
        """Ejecuta detección de gestos y envía a API"""
        print("🚀 Servicio de detección de gestos iniciado")
        print(f"📡 API: {self.api_url}")
        print("📹 Presiona 'q' para salir\n")
        
        try:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            
            if not cap.isOpened():
                print("❌ Error: No se puede abrir la webcam")
                return
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("❌ No se puede leer frame de la webcam")
                    break
                
                frame = cv2.flip(frame, 1)
                # Siempre procesamos el frame con el detector heurístico para obtener landmarks y finger_count
                gesture, hand_detected, results = self.detector.process_frame(frame)

                # Si tenemos recognizer de MediaPipe, usar su salida preferentemente
                mp_gesture = None
                mp_score = None
                if self.use_mediapipe_recognizer and self.mp_recognizer:
                    try:
                        ts = int(time.time() * 1000)
                        name, score = self.mp_recognizer.recognize_for_video(frame, ts)
                        if name:
                            mp_gesture = name
                            mp_score = score
                    except Exception as e:
                        # si falla el recognizer en runtime, desactivar para no romper el bucle
                        print(f"⚠️ Error recognizer runtime: {e}. Desactivando recognizer.")
                        self.use_mediapipe_recognizer = False
                        self.mp_recognizer = None

                # Depuración: loguear gesto detectado en consola para verificar
                if mp_gesture:
                    print(f"[MP-RECOG] gesto detectado: {mp_gesture}, score={mp_score}")
                    gesture_to_use = mp_gesture
                else:
                    if gesture:
                        print(f"[DETECTOR] gesto detectado: {gesture}, fingers={getattr(self.detector, 'last_finger_count', None)}")
                    gesture_to_use = gesture

                # Dibujar landmarks y info
                frame = self.detector.draw_landmarks(frame, results)
                frame = draw_gesture_info(frame, gesture_to_use, hand_detected)

                # Depuración en pantalla: finger_count, pointing y score del recognizer
                try:
                    finger_count = getattr(self.detector, 'last_finger_count', None)
                    pointing = False
                    if getattr(self.detector, 'last_landmarks', None) is not None:
                        try:
                            pointing = self.detector.is_pointing(self.detector.last_landmarks)
                        except Exception:
                            pointing = False
                    frame = draw_debug_info(frame, finger_count=finger_count, pointing=pointing, mp_score=mp_score)
                except Exception:
                    pass
                
                # Enviar gesto a API (con throttling) — usar la salida preferida (recognizer o detector)
                if gesture_to_use:
                    current_time = time.time()
                    if current_time - self.last_gesture_time > self.min_gesture_interval:
                        threading.Thread(
                            target=self.send_gesture_to_api,
                            args=(gesture_to_use, mp_score),
                            daemon=True
                        ).start()
                        self.last_gesture_time = current_time
                
                cv2.imshow('Gesture Control Service', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        except Exception as e:
            print(f"❌ Error durante detección: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                cap.release()
                cv2.destroyAllWindows()
                self.detector.close()
            except:
                pass
            print("\n✅ Servicio finalizado")

if __name__ == "__main__":
    try:
        service = GestureService()
        service.run_detection()
    except KeyboardInterrupt:
        print("\n🛑 Servicio interrumpido por usuario")
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()