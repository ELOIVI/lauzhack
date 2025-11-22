"""
Funciones auxiliares para detección de gestos
"""

def get_gesture_description(gesture):
    """
    Retorna la descripción de un gesto
    
    Args:
        gesture (str): Nombre del gesto
        
    Returns:
        str: Descripción del gesto
    """
    from . import config
    return config.GESTURES.get(gesture, "Gesto desconocido")


def draw_gesture_info(frame, gesture, hand_detected, fps=None):
    """
    Dibuja información del gesto en el frame
    
    Args:
        frame: Frame de OpenCV
        gesture: Gesto detectado
        hand_detected: Si hay mano detectada
        fps: FPS opcionales
        
    Returns:
        frame: Frame con información dibujada
    """
    import cv2
    
    # Fondo negro
    cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)
    
    # Texto del gesto
    if gesture:
        text = f"GESTO: {gesture}"
        color = (0, 255, 0)
    elif hand_detected:
        text = "Esperando gesto..."
        color = (255, 255, 0)
    else:
        text = "Sin mano detectada"
        color = (0, 0, 255)
    
    cv2.putText(frame, text, (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # FPS
    if fps:
        cv2.putText(frame, f"FPS: {fps}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame