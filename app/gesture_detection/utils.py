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
    
    # Fondo negro (más alto para mostrar info de depuración)
    cv2.rectangle(frame, (0, 0), (640, 140), (0, 0, 0), -1)

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

    cv2.putText(frame, text, (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Información de depuración adicional (opcionales)
    info_y = 70
    extra_lines = []
    # permitir parámetros opcionales pasados como kwargs en versiones futuras
    # El caller puede añadir más texto a mostrar concatenando al frame antes de llamar
    # Para compatibilidad, intentaremos no romper llamadas existentes.

    if hand_detected:
        # mostrar FPS, si existe
        if fps:
            extra_lines.append(f"FPS: {fps}")

    # Dibujar líneas extra (si existen)
    for i, line in enumerate(extra_lines):
        cv2.putText(frame, line, (10, info_y + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return frame


def draw_debug_info(frame, finger_count=None, pointing=False, mp_score=None):
    """Dibuja información de depuración adicional en el frame.

    Usar para mostrar finger_count, si está apuntando, y score del recognizer.
    """
    import cv2

    y = 100
    x = 10
    if finger_count is not None:
        cv2.putText(frame, f"Fingers: {finger_count}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 20
    cv2.putText(frame, f"Pointing: {'YES' if pointing else 'NO'}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255) if pointing else (100,100,100), 1)
    y += 20
    if mp_score is not None:
        cv2.putText(frame, f"MP score: {mp_score:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,250), 1)
    return frame