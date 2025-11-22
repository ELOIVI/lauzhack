# Gesture Detection Module

## Instalación
```bash
cd app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rápido
```python
from gesture_detection.detector import GestureDetector

detector = GestureDetector()
gesture = detector.get_current_gesture()
print(gesture)  # "LEFT", "RIGHT", "UP", "DOWN", None
```

## Desarrollo

- **Eloi**: Trabajar en `gesture_detection/`
- **Compañera**: Trabajar en `integration/`

## Gestos soportados

- `LEFT` - Mano hacia la izquierda
- `RIGHT` - Mano hacia la derecha  
- `UP` - Mano hacia arriba
- `DOWN` - Mano hacia abajo
- `OPEN` - Mano abierta (5 dedos)
- `CLOSED` - Puño cerrado

## Testing
```bash
python tests/demo.py
```