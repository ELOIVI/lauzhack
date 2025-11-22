"""
Modelos de datos para la API
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class GestureEvent(BaseModel):
    gesture: str
    confidence: float = 1.0
    timestamp: datetime
    device_id: Optional[str] = "default"
    # Información opcional sobre el estado de la mano
    hand_state: Optional[str] = None  # e.g. 'OPEN' o 'CLOSED'
    finger_count: Optional[int] = None

class ActionResponse(BaseModel):
    success: bool
    message: str
    action_executed: str

class SystemStatus(BaseModel):
    status: str
    available_actions: list
    last_gesture: Optional[str] = None