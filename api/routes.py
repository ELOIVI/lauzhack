"""
Endpoints de la API
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from .models import GestureEvent, ActionResponse, SystemStatus
from .actions import SystemController

router = APIRouter()
system_controller = SystemController()
last_gesture = None

@router.post("/gesture", response_model=ActionResponse)
async def process_gesture(gesture_event: GestureEvent):
    """Procesa un gesto y ejecuta la acción correspondiente"""
    global last_gesture
    
    try:
        success, message = system_controller.execute_action(gesture_event.gesture)
        last_gesture = gesture_event.gesture
        
        return ActionResponse(
            success=success,
            message=message,
            action_executed=gesture_event.gesture
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=SystemStatus)
async def get_status():
    """Estado del sistema y acciones disponibles"""
    return SystemStatus(
        status="active",
        available_actions=system_controller.get_available_actions(),
        last_gesture=last_gesture
    )

@router.get("/gestures")
async def get_available_gestures():
    """Lista de gestos disponibles"""
    from gesture_detection.config import GESTURES
    return {"gestures": GESTURES}