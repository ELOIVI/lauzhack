from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import uvicorn
from typing import List

app = FastAPI(title="Gesture Events API")


class GestureEvent(BaseModel):
	gesture: str
	confidence: float
	timestamp: float | None = None


class ConnectionManager:
	def __init__(self):
		self.active_websockets: List[WebSocket] = []

	async def connect(self, websocket: WebSocket):
		await websocket.accept()
		self.active_websockets.append(websocket)

	def disconnect(self, websocket: WebSocket):
		try:
			self.active_websockets.remove(websocket)
		except ValueError:
			pass

	async def broadcast(self, message: dict):
		living = []
		for ws in list(self.active_websockets):
			try:
				await ws.send_json(message)
				living.append(ws)
			except Exception:
				# remove dead socket
				try:
					await ws.close()
				except Exception:
					pass
		self.active_websockets = living


manager = ConnectionManager()


@app.get("/health")
async def health():
	return {"status": "ok"}


@app.post("/event")
async def post_event(event: GestureEvent):
	"""Endpoint para recibir eventos desde el detector.

	El detector puede enviar un POST con JSON {gesture, confidence, timestamp}.
	El servidor re-broadcasteará ese JSON a todos los clientes WebSocket conectados.
	"""
	payload = event.dict()
	# Broadcast asynchronously but don't wait too long
	asyncio.create_task(manager.broadcast(payload))
	return JSONResponse({"ok": True, "payload": payload})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
	await manager.connect(websocket)
	try:
		while True:
			# Keep connection open; the server pushes events when they arrive.
			await websocket.receive_text()
	except WebSocketDisconnect:
		manager.disconnect(websocket)


if __name__ == "__main__":
	# Uvicorn entrypoint for local testing
	uvicorn.run("app.integration.api:app", host="0.0.0.0", port=8000, reload=True)