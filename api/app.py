from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

app = FastAPI(
    title="Gesture Control API",
    description="API para control del sistema mediante gestos",
    version="1.0.0"
)

# CORS para permitir conexiones desde diferentes orígenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Gesture Control API está funcionando! 🚀"}