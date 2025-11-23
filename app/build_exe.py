"""
Script para crear ejecutable con PyInstaller
"""

import PyInstaller.__main__
import os
import sys

print("🔨 Construyendo ejecutable...")
print("📦 Esto puede tardar varios minutos...")

PyInstaller.__main__.run([
    'main_app.py',
    '--name=GestureDetectionAI',
    '--onefile',
    '--noconsole',  # Sin ventana de consola
    '--add-data=gesture_model_v2.pkl;.',
    '--add-data=feature_scaler.pkl;.',
    '--hidden-import=mediapipe',
    '--hidden-import=sklearn',
    '--hidden-import=sklearn.ensemble',
    '--hidden-import=sklearn.tree',
    '--hidden-import=cv2',
    '--collect-all=mediapipe',
    '--noconfirm',  # Sobrescribir sin preguntar
])

print("✅ Ejecutable creado en dist/GestureDetectionAI.exe")