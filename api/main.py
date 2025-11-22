"""
Script principal para ejecutar el servicio completo
"""
import subprocess
import sys
import time
import threading
import os

# Añadir el directorio app al path para imports (igual que en demo.py y gesture_service.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
# Asegurar que la raíz del proyecto esté en sys.path para poder importar modules en la raíz
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_api_server():
    """Ejecuta el servidor FastAPI"""
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api.app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Servidor API detenido")
    except Exception as e:
        print(f"❌ Error ejecutando servidor: {e}")

def wait_for_api():
    """Espera a que la API esté disponible"""
    try:
        import requests
    except ImportError:
        print("❌ Error: requests no está instalado. Instala con: pip install requests")
        return False
    
    max_attempts = 30
    for i in range(max_attempts):
        try:
            # Usar endpoint disponible en la API para comprobar estado
            response = requests.get("http://localhost:8000/api/v1/status", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        
        time.sleep(1)
        print(f"⏳ Esperando API... ({i+1}/{max_attempts})")
    return False

def main():
    print("🎯 Iniciando Gesture Control System")
    print("=" * 50)
    
    # Verificar imports necesarios
    try:
        from gesture_service import GestureService
    except ImportError as e:
        print(f"❌ Error: No se puede importar GestureService: {e}")
        print("Detalles del error:")
        import traceback
        traceback.print_exc()
        print(f"\nDirectorio actual: {os.getcwd()}")
        print(f"Archivos en directorio actual: {os.listdir('.')}")
        print(f"Python path: {sys.path}")
        sys.exit(1)
    
    # Verificar uvicorn
    try:
        import uvicorn
    except ImportError:
        print("❌ Error: uvicorn no está instalado. Instala con: pip install uvicorn fastapi")
        sys.exit(1)
    
    # Iniciar API en hilo separado
    print("🚀 Iniciando servidor API...")
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Esperar que la API se inicie
    if wait_for_api():
        print("✅ API iniciada correctamente!")
        print("📱 Puedes probar la API en: http://localhost:8000/docs")
        print("=" * 50)
        
        # Iniciar detección de gestos
        try:
            service = GestureService()
            service.run_detection()
        except KeyboardInterrupt:
            print("\n🛑 Servicio de gestos detenido")
        except Exception as e:
            print(f"❌ Error ejecutando servicio: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Error: No se pudo iniciar la API")

if __name__ == "__main__":
    main()