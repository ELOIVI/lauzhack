"""
Acciones del sistema basadas en gestos
"""
import os
import subprocess
import platform
import shutil
from typing import Dict, Callable

class SystemController:
    def __init__(self):
        self.actions: Dict[str, Callable] = {
            "CLOSED": self.turn_off_screen,
            "OPEN": self.turn_on_screen,
            "LEFT": self.previous_desktop,
            "RIGHT": self.next_desktop,
            "UP": self.increase_volume,
            "DOWN": self.decrease_volume,
        }
    
    def execute_action(self, gesture: str) -> tuple[bool, str]:
        """Ejecuta acción basada en el gesto"""
        if gesture not in self.actions:
            return False, f"Acción no disponible para gesto: {gesture}"
        
        try:
            result = self.actions[gesture]()
            return True, result
        except Exception as e:
            return False, f"Error ejecutando acción: {str(e)}"
    
    def turn_off_screen(self) -> str:
        """Apaga la pantalla"""
        if platform.system() == "Linux":
            return self._run_shell_cmd(
                "xset dpms force off",
                cmd_name="xset",
                pkg_hint="x11-xserver-utils (Debian/Ubuntu)"
            )
        return "Acción no soportada en este sistema"
    
    def turn_on_screen(self) -> str:
        """Enciende la pantalla"""
        if platform.system() == "Linux":
            return self._run_shell_cmd(
                "xset dpms force on",
                cmd_name="xset",
                pkg_hint="x11-xserver-utils (Debian/Ubuntu)"
            )
        return "Acción no soportada en este sistema"
    
    def previous_desktop(self) -> str:
        """Escritorio anterior"""
        if platform.system() == "Linux":
            return self._run_shell_cmd(
                "wmctrl -s $(( $(wmctrl -d | grep '*' | cut -d' ' -f1) - 1 ))",
                cmd_name="wmctrl",
                pkg_hint="wmctrl"
            )
        return "Acción no soportada en este sistema"
    
    def next_desktop(self) -> str:
        """Escritorio siguiente"""
        if platform.system() == "Linux":
            return self._run_shell_cmd(
                "wmctrl -s $(( $(wmctrl -d | grep '*' | cut -d' ' -f1) + 1 ))",
                cmd_name="wmctrl",
                pkg_hint="wmctrl"
            )
        return "Acción no soportada en este sistema"
    
    def increase_volume(self) -> str:
        """Subir volumen"""
        if platform.system() == "Linux":
            return self._run_shell_cmd(
                "pactl set-sink-volume @DEFAULT_SINK@ +10%",
                cmd_name="pactl",
                pkg_hint="pulseaudio-utils o pipewire-pulse según tu pila de audio"
            )
        return "Acción no soportada en este sistema"
    
    def decrease_volume(self) -> str:
        """Bajar volumen"""
        if platform.system() == "Linux":
            return self._run_shell_cmd(
                "pactl set-sink-volume @DEFAULT_SINK@ -10%",
                cmd_name="pactl",
                pkg_hint="pulseaudio-utils o pipewire-pulse según tu pila de audio"
            )
        return "Acción no soportada en este sistema"

    def _run_shell_cmd(self, cmd: str, cmd_name: str, pkg_hint: str | None = None) -> str:
        """Ejecuta un comando de shell comprobando primero si la utilidad existe.

        Devuelve un mensaje amigable si la utilidad no está instalada.
        """
        # comprobar si el ejecutable existe en PATH
        if shutil.which(cmd_name) is None:
            hint = f" Instala: {pkg_hint}." if pkg_hint else ""
            return f"Error: '{cmd_name}' no encontrado en el sistema.{hint}"

        try:
            # usamos shell=True para soportar expresiones compuestas (wmctrl con $(( )) )
            completed = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if completed.returncode == 0:
                # Normalizar mensajes cortos para respuestas API
                # Mapear según comando
                if cmd_name == "xset":
                    return "Pantalla actualizada"
                if cmd_name == "wmctrl":
                    return "Cambio de escritorio realizado"
                if cmd_name == "pactl":
                    return "Volumen ajustado"
                return "Acción ejecutada"
            else:
                return f"Error ejecutando '{cmd_name}': {completed.stderr.strip()}"
        except Exception as e:
            return f"Excepción ejecutando '{cmd_name}': {e}"
    
    def get_available_actions(self) -> list:
        """Lista de acciones disponibles"""
        return list(self.actions.keys())