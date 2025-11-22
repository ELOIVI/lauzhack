import { useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
  const [lastEvent, setLastEvent] = useState(null)
  const wsRef = useRef(null)
  const boxRef = useRef(null)

  useEffect(() => {
    // Conectar al WebSocket del servidor (asegúrate de que esté en ejecución)
    const ws = new WebSocket('ws://localhost:8000/ws')
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WS connected')
      // algunos servidores esperan un primer mensaje para mantener la conexión
      try { ws.send('hello') } catch (e) {}
    }

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        setLastEvent(data)
        // mover caja según gesto
        if (boxRef.current && data.gesture) {
          const box = boxRef.current
          const style = box.style
          switch (data.gesture) {
            case 'LEFT':
              style.transform = 'translateX(-120px)'
              break
            case 'RIGHT':
              style.transform = 'translateX(120px)'
              break
            case 'UP':
              style.transform = 'translateY(-120px)'
              break
            case 'DOWN':
              style.transform = 'translateY(120px)'
              break
            default:
              style.transform = 'translateX(0) translateY(0)'
          }
          // reset suave
          setTimeout(() => {
            if (box) box.style.transform = 'translateX(0) translateY(0)'
          }, 700)
        }
      } catch (err) {
        console.error('Invalid WS message', err)
      }
    }

    ws.onclose = () => console.log('WS closed')
    ws.onerror = (e) => console.error('WS error', e)

    return () => {
      try { ws.close() } catch (e) {}
    }
  }, [])

  return (
    <div className="App" style={{ padding: 20 }}>
      <h1>Realtime gestures (demo)</h1>
      <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
        <div style={{ width: 300, height: 220, border: '1px solid #ddd', padding: 16 }}>
          <div style={{ marginBottom: 8 }}>Último evento:</div>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{lastEvent ? JSON.stringify(lastEvent, null, 2) : '—'}</pre>
        </div>

        <div style={{ width: 200, height: 200, position: 'relative' }}>
          <div
            ref={boxRef}
            style={{
              width: 60,
              height: 60,
              background: '#2563eb',
              borderRadius: 8,
              transition: 'transform 0.3s ease',
              transform: 'translateX(0) translateY(0)'
            }}
          />
        </div>
      </div>
      <p style={{ marginTop: 18, color: '#666' }}>
        Conecta el detector y ejecuta el servidor en <code>localhost:8000</code>.
      </p>
    </div>
  )
}

export default App
