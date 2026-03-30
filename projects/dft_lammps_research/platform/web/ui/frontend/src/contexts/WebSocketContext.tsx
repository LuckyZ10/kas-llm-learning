import React, { createContext, useContext, useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'

interface WebSocketContextType {
  isConnected: boolean
  sendMessage: (message: any) => void
  lastMessage: any
}

const WebSocketContext = createContext<WebSocketContextType | null>(null)

export function useWebSocket() {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

interface WebSocketProviderProps {
  children: React.ReactNode
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<any>(null)
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeout = useRef<NodeJS.Timeout>()
  const queryClient = useQueryClient()

  const connect = () => {
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/global`
    
    ws.current = new WebSocket(wsUrl)

    ws.current.onopen = () => {
      setIsConnected(true)
      console.log('WebSocket connected')
    }

    ws.current.onclose = () => {
      setIsConnected(false)
      console.log('WebSocket disconnected')
      // Reconnect after 3 seconds
      reconnectTimeout.current = setTimeout(connect, 3000)
    }

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data)
      setLastMessage(message)
      handleMessage(message)
    }
  }

  const handleMessage = (message: any) => {
    switch (message.type) {
      case 'task_update':
        // Invalidate task queries
        queryClient.invalidateQueries({ queryKey: ['tasks'] })
        if (message.status === 'failed') {
          toast.error(`Task ${message.task_id} failed`)
        } else if (message.status === 'completed') {
          toast.success(`Task ${message.task_id} completed`)
        }
        break
      
      case 'workflow_update':
        queryClient.invalidateQueries({ queryKey: ['workflows'] })
        break
      
      case 'project_update':
        queryClient.invalidateQueries({ queryKey: ['projects'] })
        break
      
      case 'system_stats':
        // Could update a global stats store
        break
      
      case 'log_message':
        if (message.level === 'error') {
          console.error(`[${message.source}]`, message.message)
        }
        break
    }
  }

  const sendMessage = (message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
    }
  }

  useEffect(() => {
    connect()

    // Heartbeat
    const heartbeat = setInterval(() => {
      sendMessage({ type: 'ping', timestamp: Date.now() })
    }, 30000)

    return () => {
      clearInterval(heartbeat)
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current)
      }
      ws.current?.close()
    }
  }, [])

  return (
    <WebSocketContext.Provider value={{ isConnected, sendMessage, lastMessage }}>
      {children}
      {/* Connection status indicator */}
      <div className={`fixed bottom-4 right-4 w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} 
           title={isConnected ? 'Connected' : 'Disconnected'} />
    </WebSocketContext.Provider>
  )
}
