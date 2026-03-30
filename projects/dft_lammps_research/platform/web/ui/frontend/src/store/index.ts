import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface User {
  id: string
  email: string
  username: string
  full_name?: string
  role: string
  avatar_url?: string
}

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  setUser: (user: User | null) => void
  login: (user: User, token: string) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      setUser: (user) => set({ user, isAuthenticated: !!user }),
      login: (user, token) => {
        localStorage.setItem('access_token', token)
        set({ user, isAuthenticated: true })
      },
      logout: () => {
        localStorage.removeItem('access_token')
        set({ user: null, isAuthenticated: false })
      },
    }),
    {
      name: 'auth-storage',
    }
  )
)

interface UIState {
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  theme: 'light' | 'dark'
  setTheme: (theme: 'light' | 'dark') => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      theme: 'light',
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'ui-storage',
    }
  )
)

interface WorkflowEditorState {
  selectedNode: string | null
  setSelectedNode: (nodeId: string | null) => void
  nodes: any[]
  setNodes: (nodes: any[]) => void
  edges: any[]
  setEdges: (edges: any[]) => void
}

export const useWorkflowEditorStore = create<WorkflowEditorState>()((set) => ({
  selectedNode: null,
  setSelectedNode: (nodeId) => set({ selectedNode: nodeId }),
  nodes: [],
  setNodes: (nodes) => set({ nodes }),
  edges: [],
  setEdges: (edges) => set({ edges }),
}))
