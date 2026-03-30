import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import api from '../utils/api';

export const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      
      login: async (username, password) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/auth/login', { username, password });
          const { access_token, user } = response.data;
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
          });
          
          // Set default auth header
          api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          return true;
        } catch (error) {
          set({
            error: error.response?.data?.detail || 'Login failed',
            isLoading: false,
          });
          return false;
        }
      },
      
      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        });
        delete api.defaults.headers.common['Authorization'];
      },
      
      checkAuth: () => {
        const { token } = get();
        if (token) {
          api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          set({ isAuthenticated: true });
        }
      },
      
      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token, user: state.user }),
    }
  )
);

export const useTaskStore = create((set, get) => ({
  tasks: [],
  currentTask: null,
  isLoading: false,
  error: null,
  totalTasks: 0,
  
  fetchTasks: async (params = {}) => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.get('/tasks', { params });
      set({
        tasks: response.data.tasks,
        totalTasks: response.data.total,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error.response?.data?.detail || 'Failed to fetch tasks',
        isLoading: false,
      });
    }
  },
  
  fetchTask: async (taskId) => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.get(`/tasks/${taskId}`);
      set({
        currentTask: response.data,
        isLoading: false,
      });
      return response.data;
    } catch (error) {
      set({
        error: error.response?.data?.detail || 'Failed to fetch task',
        isLoading: false,
      });
      return null;
    }
  },
  
  submitTask: async (taskData) => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.post('/tasks', taskData);
      set({ isLoading: false });
      return response.data;
    } catch (error) {
      set({
        error: error.response?.data?.detail || 'Failed to submit task',
        isLoading: false,
      });
      return null;
    }
  },
  
  cancelTask: async (taskId) => {
    try {
      const response = await api.post(`/tasks/${taskId}/cancel`);
      return response.data;
    } catch (error) {
      set({ error: error.response?.data?.detail || 'Failed to cancel task' });
      return null;
    }
  },
  
  deleteTask: async (taskId) => {
    try {
      await api.delete(`/tasks/${taskId}`);
      set((state) => ({
        tasks: state.tasks.filter((t) => t.task_id !== taskId),
      }));
      return true;
    } catch (error) {
      set({ error: error.response?.data?.detail || 'Failed to delete task' });
      return false;
    }
  },
  
  clearError: () => set({ error: null }),
  clearCurrentTask: () => set({ currentTask: null }),
}));

export const useStructureStore = create((set) => ({
  structures: [],
  currentStructure: null,
  isLoading: false,
  error: null,
  
  searchStructures: async (query) => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.get('/structures/search', { params: { query } });
      set({
        structures: response.data.results,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error.response?.data?.detail || 'Failed to search structures',
        isLoading: false,
      });
    }
  },
  
  setCurrentStructure: (structure) => set({ currentStructure: structure }),
  clearStructures: () => set({ structures: [], currentStructure: null }),
}));

export const usePresetStore = create((set) => ({
  presets: null,
  isLoading: false,
  error: null,
  
  fetchPresets: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.get('/presets');
      set({
        presets: response.data,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error.response?.data?.detail || 'Failed to fetch presets',
        isLoading: false,
      });
    }
  },
}));
