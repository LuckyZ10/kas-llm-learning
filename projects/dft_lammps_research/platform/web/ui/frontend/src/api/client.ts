import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Project API
export const projectsApi = {
  getAll: (params?: { status?: string; search?: string; page?: number; page_size?: number }) =>
    api.get('/projects', { params }),
  getById: (id: string) => api.get(`/projects/${id}`),
  create: (data: any) => api.post('/projects', data),
  update: (id: string, data: any) => api.patch(`/projects/${id}`, data),
  delete: (id: string) => api.delete(`/projects/${id}`),
}

// Workflow API
export const workflowsApi = {
  getAll: (params?: { project_id?: string; status?: string; page?: number; page_size?: number }) =>
    api.get('/workflows', { params }),
  getById: (id: string) => api.get(`/workflows/${id}`),
  create: (data: any) => api.post('/workflows', data),
  update: (id: string, data: any) => api.patch(`/workflows/${id}`, data),
  execute: (id: string, data?: any) => api.post(`/workflows/${id}/execute`, data),
  pause: (id: string) => api.post(`/workflows/${id}/pause`),
  cancel: (id: string) => api.post(`/workflows/${id}/cancel`),
  delete: (id: string) => api.delete(`/workflows/${id}`),
}

// Task API
export const tasksApi = {
  getAll: (params?: { workflow_id?: string; status?: string; page?: number; page_size?: number }) =>
    api.get('/tasks', { params }),
  getById: (id: string) => api.get(`/tasks/${id}`),
  create: (data: any) => api.post('/tasks', data),
  update: (id: string, data: any) => api.patch(`/tasks/${id}`, data),
  getLogs: (id: string, params?: { lines?: number }) => api.get(`/tasks/${id}/logs`, { params }),
}

// Screening API
export const screeningApi = {
  getAll: (params?: { project_id?: string; formula?: string; page?: number; page_size?: number }) =>
    api.get('/screening', { params }),
  getById: (id: string) => api.get(`/screening/${id}`),
  filter: (data: any, params?: { page?: number; page_size?: number }) => api.post('/screening/filter', data, { params }),
  compare: (data: any) => api.post('/screening/compare', data),
}

// Monitoring API
export const monitoringApi = {
  getSystemStats: () => api.get('/monitoring/stats'),
  getActiveWorkflows: (params?: { limit?: number }) => api.get('/monitoring/workflows/active', { params }),
  getRecentTasks: (params?: { status?: string; limit?: number }) => api.get('/monitoring/tasks/recent', { params }),
  getResourceUsage: () => api.get('/monitoring/resources'),
  getTrainingMetrics: (modelId?: string) => api.get('/monitoring/training', { params: { model_id: modelId } }),
  getMDMetrics: (trajectoryId: string, metric: string) => api.get(`/monitoring/md/${trajectoryId}`, { params: { metric } }),
  getALProgress: (projectId?: string) => api.get('/monitoring/al/progress', { params: { project_id: projectId } }),
}

// Files API
export const filesApi = {
  list: (params?: { path?: string; pattern?: string }) => api.get('/files/list', { params }),
  download: (filePath: string) => api.get(`/files/download/${filePath}`, { responseType: 'blob' }),
  upload: (file: File, path?: string) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/files/upload', formData, {
      params: { path },
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
}

// Reports API
export const reportsApi = {
  generateProject: (projectId: string, params?: { format?: string; include_structures?: boolean; include_charts?: boolean }) =>
    api.post(`/reports/project/${projectId}`, null, { params, responseType: 'blob' }),
  generateWorkflow: (workflowId: string, params?: { format?: string }) =>
    api.post(`/reports/workflow/${workflowId}`, null, { params, responseType: 'blob' }),
  generateScreening: (projectId: string, params?: { top_n?: number; format?: string }) =>
    api.post('/reports/screening', null, { params: { ...params, project_id: projectId }, responseType: 'blob' }),
}

// Auth API
export const authApi = {
  login: (username: string, password: string) =>
    api.post('/auth/login', { username, password }),
  register: (data: any) => api.post('/auth/register', data),
  getCurrentUser: () => api.get('/auth/me'),
}

export default api
