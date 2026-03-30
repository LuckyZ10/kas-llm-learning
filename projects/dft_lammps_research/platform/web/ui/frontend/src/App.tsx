import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { WebSocketProvider } from './contexts/WebSocketContext'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Projects from './pages/Projects'
import ProjectDetail from './pages/ProjectDetail'
import WorkflowEditor from './pages/WorkflowEditor'
import WorkflowDetail from './pages/WorkflowDetail'
import Tasks from './pages/Tasks'
import Monitoring from './pages/Monitoring'
import Screening from './pages/Screening'
import StructureViewer from './pages/StructureViewer'
import Reports from './pages/Reports'
import Settings from './pages/Settings'

function App() {
  return (
    <WebSocketProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/projects" element={<Projects />} />
            <Route path="/projects/:id" element={<ProjectDetail />} />
            <Route path="/workflows" element={<WorkflowEditor />} />
            <Route path="/workflows/:id" element={<WorkflowDetail />} />
            <Route path="/tasks" element={<Tasks />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="/screening" element={<Screening />} />
            <Route path="/structures" element={<StructureViewer />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Layout>
      </Router>
    </WebSocketProvider>
  )
}

export default App
