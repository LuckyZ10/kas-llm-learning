import React from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeftIcon, BeakerIcon, FolderIcon } from '@heroicons/react/24/outline'
import { projectsApi, workflowsApi } from '../api/client'

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>()

  const { data: project } = useQuery({
    queryKey: ['project', id],
    queryFn: () => projectsApi.getById(id!).then(res => res.data),
    enabled: !!id,
  })

  const { data: workflows } = useQuery({
    queryKey: ['project-workflows', id],
    queryFn: () => workflowsApi.getAll({ project_id: id }).then(res => res.data),
    enabled: !!id,
  })

  if (!project) {
    return <div className="text-center py-12">Loading...</div>
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link to="/projects" className="p-2 hover:bg-gray-100 rounded-lg">
          <ArrowLeftIcon className="h-5 w-5" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{project.name}</h1>
          <p className="text-gray-500">{project.id}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-lg font-medium">Description</h2>
            </div>
            <div className="p-6">
              <p className="text-gray-700">{project.description || 'No description provided.'}</p>
            </div>
          </div>

          <div className="card">
            <div className="p-6 border-b border-gray-200 flex items-center justify-between">
              <h2 className="text-lg font-medium">Workflows</h2>
              <Link to="/workflows" className="btn-primary text-sm">
                Create Workflow
              </Link>
            </div>
            
            <div className="divide-y divide-gray-200">
              {workflows?.items?.map((workflow: any) => (
                <Link
                  key={workflow.id}
                  to={`/workflows/${workflow.id}`}
                  className="block p-6 hover:bg-gray-50"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <BeakerIcon className="h-5 w-5 text-gray-400 mr-3" />
                      <div>
                        <p className="font-medium text-gray-900">{workflow.name}</p>
                        <p className="text-sm text-gray-500">{workflow.workflow_type}</p>
                      </div>
                    </div>
                    
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      workflow.status === 'running'
                        ? 'bg-green-100 text-green-800'
                        : workflow.status === 'completed'
                        ? 'bg-blue-100 text-blue-800'
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {workflow.status}
                    </span>
                  </div>
                </Link>
              ))}
              
              {(!workflows?.items || workflows.items.length === 0) && (
                <div className="p-8 text-center text-gray-500">
                  No workflows yet. Create one to get started.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-lg font-medium">Project Info</h2>
            </div>
            
            <div className="p-6 space-y-4">
              <div>
                <label className="text-sm font-medium text-gray-500">Type</label>
                <p className="mt-1 text-gray-900">{project.project_type}</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-gray-500">Material System</label>
                <p className="mt-1 text-gray-900">{project.material_system || 'N/A'}</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-gray-500">Work Directory</label>
                <p className="mt-1 text-sm text-gray-900 font-mono">{project.work_directory}</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-gray-500">Created</label>
                <p className="mt-1 text-gray-900">{new Date(project.created_at).toLocaleString()}</p>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-lg font-medium">Statistics</h2>
            </div>
            
            <div className="p-6 grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">{project.total_structures}</p>
                <p className="text-sm text-gray-500">Total Structures</p>
              </div>
              
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <p className="text-2xl font-bold text-green-600">{project.completed_calculations}</p>
                <p className="text-sm text-gray-500">Completed</p>
              </div>
              
              <div className="text-center p-4 bg-red-50 rounded-lg">
                <p className="text-2xl font-bold text-red-600">{project.failed_calculations}</p>
                <p className="text-sm text-gray-500">Failed</p>
              </div>
              
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <p className="text-2xl font-bold text-blue-600">{workflows?.items?.length || 0}</p>
                <p className="text-sm text-gray-500">Workflows</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
