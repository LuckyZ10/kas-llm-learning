import React from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeftIcon, PlayIcon, PauseIcon, StopIcon } from '@heroicons/react/24/outline'
import { workflowsApi, tasksApi } from '../api/client'

export default function WorkflowDetail() {
  const { id } = useParams<{ id: string }>()

  const { data: workflow } = useQuery({
    queryKey: ['workflow', id],
    queryFn: () => workflowsApi.getById(id!).then(res => res.data),
    enabled: !!id,
  })

  const { data: tasks } = useQuery({
    queryKey: ['workflow-tasks', id],
    queryFn: () => tasksApi.getAll({ workflow_id: id }).then(res => res.data),
    enabled: !!id,
  })

  if (!workflow) {
    return <div className="text-center py-12">Loading...</div>
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/workflows" className="p-2 hover:bg-gray-100 rounded-lg">
            <ArrowLeftIcon className="h-5 w-5" />
          </Link>
          
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{workflow.name}</h1>
            <p className="text-gray-500">{workflow.workflow_type}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {workflow.status === 'running' ? (
            <>
              <button className="btn-secondary">
                <PauseIcon className="h-5 w-5 mr-2" />
                Pause
              </button>
              <button className="btn-secondary">
                <StopIcon className="h-5 w-5 mr-2" />
                Stop
              </button>
            <//>
          ) : (
            <button className="btn-primary">
              <PlayIcon className="h-5 w-5 mr-2" />
              Execute
            </button>
          )}
        </div>
      </div>

      {/* Progress */}
      <div className="card">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-500">Progress</p>
              <p className="text-3xl font-bold">{workflow.progress_percent.toFixed(1)}%</p>
            </div>
            
            <span className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${
              workflow.status === 'running'
                ? 'bg-green-100 text-green-800'
                : workflow.status === 'completed'
                ? 'bg-blue-100 text-blue-800'
                : 'bg-gray-100 text-gray-800'
            }`}>
              {workflow.status}
            </span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-primary-600 h-3 rounded-full transition-all duration-500"
              style={{ width: `${workflow.progress_percent}%` }}
            />
          </div>
          
          <div className="mt-4 flex gap-6 text-sm text-gray-500">
            <span>{workflow.completed_tasks} completed</span>
            <span>{workflow.total_tasks - workflow.completed_tasks} remaining</span>
            <span>{workflow.failed_tasks} failed</span>
          </div>
        </div>
      </div>

      {/* Tasks */}
      <div className="card">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-lg font-medium">Tasks</h2>
        </div>
        
        <div className="divide-y divide-gray-200">
          {tasks?.items?.map((task: any) => (
            <div key={task.id} className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">{task.name}</p>
                  <p className="text-sm text-gray-500">{task.task_type}</p>
                </div>
                
                <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                  task.status === 'completed'
                    ? 'bg-green-100 text-green-800'
                    : task.status === 'running'
                    ? 'bg-blue-100 text-blue-800'
                    : task.status === 'failed'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  {task.status}
                </span>
              </div>
              
              {task.error_message && (
                <p className="mt-2 text-sm text-red-600">{task.error_message}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
