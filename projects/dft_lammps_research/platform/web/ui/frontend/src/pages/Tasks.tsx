import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { tasksApi } from '../api/client'
import { PlayIcon, PauseIcon, CheckCircleIcon, XCircleIcon, ClockIcon } from '@heroicons/react/24/outline'

const statusIcons: Record<string, any> = {
  pending: ClockIcon,
  queued: ClockIcon,
  running: PlayIcon,
  completed: CheckCircleIcon,
  failed: XCircleIcon,
  cancelled: XCircleIcon,
}

const statusColors: Record<string, string> = {
  pending: 'text-gray-500',
  queued: 'text-blue-500',
  running: 'text-green-500',
  completed: 'text-green-600',
  failed: 'text-red-600',
  cancelled: 'text-gray-500',
}

export default function Tasks() {
  const [statusFilter, setStatusFilter] = useState('')
  const [typeFilter, setTypeFilter] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['tasks', { status: statusFilter, type: typeFilter }],
    queryFn: () => tasksApi.getAll({ status: statusFilter || undefined, task_type: typeFilter || undefined }).then(res => res.data),
    refetchInterval: 5000,
  })

  const tasks = data?.items || []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Tasks</h1>
        
        <div className="flex gap-3">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="form-input w-40"
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
          
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="form-input w-48"
          >
            <option value="">All Types</option>
            <option value="dft_calculation">DFT Calculation</option>
            <option value="md_simulation">MD Simulation</option>
            <option value="ml_training">ML Training</option>
          </select>
        </div>
      </div>

      <div className="card overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Task</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Resources</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
            </tr>
          </thead>
          
          <tbody className="bg-white divide-y divide-gray-200">
            {isLoading ? (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-gray-500">Loading...</td>
              </tr>
            ) : (
              tasks.map((task: any) => {
                const StatusIcon = statusIcons[task.status] || ClockIcon
                return (
                  <tr key={task.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-900">{task.name}</div>
                      <div className="text-xs text-gray-500 truncate max-w-xs">{task.id}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {task.task_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className={`flex items-center text-sm ${statusColors[task.status]}`}>
                        <StatusIcon className="h-4 w-4 mr-1" />
                        <span className="capitalize">{task.status}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {task.cpu_time_seconds > 0 && (
                        <div>{(task.cpu_time_seconds / 3600).toFixed(2)} h</div>
                      )}
                      {task.memory_peak_mb > 0 && (
                        <div>{(task.memory_peak_mb / 1024).toFixed(2)} GB</div>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(task.created_at).toLocaleString()}
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
