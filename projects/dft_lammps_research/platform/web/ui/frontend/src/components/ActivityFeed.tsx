import React from 'react'
import { formatDistanceToNow } from 'date-fns'
import { CheckCircleIcon, XCircleIcon, ClockIcon, PlayIcon } from '@heroicons/react/24/outline'

interface Task {
  id: string
  name: string
  status: string
  task_type: string
  created_at: string
}

interface ActivityFeedProps {
  tasks: Task[]
}

const statusConfig: Record<string, { icon: any; color: string; bg: string }> = {
  completed: { icon: CheckCircleIcon, color: 'text-green-600', bg: 'bg-green-50' },
  failed: { icon: XCircleIcon, color: 'text-red-600', bg: 'bg-red-50' },
  running: { icon: PlayIcon, color: 'text-blue-600', bg: 'bg-blue-50' },
  pending: { icon: ClockIcon, color: 'text-gray-600', bg: 'bg-gray-50' },
}

export default function ActivityFeed({ tasks }: ActivityFeedProps) {
  if (!tasks || tasks.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        No recent activity
      </div>
    )
  }

  return (
    <ul className="divide-y divide-gray-200">
      {tasks.slice(0, 10).map((task) => {
        const config = statusConfig[task.status] || statusConfig.pending
        const Icon = config.icon

        return (
          <li key={task.id} className="p-4 hover:bg-gray-50 transition-colors">
            <div className="flex items-start space-x-3">
              <div className={`flex-shrink-0 h-8 w-8 rounded-full ${config.bg} flex items-center justify-center`}>
                <Icon className={`h-4 w-4 ${config.color}`} />
              </div>
              
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">{task.name}</p>
                <p className="text-xs text-gray-500">{task.task_type}</p>
              </div>
              
              <div className="flex-shrink-0 text-xs text-gray-400">
                {task.created_at && formatDistanceToNow(new Date(task.created_at), { addSuffix: true })}
              </div>
            </div>
          </li>
        )
      })}
    </ul>
  )
}
