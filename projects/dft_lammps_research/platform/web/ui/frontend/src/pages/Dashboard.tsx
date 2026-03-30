import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  FolderIcon,
  BeakerIcon,
  QueueListIcon,
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline'
import { monitoringApi, projectsApi, workflowsApi, tasksApi } from '../api/client'
import StatsCard from '../components/StatsCard'
import ActivityFeed from '../components/ActivityFeed'
import TrainingChart from '../components/charts/TrainingChart'

export default function Dashboard() {
  const { data: stats } = useQuery({
    queryKey: ['system-stats'],
    queryFn: () => monitoringApi.getSystemStats().then(res => res.data),
    refetchInterval: 10000,
  })

  const { data: activeWorkflows } = useQuery({
    queryKey: ['active-workflows'],
    queryFn: () => monitoringApi.getActiveWorkflows({ limit: 5 }).then(res => res.data),
    refetchInterval: 5000,
  })

  const { data: recentTasks } = useQuery({
    queryKey: ['recent-tasks'],
    queryFn: () => monitoringApi.getRecentTasks({ limit: 10 }).then(res => res.data),
    refetchInterval: 5000,
  })

  const workflows = stats?.workflows || {}
  const tasks = stats?.tasks || {}

  const statsCards = [
    { name: 'Total Projects', value: '-', icon: FolderIcon, href: '/projects' },
    { name: 'Active Workflows', value: (workflows.running || 0) + (workflows.queued || 0), icon: BeakerIcon, href: '/workflows' },
    { name: 'Running Tasks', value: tasks.running || 0, icon: QueueListIcon, href: '/tasks' },
    { name: 'Completed', value: tasks.completed || 0, icon: CheckCircleIcon, href: '/tasks' },
  ]

  return (
    <div className="space-y-8">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {statsCards.map((card) => (
          <Link key={card.name} to={card.href}>
            <StatsCard
              name={card.name}
              value={card.value}
              icon={card.icon}
            />
          </Link>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-8">
          {/* Training Progress */}
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900">ML Training Progress</h3>
                <Link to="/monitoring" className="text-sm text-primary-600 hover:text-primary-500">
                  View details →
                </Link>
              </div>
            </div>
            <div className="p-6">
              <TrainingChart />
            </div>
          </div>

          {/* Active Workflows */}
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900">Active Workflows</h3>
                <Link to="/workflows" className="text-sm text-primary-600 hover:text-primary-500">
                  View all →
                </Link>
              </div>
            </div>
            <div className="divide-y divide-gray-200">
              {activeWorkflows?.map((workflow: any) => (
                <div key={workflow.id} className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Link to={`/workflows/${workflow.id}`} className="text-sm font-medium text-gray-900 hover:text-primary-600">
                        {workflow.name}
                      </Link>
                      <p className="text-sm text-gray-500">{workflow.type}</p>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="w-32">
                        <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                          <span>{workflow.progress.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${workflow.progress}%` }}
                          />
                        </div>
                      </div>
                      
                      <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
                        workflow.status === 'running'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {workflow.status}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
              
              {(!activeWorkflows || activeWorkflows.length === 0) && (
                <div className="p-8 text-center text-gray-500">
                  No active workflows
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-8">
          {/* Resource Usage */}
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">Resource Usage</h3>
            </div>
            <div className="p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <ClockIcon className="h-5 w-5 text-gray-400" />
                  <span className="ml-2 text-sm text-gray-600">CPU Hours</span>
                </div>
                <span className="text-sm font-medium text-gray-900">{stats?.resources?.total_cpu_hours || 0}</span>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <ChartBarIcon className="h-5 w-5 text-gray-400" />
                  <span className="ml-2 text-sm text-gray-600">Memory (GB)</span>
                </div>
                <span className="text-sm font-medium text-gray-900">{stats?.resources?.total_memory_gb || 0}</span>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">Recent Activity</h3>
            </div>
            <div className="p-0">
              <ActivityFeed tasks={recentTasks || []} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
