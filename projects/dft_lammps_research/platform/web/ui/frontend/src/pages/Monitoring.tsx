import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { monitoringApi } from '../api/client'
import { CpuChipIcon, ServerIcon, CircleStackIcon, BoltIcon } from '@heroicons/react/24/outline'

export default function Monitoring() {
  const [selectedTab, setSelectedTab] = useState<'overview' | 'training' | 'md' | 'al'>('overview')

  const { data: stats } = useQuery({
    queryKey: ['system-stats'],
    queryFn: () => monitoringApi.getSystemStats().then(res => res.data),
    refetchInterval: 5000,
  })

  const { data: trainingMetrics } = useQuery({
    queryKey: ['training-metrics'],
    queryFn: () => monitoringApi.getTrainingMetrics().then(res => res.data),
    refetchInterval: 10000,
    enabled: selectedTab === 'training' || selectedTab === 'overview',
  })

  const { data: activeWorkflows } = useQuery({
    queryKey: ['active-workflows-monitoring'],
    queryFn: () => monitoringApi.getActiveWorkflows({ limit: 10 }).then(res => res.data),
    refetchInterval: 5000,
  })

  const { data: alProgress } = useQuery({
    queryKey: ['al-progress'],
    queryFn: () => monitoringApi.getALProgress().then(res => res.data),
    refetchInterval: 10000,
    enabled: selectedTab === 'al',
  })

  const trainingChartData = trainingMetrics?.history ? [
    {
      x: trainingMetrics.history.batch,
      y: trainingMetrics.history.loss,
      type: 'scatter',
      mode: 'lines',
      name: 'Loss',
      line: { color: '#3b82f6' },
    },
    {
      x: trainingMetrics.history.batch,
      y: trainingMetrics.history.force_rmse,
      type: 'scatter',
      mode: 'lines',
      name: 'Force RMSE',
      line: { color: '#10b981' },
      yaxis: 'y2',
    },
  ] : []

  return (
    <div className="space-y-6">
      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', name: 'Overview' },
            { id: 'training', name: 'ML Training' },
            { id: 'md', name: 'MD Simulation' },
            { id: 'al', name: 'Active Learning' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id as any)}
              className={`
                py-4 px-1 border-b-2 font-medium text-sm
                ${selectedTab === tab.id
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {selectedTab === 'overview' && (
        <div className="space-y-6">
          {/* Resource Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="card p-6">
              <div className="flex items-center">
                <CpuChipIcon className="h-8 w-8 text-primary-500" />
                <div className="ml-4">
                  <p className="text-sm text-gray-500">Active Workflows</p>
                  <p className="text-2xl font-semibold text-gray-900">{activeWorkflows?.length || 0}</p>
                </div>
              </div>
            </div>

            <div className="card p-6">
              <div className="flex items-center">
                <ServerIcon className="h-8 w-8 text-green-500" />
                <div className="ml-4">
                  <p className="text-sm text-gray-500">Running Tasks</p>
                  <p className="text-2xl font-semibold text-gray-900">{stats?.tasks?.running || 0}</p>
                </div>
              </div>
            </div>

            <div className="card p-6">
              <div className="flex items-center">
                <CircleStackIcon className="h-8 w-8 text-purple-500" />
                <div className="ml-4">
                  <p className="text-sm text-gray-500">CPU Hours</p>
                  <p className="text-2xl font-semibold text-gray-900">{stats?.resources?.total_cpu_hours?.toFixed(1) || 0}</p>
                </div>
              </div>
            </div>

            <div className="card p-6">
              <div className="flex items-center">
                <BoltIcon className="h-8 w-8 text-yellow-500" />
                <div className="ml-4">
                  <p className="text-sm text-gray-500">Memory Used</p>
                  <p className="text-2xl font-semibold text-gray-900">{stats?.resources?.total_memory_gb?.toFixed(1) || 0} GB</p>
                </div>
              </div>
            </div>
          </div>

          {/* Active Workflows Table */}
          <div className="card">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">Active Workflows</h3>
            </div>
            
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Progress</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {activeWorkflows?.map((workflow: any) => (
                    <tr key={workflow.id}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{workflow.name}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{workflow.type}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div
                            className="bg-primary-600 h-2.5 rounded-full"
                            style={{ width: `${workflow.progress}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-500">{workflow.progress.toFixed(1)}%</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          workflow.status === 'running'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {workflow.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {selectedTab === 'training' && trainingMetrics?.history && (
        <div className="card">
          <div className="p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Training Progress</h3>
            <Plot
              data={trainingChartData}
              layout={{
                autosize: true,
                height: 500,
                xaxis: { title: 'Training Steps' },
                yaxis: { title: 'Loss', type: 'log' },
                yaxis2: {
                  title: 'Force RMSE (eV/Å)',
                  overlaying: 'y',
                  side: 'right',
                },
                showlegend: true,
              }}
              config={{ responsive: true }}
            />
          </div>
        </div>
      )}

      {selectedTab === 'al' && alProgress && (
        <div className="card">
          <div className="p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Active Learning Progress</h3>
            <div className="space-y-4">
              {alProgress.iterations?.map((iter: any) => (
                <div key={iter.iteration} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Iteration {iter.iteration}</span>
                    <div className="flex gap-4 text-sm text-gray-500">
                      <span>Accurate: {iter.accurate || 0}</span>
                      <span>Candidates: {iter.candidate || 0}</span>
                      <span>Failed: {iter.failed || 0}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
