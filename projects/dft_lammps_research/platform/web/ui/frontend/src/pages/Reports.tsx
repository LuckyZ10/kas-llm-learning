import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { projectsApi, reportsApi } from '../api/client'
import { DocumentArrowDownIcon, FilePdfIcon, FileCodeIcon, TableCellsIcon } from '@heroicons/react/24/outline'

export default function Reports() {
  const [selectedProject, setSelectedProject] = useState('')

  const { data: projects } = useQuery({
    queryKey: ['projects'],
    queryFn: () => projectsApi.getAll().then(res => res.data),
  })

  const generateReport = async (type: string, format: string) => {
    if (!selectedProject) return

    let response
    switch (type) {
      case 'project':
        response = await reportsApi.generateProject(selectedProject, { format })
        break
      case 'screening':
        response = await reportsApi.generateScreening(selectedProject, { format, top_n: 50 })
        break
    }

    // Download file
    const blob = new Blob([response.data])
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `report_${selectedProject}.${format}`
    link.click()
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Generate Reports</h1>

      <div className="card p-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">Select Project</label>
        <select
          value={selectedProject}
          onChange={(e) => setSelectedProject(e.target.value)}
          className="form-input max-w-md"
        >
          <option value="">Choose a project...</option>
          {projects?.items?.map((p: any) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="card p-6">
          <div className="flex items-center mb-4">
            <div className="p-3 bg-red-100 rounded-lg">
              <DocumentArrowDownIcon className="h-6 w-6 text-red-600" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium">Project Report</h3>
              <p className="text-sm text-gray-500">Complete project summary</p>
            </div>
          </div>
          
          <div className="space-y-2">
            <button
              onClick={() => generateReport('project', 'pdf')}
              disabled={!selectedProject}
              className="w-full btn-secondary disabled:opacity-50"
            >
              <FilePdfIcon className="h-5 w-5 mr-2 inline" />
              Download PDF
            </button>
            <button
              onClick={() => generateReport('project', 'html')}
              disabled={!selectedProject}
              className="w-full btn-secondary disabled:opacity-50"
            >
              <FileCodeIcon className="h-5 w-5 mr-2 inline" />
              Download HTML
            </button>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center mb-4">
            <div className="p-3 bg-green-100 rounded-lg">
              <TableCellsIcon className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium">Screening Results</h3>
              <p className="text-sm text-gray-500">Top 50 candidates</p>
            </div>
          </div>
          
          <div className="space-y-2">
            <button
              onClick={() => generateReport('screening', 'pdf')}
              disabled={!selectedProject}
              className="w-full btn-secondary disabled:opacity-50"
            >
              <FilePdfIcon className="h-5 w-5 mr-2 inline" />
              Download PDF
            </button>
            <button
              onClick={() => generateReport('screening', 'csv')}
              disabled={!selectedProject}
              className="w-full btn-secondary disabled:opacity-50"
            >
              <TableCellsIcon className="h-5 w-5 mr-2 inline" />
              Download CSV
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
