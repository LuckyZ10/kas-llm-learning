import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { screeningApi } from '../api/client'
import { MagnifyingGlassIcon, FunnelIcon, ArrowsRightLeftIcon } from '@heroicons/react/24/outline'

interface ScreeningResult {
  id: string
  structure_id: string
  formula: string
  formation_energy: number
  band_gap: number
  ionic_conductivity: number
  overall_score: number
}

export default function Screening() {
  const [search, setSearch] = useState('')
  const [minConductivity, setMinConductivity] = useState('')
  const [selectedResults, setSelectedResults] = useState<string[]>([])

  const { data, isLoading } = useQuery({
    queryKey: ['screening-results', { search, minConductivity }],
    queryFn: () => screeningApi.getAll({ 
      formula: search,
      min_ionic_conductivity: minConductivity ? parseFloat(minConductivity) : undefined,
      page_size: 100,
    }).then(res => res.data),
  })

  const results: ScreeningResult[] = data?.items || []

  const scatterData = [
    {
      x: results.map(r => r.formation_energy),
      y: results.map(r => r.band_gap),
      mode: 'markers',
      type: 'scatter',
      text: results.map(r => r.formula),
      marker: {
        color: results.map(r => r.ionic_conductivity || 0),
        colorscale: 'Viridis',
        size: 10,
        showscale: true,
        colorbar: { title: 'Ionic Cond.' },
      },
      hovertemplate: '<b>%{text}</b><br>Formation: %{x:.3f} eV/atom<br>Band Gap: %{y:.3f} eV<extra></extra>',
    },
  ]

  const toggleSelection = (id: string) => {
    setSelectedResults(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Screening Results</h1>
        
        <div className="flex items-center gap-3">
          <button
            disabled={selectedResults.length < 2}
            className="btn-secondary disabled:opacity-50"
          >
            <ArrowsRightLeftIcon className="h-5 w-5 mr-2" />
            Compare ({selectedResults.length})
          </button>
          
          <button className="btn-primary">
            Export CSV
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="card p-4">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-1">Search Formula</label>
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="e.g., LiMnO2"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="form-input pl-10"
              />
            </div>
          </div>
          
          <div className="w-48">
            <label className="block text-sm font-medium text-gray-700 mb-1">Min Conductivity</label>
            <input
              type="number"
              step="0.001"
              placeholder="S/cm"
              value={minConductivity}
              onChange={(e) => setMinConductivity(e.target.value)}
              className="form-input"
            />
          </div>
          
          <button className="btn-secondary">
            <FunnelIcon className="h-5 w-5 mr-2" />
            More Filters
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Scatter Plot */}
        <div className="card">
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Property Map</h3>
            <p className="text-sm text-gray-500">Formation Energy vs Band Gap (colored by Ionic Conductivity)</p>
          </div>
          
          <div className="p-4">
            <Plot
              data={scatterData}
              layout={{
                autosize: true,
                height: 450,
                xaxis: { title: 'Formation Energy (eV/atom)' },
                yaxis: { title: 'Band Gap (eV)' },
                hovermode: 'closest',
              }}
              config={{ responsive: true, displayModeBar: true }}
            />
          </div>
        </div>

        {/* Results Table */}
        <div className="card overflow-hidden">
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Top Candidates</h3>
            <p className="text-sm text-gray-500">{results.length} structures found</p>
          </div>
          
          <div className="overflow-auto max-h-[450px]">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-4 py-3">
                    <input
                      type="checkbox"
                      className="rounded border-gray-300"
                      checked={selectedResults.length === results.length && results.length > 0}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedResults(results.map(r => r.id))
                        } else {
                          setSelectedResults([])
                        }
                      }}
                    />
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Formula</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Formation (eV/atom)</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Band Gap (eV)</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Ionic Cond.</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Score</th>
                </tr>
              </thead>
              
              <tbody className="bg-white divide-y divide-gray-200">
                {isLoading ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-gray-500">Loading...</td>
                  </tr>
                ) : (
                  results.slice(0, 50).map((result) => (
                    <tr
                      key={result.id}
                      className={selectedResults.includes(result.id) ? 'bg-primary-50' : 'hover:bg-gray-50'}
                    >
                      <td className="px-4 py-3">
                        <input
                          type="checkbox"
                          className="rounded border-gray-300"
                          checked={selectedResults.includes(result.id)}
                          onChange={() => toggleSelection(result.id)}
                        />
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">{result.formula}</td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                        {result.formation_energy?.toFixed(3)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                        {result.band_gap?.toFixed(3)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                        {result.ionic_conductivity?.toExponential(2)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium">
                        <span className={result.overall_score > 0.8 ? 'text-green-600' : 'text-gray-900'}>
                          {result.overall_score?.toFixed(3)}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
