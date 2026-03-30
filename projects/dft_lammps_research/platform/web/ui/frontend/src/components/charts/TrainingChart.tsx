import React from 'react'
import Plot from 'react-plotly.js'

// Mock data for demonstration
const mockTrainingData = {
  steps: Array.from({ length: 100 }, (_, i) => i * 100),
  loss: Array.from({ length: 100 }, (_, i) => 10 * Math.exp(-i / 30) + 0.1 + Math.random() * 0.05),
  force_rmse: Array.from({ length: 100 }, (_, i) => 2 * Math.exp(-i / 25) + 0.05 + Math.random() * 0.02),
}

export default function TrainingChart() {
  const data = [
    {
      x: mockTrainingData.steps,
      y: mockTrainingData.loss,
      type: 'scatter',
      mode: 'lines',
      name: 'Loss',
      line: { color: '#3b82f6', width: 2 },
      yaxis: 'y',
    },
    {
      x: mockTrainingData.steps,
      y: mockTrainingData.force_rmse,
      type: 'scatter',
      mode: 'lines',
      name: 'Force RMSE',
      line: { color: '#10b981', width: 2 },
      yaxis: 'y2',
    },
  ]

  const layout = {
    autosize: true,
    height: 350,
    margin: { t: 20, r: 60, b: 40, l: 60 },
    xaxis: {
      title: 'Training Steps',
      gridcolor: '#e5e7eb',
    },
    yaxis: {
      title: 'Loss',
      type: 'log',
      gridcolor: '#e5e7eb',
    },
    yaxis2: {
      title: 'Force RMSE (eV/Å)',
      overlaying: 'y',
      side: 'right',
      type: 'log',
    },
    showlegend: true,
    legend: {
      x: 0.5,
      y: 1.1,
      orientation: 'h',
      xanchor: 'center',
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
  }

  return (
    <Plot
      data={data as any}
      layout={layout as any}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
    />
  )
}
