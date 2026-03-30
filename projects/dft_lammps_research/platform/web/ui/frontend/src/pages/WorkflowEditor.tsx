import React, { useState, useCallback } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Connection,
  Edge,
  Node,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { PlayIcon, PauseIcon, StopIcon, SaveIcon, PlusIcon } from '@heroicons/react/24/outline'

// Node types
const nodeTypes = {
  dft_calculation: DFTNode,
  md_simulation: MDNode,
  ml_training: MLNode,
  structure_input: InputNode,
  analysis: AnalysisNode,
}

function DFTNode({ data }: { data: any }) {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-blue-500">
      <div className="font-bold text-blue-700">{data.label}</div>
      <div className="text-xs text-gray-500">DFT Calculation</div>
    </div>
  )
}

function MDNode({ data }: { data: any }) {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-green-500">
      <div className="font-bold text-green-700">{data.label}</div>
      <div className="text-xs text-gray-500">MD Simulation</div>
    </div>
  )
}

function MLNode({ data }: { data: any }) {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-purple-500">
      <div className="font-bold text-purple-700">{data.label}</div>
      <div className="text-xs text-gray-500">ML Training</div>
    </div>
  )
}

function InputNode({ data }: { data: any }) {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-gray-500">
      <div className="font-bold text-gray-700">{data.label}</div>
      <div className="text-xs text-gray-500">Input</div>
    </div>
  )
}

function AnalysisNode({ data }: { data: any }) {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-orange-500">
      <div className="font-bold text-orange-700">{data.label}</div>
      <div className="text-xs text-gray-500">Analysis</div>
    </div>
  )
}

const initialNodes: Node[] = [
  {
    id: '1',
    type: 'structure_input',
    position: { x: 100, y: 100 },
    data: { label: 'Initial Structures' },
  },
  {
    id: '2',
    type: 'dft_calculation',
    position: { x: 300, y: 100 },
    data: { label: 'DFT Relaxation' },
  },
  {
    id: '3',
    type: 'ml_training',
    position: { x: 500, y: 100 },
    data: { label: 'Train ML Potential' },
  },
  {
    id: '4',
    type: 'md_simulation',
    position: { x: 700, y: 100 },
    data: { label: 'MD Simulation' },
  },
  {
    id: '5',
    type: 'analysis',
    position: { x: 900, y: 100 },
    data: { label: 'Results Analysis' },
  },
]

const initialEdges: Edge[] = [
  { id: 'e1-2', source: '1', target: '2' },
  { id: 'e2-3', source: '2', target: '3' },
  { id: 'e3-4', source: '3', target: '4' },
  { id: 'e4-5', source: '4', target: '5' },
]

export default function WorkflowEditor() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)

  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges]
  )

  const onNodeClick = (_: React.MouseEvent, node: Node) => {
    setSelectedNode(node)
  }

  const addNode = (type: string) => {
    const newNode: Node = {
      id: `${nodes.length + 1}`,
      type,
      position: { x: 100 + Math.random() * 200, y: 200 + Math.random() * 200 },
      data: { label: `New ${type}` },
    }
    setNodes((nds) => [...nds, newNode])
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4">
      {/* Sidebar */}
      <div className="w-64 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Node Palette</h2>
        
        <div className="space-y-2">
          <button
            onClick={() => addNode('structure_input')}
            className="w-full flex items-center px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            Input Node
          </button>
          
          <button
            onClick={() => addNode('dft_calculation')}
            className="w-full flex items-center px-4 py-2 border border-blue-300 rounded-md text-sm font-medium text-blue-700 bg-blue-50 hover:bg-blue-100"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            DFT Calculation
          </button>
          
          <button
            onClick={() => addNode('md_simulation')}
            className="w-full flex items-center px-4 py-2 border border-green-300 rounded-md text-sm font-medium text-green-700 bg-green-50 hover:bg-green-100"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            MD Simulation
          </button>
          
          <button
            onClick={() => addNode('ml_training')}
            className="w-full flex items-center px-4 py-2 border border-purple-300 rounded-md text-sm font-medium text-purple-700 bg-purple-50 hover:bg-purple-100"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            ML Training
          </button>
          
          <button
            onClick={() => addNode('analysis')}
            className="w-full flex items-center px-4 py-2 border border-orange-300 rounded-md text-sm font-medium text-orange-700 bg-orange-50 hover:bg-orange-100"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            Analysis
          </button>
        </div>

        {selectedNode && (
          <div className="mt-8 pt-8 border-t border-gray-200">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Node Configuration</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-700">Label</label>
                <input
                  type="text"
                  value={selectedNode.data.label}
                  onChange={(e) => {
                    setNodes((nds) =>
                      nds.map((n) =>
                        n.id === selectedNode.id
                          ? { ...n, data: { ...n.data, label: e.target.value } }
                          : n
                      )
                    )
                  }}
                  className="mt-1 block w-full rounded-md border-gray-300 text-sm"
                />
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700">Type</label>
                <p className="mt-1 text-sm text-gray-500">{selectedNode.type}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Workflow Canvas */}
      <div className="flex-1 card overflow-hidden">
        <div className="h-14 border-b border-gray-200 px-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-700">Active Learning Workflow</span>
            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
              Draft
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            <button className="btn-secondary text-sm">
              <SaveIcon className="h-4 w-4 mr-1" />
              Save
            </button>
            <button className="btn-primary text-sm">
              <PlayIcon className="h-4 w-4 mr-1" />
              Execute
            </button>
          </div>
        </div>

        <div className="h-[calc(100%-3.5rem)]">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
          >
            <Background />
            <Controls />
            <MiniMap />
          </ReactFlow>
        </div>
      </div>
    </div>
  )
}
