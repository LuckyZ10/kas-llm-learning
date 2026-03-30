import React, { useState, useRef, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, Grid } from '@react-three/drei'
import * as THREE from 'three'

// Mock structure data - in real app this would come from API
const mockAtoms = [
  { element: 'Li', x: 0, y: 0, z: 0, color: '#cc00ff' },
  { element: 'Li', x: 2, y: 0, z: 0, color: '#cc00ff' },
  { element: 'Mn', x: 1, y: 1.5, z: 0.5, color: '#ff0000' },
  { element: 'Mn', x: 1, y: -1.5, z: -0.5, color: '#ff0000' },
  { element: 'O', x: 1, y: 0, z: 2, color: '#ff4400' },
  { element: 'O', x: 1, y: 0, z: -2, color: '#ff4400' },
  { element: 'O', x: -0.5, y: 0.8, z: 0, color: '#ff4400' },
  { element: 'O', x: 2.5, y: -0.8, z: 0, color: '#ff4400' },
]

const elementSizes: Record<string, number> = {
  Li: 0.3,
  Mn: 0.5,
  O: 0.25,
}

function Atom({ position, color, element }: { position: [number, number, number]; color: string; element: string }) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[elementSizes[element] || 0.3, 32, 32]} />
      <meshStandardMaterial 
        color={color} 
        roughness={0.3}
        metalness={0.2}
      />
    </mesh>
  )
}

function Structure() {
  const groupRef = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (groupRef.current) {
      // Slow rotation
      groupRef.current.rotation.y += 0.002
    }
  })
  
  return (
    <group ref={groupRef}>
      {mockAtoms.map((atom, i) => (
        <Atom
          key={i}
          position={[atom.x, atom.y, atom.z]}
          color={atom.color}
          element={atom.element}
        />
      ))}
      
      {/* Unit cell box */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(4, 4, 4)]} />
        <lineBasicMaterial color="#444" />
      </lineSegments>
    </group>
  )
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      <Suspense fallback={null}>
        <Structure />
      </Suspense>
      <Grid args={[20, 20]} cellSize={1} cellThickness={0.5} cellColor="#333" />
      <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
    <//>
  )
}

export default function StructureViewer() {
  const [selectedStructure, setSelectedStructure] = useState('LiMnO2-001')
  const [viewMode, setViewMode] = useState<'ball-stick' | 'space-fill' | 'wireframe'>('ball-stick')
  const [showUnitCell, setShowUnitCell] = useState(true)

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4">
      {/* Sidebar */}
      <div className="w-72 bg-white rounded-lg shadow-sm border border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">Structure Library</h2>
        </div>
        
        <div className="flex-1 overflow-auto p-4">
          <div className="space-y-2">
            {['LiMnO2-001', 'LiCoO2-001', 'LiFePO4-001', 'Si-anode-001'].map((id) => (
              <button
                key={id}
                onClick={() => setSelectedStructure(id)}
                className={`w-full text-left px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                  selectedStructure === id
                    ? 'bg-primary-50 text-primary-700 border border-primary-200'
                    : 'text-gray-700 hover:bg-gray-50 border border-transparent'
                }`}
              >
                <div>{id}</div>
                <div className="text-xs text-gray-500">Battery cathode material</div>
              </button>
            ))}
          </div>
        </div>

        <div className="p-4 border-t border-gray-200 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">View Mode</label>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as any)}
              className="form-input"
            >
              <option value="ball-stick">Ball and Stick</option>
              <option value="space-fill">Space Fill</option>
              <option value="wireframe">Wireframe</option>
            </select>
          </div>
          
          <div>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showUnitCell}
                onChange={(e) => setShowUnitCell(e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="ml-2 text-sm text-gray-700">Show Unit Cell</span>
            </label>
          </div>
        </div>
      </div>

      {/* 3D Viewer */}
      <div className="flex-1 card overflow-hidden flex flex-col">
        <div className="h-14 border-b border-gray-200 px-4 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">{selectedStructure}</h3>
            <p className="text-xs text-gray-500">LiMnO₂ - P6₃/mmc - 8 atoms</p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 text-sm">
              <div className="flex items-center">
                <span className="w-3 h-3 rounded-full bg-purple-500 mr-1"></span>
                Li
              </div>
              <div className="flex items-center">
                <span className="w-3 h-3 rounded-full bg-red-500 mr-1"></span>
                Mn
              </div>
              <div className="flex items-center">
                <span className="w-3 h-3 rounded-full bg-orange-500 mr-1"></span>
                O
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 bg-gray-900">
          <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
            <Scene />
          </Canvas>
        </div>

        <div className="h-48 border-t border-gray-200 p-4">
          <div className="grid grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-xs text-gray-500 uppercase">Formation Energy</p>
              <p className="text-lg font-semibold">-2.34 eV/atom</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-500 uppercase">Band Gap</p>
              <p className="text-lg font-semibold">2.15 eV</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-500 uppercase">Ionic Conductivity</p>
              <p className="text-lg font-semibold">1.2×10⁻³ S/cm</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-500 uppercase">Volume</p>
              <p className="text-lg font-semibold">45.2 Å³</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
