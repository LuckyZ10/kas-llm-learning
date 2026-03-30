import React from 'react'

interface StatsCardProps {
  name: string
  value: string | number
  icon: React.ComponentType<{ className?: string }>
  change?: string
  changeType?: 'positive' | 'negative' | 'neutral'
}

export default function StatsCard({ name, value, icon: Icon, change, changeType = 'neutral' }: StatsCardProps) {
  return (
    <div className="card p-6 hover:shadow-md transition-shadow">
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <div className="h-12 w-12 rounded-lg bg-primary-100 flex items-center justify-center">
            <Icon className="h-6 w-6 text-primary-600" aria-hidden="true" />
          </div>
        </div>
        
        <div className="ml-4 flex-1">
          <p className="text-sm font-medium text-gray-500 truncate">{name}</p>
          
          <div className="flex items-baseline">
            <p className="text-2xl font-semibold text-gray-900">{value}</p>
            
            {change && (
              <p className={`ml-2 text-sm font-medium ${
                changeType === 'positive' ? 'text-green-600' : 
                changeType === 'negative' ? 'text-red-600' : 'text-gray-500'
              }`}>
                {change}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
