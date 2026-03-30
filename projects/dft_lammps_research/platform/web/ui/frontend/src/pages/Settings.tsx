import React from 'react'
import { CogIcon, BellIcon, ShieldCheckIcon, UserCircleIcon } from '@heroicons/react/24/outline'

export default function Settings() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Settings</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* General Settings */}
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center">
                <CogIcon className="h-6 w-6 text-gray-400 mr-3" />
                <h2 className="text-lg font-medium">General Settings</h2>
              </div>
            </div>
            
            <div className="p-6 space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Default Project Directory</label>
                <input type="text" defaultValue="./workdir" className="form-input" />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Theme</label>
                <select className="form-input">
                  <option>Light</option>
                  <option>Dark</option>
                  <option>Auto</option>
                </select>
              </div>
              
              <div className="flex items-center">
                <input type="checkbox" defaultChecked className="rounded border-gray-300" />
                <span className="ml-2 text-sm text-gray-700">Enable auto-refresh</span>
              </div>
            </div>
          </div>

          {/* Notifications */}
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center">
                <BellIcon className="h-6 w-6 text-gray-400 mr-3" />
                <h2 className="text-lg font-medium">Notifications</h2>
              </div>
            </div>
            
            <div className="p-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700">Task completion alerts</span>
                <input type="checkbox" defaultChecked className="rounded border-gray-300" />
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700">Workflow failure alerts</span>
                <input type="checkbox" defaultChecked className="rounded border-gray-300" />
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700">Email notifications</span>
                <input type="checkbox" className="rounded border-gray-300" />
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          {/* User Profile */}
          <div className="card">
            <div className="p-6 text-center">
              <UserCircleIcon className="h-20 w-20 text-gray-300 mx-auto" />
              <h3 className="mt-4 text-lg font-medium">Researcher</h3>
              <p className="text-sm text-gray-500">researcher@example.com</p>
              <p className="mt-2 inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-primary-100 text-primary-800">
                Researcher
              </p>
            </div>
          </div>

          {/* API Access */}
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center">
                <ShieldCheckIcon className="h-6 w-6 text-gray-400 mr-3" />
                <h2 className="text-lg font-medium">API Access</h2>
              </div>
            </div>
            
            <div className="p-6">
              <p className="text-sm text-gray-600 mb-4">Use this API key to access the platform programmatically.</p>
              
              <div className="flex gap-2">
                <input
                  type="password"
                  value="sk-xxxxxxxxxxxxxxxx"
                  readOnly
                  className="form-input flex-1"
                />
                <button className="btn-secondary">Show</button>
                <button className="btn-secondary">Regenerate</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
