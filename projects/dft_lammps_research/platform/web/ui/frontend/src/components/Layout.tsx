import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  HomeIcon,
  FolderIcon,
  BeakerIcon,
  QueueListIcon,
  ChartBarIcon,
  MagnifyingGlassIcon,
  CubeTransparentIcon,
  DocumentTextIcon,
  Cog6ToothIcon,
  BellIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline'

const navigation = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'Projects', href: '/projects', icon: FolderIcon },
  { name: 'Workflows', href: '/workflows', icon: BeakerIcon },
  { name: 'Tasks', href: '/tasks', icon: QueueListIcon },
  { name: 'Monitoring', href: '/monitoring', icon: ChartBarIcon },
  { name: 'Screening', href: '/screening', icon: MagnifyingGlassIcon },
  { name: 'Structures', href: '/structures', icon: CubeTransparentIcon },
  { name: 'Reports', href: '/reports', icon: DocumentTextIcon },
]

const bottomNavigation = [
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
]

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-gray-900">
        <div className="flex h-16 items-center px-6">
          <BeakerIcon className="h-8 w-8 text-primary-400" />
          <span className="ml-3 text-xl font-bold text-white">DFT+LAMMPS</span>
        </div>

        <nav className="mt-8 flex flex-1 flex-col px-4">
          <ul role="list" className="flex flex-1 flex-col gap-y-7">
            <li>
              <ul role="list" className="-mx-2 space-y-1">
                {navigation.map((item) => (
                  <li key={item.name}>
                    <Link
                      to={item.href}
                      className={`
                        group flex gap-x-3 rounded-md p-2 text-sm font-semibold leading-6
                        ${location.pathname === item.href || location.pathname.startsWith(item.href + '/')
                          ? 'bg-primary-600 text-white'
                          : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                        }
                      `}
                    >
                      <item.icon className="h-6 w-6 shrink-0" aria-hidden="true" />
                      {item.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </li>

            <li className="mt-auto">
              <ul role="list" className="-mx-2 space-y-1">
                {bottomNavigation.map((item) => (
                  <li key={item.name}>
                    <Link
                      to={item.href}
                      className={`
                        group flex gap-x-3 rounded-md p-2 text-sm font-semibold leading-6
                        ${location.pathname === item.href
                          ? 'bg-primary-600 text-white'
                          : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                        }
                      `}
                    >
                      <item.icon className="h-6 w-6 shrink-0" aria-hidden="true" />
                      {item.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </li>
          </ul>
        </nav>
      </div>

      {/* Main content */}
      <div className="pl-64">
        {/* Top header */}
        <div className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-gray-200 bg-white px-8">
          <h1 className="text-lg font-semibold text-gray-900">
            {navigation.find(n => n.href === location.pathname)?.name || 'Research Platform'}
          </h1>
          
          <div className="flex items-center gap-4">
            <button className="p-2 text-gray-400 hover:text-gray-500">
              <BellIcon className="h-6 w-6" />
            </button>
            <button className="flex items-center gap-2 p-2 text-gray-400 hover:text-gray-500">
              <UserCircleIcon className="h-6 w-6" />
              <span className="text-sm font-medium text-gray-700">Researcher</span>
            </button>
          </div>
        </div>

        {/* Page content */}
        <main className="py-8 px-8">
          {children}
        </main>
      </div>
    </div>
  )
}
