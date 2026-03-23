import { ReactNode } from 'react';
import { DateSelector } from '../DateSelector/DateSelector';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Date Selectors */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <DateSelector />
      </div>
      
      {/* Main Content */}
      <div data-testid="main-content" className="flex-1 overflow-hidden">
        {children}
      </div>
    </div>
  );
}