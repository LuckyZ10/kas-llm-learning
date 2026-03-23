import { useState } from 'react';
import { FridgeBoard } from '../FridgeBoard/FridgeBoard';
import { AIChat } from '../AIChat/AIChat';

export function LeftPanel() {
  const [chatHeight, setChatHeight] = useState(250); // Default height in px
  
  return (
    <div className="w-1/2 flex flex-col border-r border-gray-200">
      {/* Fridge Board - takes remaining space */}
      <div className="flex-1 overflow-hidden">
        <FridgeBoard />
      </div>
      
      {/* Resize Handle */}
      <div 
        className="h-2 bg-gray-200 cursor-row-resize hover:bg-blue-300 transition-colors"
        onMouseDown={(e) => {
          const startY = e.clientY;
          const startHeight = chatHeight;
          
          const handleMouseMove = (e: MouseEvent) => {
            const delta = startY - e.clientY;
            setChatHeight(Math.max(150, Math.min(400, startHeight + delta)));
          };
          
          const handleMouseUp = () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
          };
          
          document.addEventListener('mousemove', handleMouseMove);
          document.addEventListener('mouseup', handleMouseUp);
        }}
      />
      
      {/* AI Chat */}
      <div style={{ height: chatHeight }} className="border-t border-gray-200">
        <AIChat />
      </div>
    </div>
  );
}