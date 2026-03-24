import { useState } from 'react';
import { useAppStore } from '../../store';
import { DropZone } from './DropZone';
import { FridgeItemCard } from './FridgeItemCard';
import { TagColor, TAG_COLORS } from '../../types';

export function FridgeBoard() {
  const { selectedTask, currentFridgeItems } = useAppStore();
  const [tagFilter, setTagFilter] = useState<TagColor | null>(null);
  
  // Filter items for current task
  let taskItems = currentFridgeItems.filter(item => 
    item.filePath.includes(selectedTask?.fridgePath || '')
  );
  
  // Apply tag filter
  if (tagFilter) {
    taskItems = taskItems.filter(item => item.tags.color === tagFilter);
  }
  
  return (
    <div className="h-full bg-gray-50 relative overflow-hidden flex flex-col">
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-gray-400 to-transparent" />
      
      {/* Header */}
      <div className="relative z-10 p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-600">
            {selectedTask ? `🧲 ${selectedTask.title}` : '冰箱贴板'}
          </h3>
          {selectedTask && (
            <p className="text-xs text-gray-400">
              {taskItems.length} 个文件
            </p>
          )}
        </div>
        
        {/* Tag Filter */}
        {selectedTask && currentFridgeItems.length > 0 && (
          <div className="flex gap-1 mt-2">
            <button
              onClick={() => setTagFilter(null)}
              className={`px-2 py-0.5 text-xs rounded-full transition-colors ${
                !tagFilter ? 'bg-gray-800 text-white' : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
              }`}
            >
              全部
            </button>
            {(['red', 'blue', 'green', 'yellow', 'purple', 'orange'] as TagColor[]).map(c => (
              <button
                key={c}
                onClick={() => setTagFilter(c)}
                className={`w-5 h-5 rounded-full border-2 transition-all ${
                  tagFilter === c ? 'border-gray-800 scale-110' : 'border-transparent hover:scale-105'
                }`}
                style={{ backgroundColor: TAG_COLORS[c].bg }}
                title={TAG_COLORS[c].label}
              />
            ))}
          </div>
        )}
      </div>
      
      {/* Board Area */}
      <div className="relative flex-1">
        <DropZone />
        
        {/* Render fridge items */}
        {taskItems.map((item) => (
          <FridgeItemCard key={item.id} item={item} />
        ))}
      </div>
    </div>
  );
}