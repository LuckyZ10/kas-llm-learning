import { useAppStore } from '../../store';
import { DropZone } from './DropZone';
import { FridgeItemCard } from './FridgeItemCard';

export function FridgeBoard() {
  const { selectedTask, currentFridgeItems } = useAppStore();
  
  // Filter items for current task
  const taskItems = currentFridgeItems.filter(item => 
    item.filePath.includes(selectedTask?.fridgePath || '')
  );
  
  return (
    <div className="h-full bg-gray-50 relative overflow-hidden flex flex-col">
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-gray-400 to-transparent" />
      
      {/* Header */}
      <div className="relative z-10 p-4 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-600">
          {selectedTask ? `🧲 ${selectedTask.title}` : '冰箱贴板'}
        </h3>
        {selectedTask && (
          <p className="text-xs text-gray-400 mt-1">
            {taskItems.length} 个文件
          </p>
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