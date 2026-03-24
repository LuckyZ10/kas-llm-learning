import { useAppStore } from '../../store';
import { DropZone } from './DropZone';

export function FridgeBoard() {
  const { selectedTask } = useAppStore();
  
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
            拖拽文件到下方区域
          </p>
        )}
      </div>
      
      {/* Board Area */}
      <div className="relative flex-1">
        <DropZone />
      </div>
    </div>
  );
}