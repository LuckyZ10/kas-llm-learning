import { Task } from '../../types';
import { Check } from 'lucide-react';

interface TaskItemProps {
  task: Task;
  isSelected: boolean;
  onSelect: (task: Task) => void;
  onToggle: (id: string) => void;
}

export function TaskItem({ task, isSelected, onSelect, onToggle }: TaskItemProps) {
  return (
    <div 
      onClick={() => onSelect(task)}
      className={`
        group flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all
        ${isSelected 
          ? 'bg-blue-50 border border-blue-200 shadow-sm' 
          : 'hover:bg-gray-50 border border-transparent'
        }
      `}
    >
      <button
        onClick={(e) => {
          e.stopPropagation();
          onToggle(task.id);
        }}
        className={`
          w-5 h-5 rounded border-2 flex items-center justify-center transition-colors
          ${task.completed 
            ? 'bg-green-500 border-green-500' 
            : 'border-gray-300 hover:border-blue-400'
          }
        `}
      >
        {task.completed && <Check className="w-3 h-3 text-white" />}
      </button>
      
      <div className="flex-1">
        <span className={`
          text-sm font-medium
          ${task.completed ? 'text-gray-400 line-through' : 'text-gray-800'}
        `}>
          {task.title}
        </span>
        {task.description && (
          <p className="text-xs text-gray-500 mt-0.5 line-clamp-1">{task.description}</p>
        )}
      </div>
    </div>
  );
}