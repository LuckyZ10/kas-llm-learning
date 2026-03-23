import { Task } from '../../types';
import { TaskItem } from './TaskItem';

interface TaskListProps {
  tasks: Task[];
  selectedTaskId?: string;
  onTaskSelect?: (task: Task) => void;
  onTaskToggle?: (id: string) => void;
}

export function TaskList({ 
  tasks, 
  selectedTaskId, 
  onTaskSelect, 
  onTaskToggle 
}: TaskListProps) {
  return (
    <div className="space-y-1">
      {tasks.map((task) => (
        <TaskItem
          key={task.id}
          task={task}
          isSelected={task.id === selectedTaskId}
          onSelect={onTaskSelect || (() => {})}
          onToggle={onTaskToggle || (() => {})}
        />
      ))}
    </div>
  );
}