import { useAppStore } from '../../store';
import { TaskList } from '../TaskList/TaskList';
import { useState } from 'react';
import { Task } from '../../types';

export function RightPanel() {
  const { selectedTask, selectTask } = useAppStore();
  const [tasks, setTasks] = useState<Task[]>([
    { 
      id: '1', 
      title: '完成设计文档', 
      completed: false, 
      createdAt: new Date(), 
      updatedAt: new Date(), 
      fridgePath: '/fridge/2024-03-25/task-1'
    },
    { 
      id: '2', 
      title: '实现基础布局', 
      completed: true, 
      createdAt: new Date(), 
      updatedAt: new Date(), 
      fridgePath: '/fridge/2024-03-25/task-2'
    },
  ]);
  
  const handleTaskToggle = (id: string) => {
    setTasks(tasks.map(t => 
      t.id === id ? { ...t, completed: !t.completed } : t
    ));
  };
  
  return (
    <div className="w-1/2 bg-white p-4 flex flex-col">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">
        {new Date().toLocaleDateString('zh-CN', { month: 'long', day: 'numeric' })}
      </h2>
      
      <div className="flex-1 overflow-y-auto">
        <TaskList 
          tasks={tasks}
          selectedTaskId={selectedTask?.id}
          onTaskSelect={selectTask}
          onTaskToggle={handleTaskToggle}
        />
      </div>
      
      {/* Add Task Input */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <input 
          type="text" 
          placeholder="+ 添加新任务..."
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-400"
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              const input = e.target as HTMLInputElement;
              if (input.value.trim()) {
                const newTask: Task = {
                  id: Date.now().toString(),
                  title: input.value.trim(),
                  completed: false,
                  createdAt: new Date(),
                  updatedAt: new Date(),
                  fridgePath: `/fridge/${new Date().toISOString().split('T')[0]}/task-${Date.now()}`
                };
                setTasks([...tasks, newTask]);
                input.value = '';
              }
            }
          }}
        />
      </div>
    </div>
  );
}