import { describe, it, expect, vi } from 'vitest';
import '@testing-library/jest-dom';
import { render, screen, fireEvent } from '@testing-library/react';
import { TaskList } from '../TaskList';
import { Task } from '../../../types';

const mockTasks: Task[] = [
  { id: '1', title: 'Task 1', completed: false, createdAt: new Date(), updatedAt: new Date(), fridgePath: '/path/1' },
  { id: '2', title: 'Task 2', completed: true, createdAt: new Date(), updatedAt: new Date(), fridgePath: '/path/2' },
];

describe('TaskList', () => {
  it('should render all tasks', () => {
    render(<TaskList tasks={mockTasks} />);
    
    expect(screen.getByText('Task 1')).toBeInTheDocument();
    expect(screen.getByText('Task 2')).toBeInTheDocument();
  });

  it('should call onTaskSelect when task is clicked', () => {
    const onSelect = vi.fn();
    render(<TaskList tasks={mockTasks} onTaskSelect={onSelect} />);
    
    fireEvent.click(screen.getByText('Task 1'));
    expect(onSelect).toHaveBeenCalledWith(mockTasks[0]);
  });
});