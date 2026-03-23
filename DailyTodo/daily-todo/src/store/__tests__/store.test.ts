import { describe, it, expect, beforeEach } from 'vitest';
import '@testing-library/jest-dom';
import { useAppStore } from '../index';

describe('App Store', () => {
  beforeEach(() => {
    useAppStore.setState({
      selectedDate: new Date('2024-03-25'),
      selectedMonth: 2,
      currentDaily: null,
      selectedTask: null,
      currentFridgeItems: [],
      chatMessages: [],
      isChatLoading: false,
    });
  });

  it('should set date correctly', () => {
    const store = useAppStore.getState();
    const newDate = new Date('2024-04-01');
    
    store.setDate(newDate);
    
    expect(useAppStore.getState().selectedDate).toEqual(newDate);
    expect(useAppStore.getState().selectedMonth).toBe(3);
  });

  it('should select task', () => {
    const store = useAppStore.getState();
    const task = {
      id: 'task-1',
      title: 'Test Task',
      completed: false,
      createdAt: new Date(),
      updatedAt: new Date(),
      fridgePath: '/path/to/fridge'
    };
    
    store.selectTask(task);
    
    expect(useAppStore.getState().selectedTask).toEqual(task);
  });
});