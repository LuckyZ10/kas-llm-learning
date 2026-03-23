import { describe, it, expect, vi, beforeAll } from 'vitest';
import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { DateSelector } from '../DateSelector';

// Mock scrollIntoView
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

// Mock zustand store
vi.mock('../../store', () => ({
  useAppStore: () => ({
    selectedDate: new Date('2024-03-25'),
    selectedMonth: 2,
    setDate: vi.fn()
  })
}));

describe('DateSelector', () => {
  it('should render month and day rows', () => {
    render(<DateSelector />);
    
    expect(screen.getByTestId('month-row')).toBeInTheDocument();
    expect(screen.getByTestId('day-row')).toBeInTheDocument();
  });
});