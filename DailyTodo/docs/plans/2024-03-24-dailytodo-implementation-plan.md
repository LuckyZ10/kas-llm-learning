# DailyTodo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use supaconductor:executing-plans to implement this plan task-by-task.

**Goal:** Build a desktop DailyTodo app with Tauri + React, featuring a draggable fridge board with free positioning and tag system.

**Architecture:** Three-panel layout with horizontal date selectors at top. Left panel contains the fridge board (core feature) and AI chat. Right panel shows daily task records. State managed by Zustand, file storage via Tauri FS.

**Tech Stack:** Tauri 2.0, React 18, TypeScript, TailwindCSS, shadcn/ui, Zustand, Dexie (IndexedDB), React DnD

---

## Phase 1: Project Setup and Foundation

### Task 1: Initialize Tauri + React Project

**Files:**
- Create: Entire project structure via `npm create tauri-app@latest`

**Step 1: Create project**

```bash
npm create tauri-app@latest daily-todo -- --template react-ts --manager npm
cd daily-todo
```

**Step 2: Install dependencies**

```bash
npm install zustand @tauri-apps/api dexie react-dnd react-dnd-html5-backend lucide-react
npm install -D @types/node
```

**Step 3: Setup TailwindCSS**

```bash
npx tailwindcss init -p
```

Modify `tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        tag: {
          red: { bg: '#FEE2E2', text: '#DC2626', border: '#FECACA' },
          blue: { bg: '#DBEAFE', text: '#2563EB', border: '#BFDBFE' },
          green: { bg: '#D1FAE5', text: '#059669', border: '#A7F3D0' },
          yellow: { bg: '#FEF3C7', text: '#D97706', border: '#FDE68A' },
          purple: { bg: '#F3E8FF', text: '#7C3AED', border: '#E9D5FF' },
          orange: { bg: '#FFEDD5', text: '#EA580C', border: '#FED7AA' },
        }
      }
    },
  },
  plugins: [],
}
```

Add to `src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Step 4: Verify setup**

Run: `npm run tauri dev`
Expected: App launches with default Tauri + React template

**Step 5: Commit**

```bash
git add .
git commit -m "chore: initialize tauri + react project with tailwindcss"
```

---

### Task 2: Create Type Definitions

**Files:**
- Create: `src/types/index.ts`

**Step 1: Write type definitions**

```typescript
// src/types/index.ts

export type TagColor = 'red' | 'blue' | 'green' | 'yellow' | 'purple' | 'orange' | 'none';

export interface Task {
  id: string;
  title: string;
  description?: string;
  completed: boolean;
  createdAt: Date;
  updatedAt: Date;
  fridgePath: string;
}

export interface DailyRecord {
  date: string; // YYYY-MM-DD
  tasks: Task[];
  aiContext: string;
  lastModified: Date;
}

export interface Position {
  x: number;
  y: number;
  rotation: number;
  zIndex: number;
}

export interface Tags {
  text?: string;
  color: TagColor;
}

export interface VisualStyle {
  magnetColor: string;
  shadow: boolean;
  scale: number;
}

export interface FridgeItem {
  id: string;
  fileName: string;
  filePath: string;
  mimeType: string;
  size: number;
  previewType: 'image' | 'document' | 'other';
  createdAt: Date;
  position: Position;
  tags: Tags;
  visual: VisualStyle;
}

export const TAG_COLORS: Record<TagColor, { label: string; bg: string; text: string; border: string }> = {
  red: { label: '紧急', bg: '#FEE2E2', text: '#DC2626', border: '#FECACA' },
  blue: { label: '参考', bg: '#DBEAFE', text: '#2563EB', border: '#BFDBFE' },
  green: { label: '已完成', bg: '#D1FAE5', text: '#059669', border: '#A7F3D0' },
  yellow: { label: '待处理', bg: '#FEF3C7', text: '#D97706', border: '#FDE68A' },
  purple: { label: '想法', bg: '#F3E8FF', text: '#7C3AED', border: '#E9D5FF' },
  orange: { label: '进行中', bg: '#FFEDD5', text: '#EA580C', border: '#FED7AA' },
  none: { label: '无标签', bg: '#F3F4F6', text: '#6B7280', border: '#E5E7EB' }
};

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
```

**Step 2: Commit**

```bash
git add src/types/index.ts
git commit -m "feat: add core type definitions"
```

---

### Task 3: Setup Zustand Store

**Files:**
- Create: `src/store/index.ts`
- Create: `src/store/__tests__/store.test.ts`

**Step 1: Write test**

```typescript
// src/store/__tests__/store.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
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
```

**Step 2: Install vitest**

```bash
npm install -D vitest @testing-library/react
```

Add to `package.json`:
```json
"scripts": {
  "test": "vitest"
}
```

**Step 3: Run test (expect FAIL)**

Run: `npm test -- src/store/__tests__/store.test.ts`
Expected: FAIL - "useAppStore" not defined

**Step 4: Implement store**

```typescript
// src/store/index.ts
import { create } from 'zustand';
import { Task, DailyRecord, FridgeItem, Message } from '../types';

interface AppState {
  // Date selection
  selectedDate: Date;
  selectedMonth: number;
  
  // Daily data
  currentDaily: DailyRecord | null;
  
  // Task selection
  selectedTask: Task | null;
  
  // Fridge board
  currentFridgeItems: FridgeItem[];
  
  // AI chat
  chatMessages: Message[];
  isChatLoading: boolean;
  
  // Actions
  setDate: (date: Date) => void;
  loadDaily: (date: string) => Promise<void>;
  selectTask: (task: Task | null) => void;
  setFridgeItems: (items: FridgeItem[]) => void;
  addFridgeItem: (item: FridgeItem) => void;
  updateFridgeItemPosition: (id: string, x: number, y: number) => void;
  updateFridgeItemTags: (id: string, tags: { text?: string; color: string }) => void;
}

export const useAppStore = create<AppState>((set) => ({
  selectedDate: new Date(),
  selectedMonth: new Date().getMonth(),
  currentDaily: null,
  selectedTask: null,
  currentFridgeItems: [],
  chatMessages: [],
  isChatLoading: false,
  
  setDate: (date) => set({ 
    selectedDate: date, 
    selectedMonth: date.getMonth() 
  }),
  
  loadDaily: async (date) => {
    // TODO: Implement loading from storage
    console.log('Loading daily for:', date);
  },
  
  selectTask: (task) => set({ selectedTask: task }),
  
  setFridgeItems: (items) => set({ currentFridgeItems: items }),
  
  addFridgeItem: (item) => set((state) => ({
    currentFridgeItems: [...state.currentFridgeItems, item]
  })),
  
  updateFridgeItemPosition: (id, x, y) => set((state) => ({
    currentFridgeItems: state.currentFridgeItems.map(item =>
      item.id === id ? { ...item, position: { ...item.position, x, y } } : item
    )
  })),
  
  updateFridgeItemTags: (id, tags) => set((state) => ({
    currentFridgeItems: state.currentFridgeItems.map(item =>
      item.id === id ? { 
        ...item, 
        tags: { ...item.tags, ...tags },
        visual: { 
          ...item.visual, 
          magnetColor: tags.color && tags.color !== 'none' 
            ? TAG_COLORS[tags.color as TagColor].border 
            : '#E5E7EB' 
        }
      } : item
    )
  })),
}));
```

**Step 5: Run test (expect PASS)**

Run: `npm test -- src/store/__tests__/store.test.ts`
Expected: PASS

**Step 6: Commit**

```bash
git add src/store/
git commit -m "feat: add zustand store with basic state management"
```

---

## Phase 2: UI Layout Components

### Task 4: Create Base Layout

**Files:**
- Create: `src/components/Layout/Layout.tsx`
- Create: `src/components/Layout/__tests__/Layout.test.tsx`
- Modify: `src/App.tsx`

**Step 1: Write test**

```typescript
// src/components/Layout/__tests__/Layout.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Layout } from '../Layout';

describe('Layout', () => {
  it('should render header, date selectors, and main content', () => {
    render(<Layout>Test Content</Layout>);
    
    expect(screen.getByTestId('date-selector')).toBeInTheDocument();
    expect(screen.getByTestId('main-content')).toBeInTheDocument();
  });
});
```

**Step 2: Run test (expect FAIL)**

Run: `npm test -- src/components/Layout/__tests__/Layout.test.tsx`
Expected: FAIL

**Step 3: Implement Layout component**

```tsx
// src/components/Layout/Layout.tsx
import { ReactNode } from 'react';
import { DateSelector } from '../DateSelector/DateSelector';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Date Selectors */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <DateSelector />
      </div>
      
      {/* Main Content */}
      <div data-testid="main-content" className="flex-1 overflow-hidden">
        {children}
      </div>
    </div>
  );
}
```

**Step 4: Update App.tsx**

```tsx
// src/App.tsx
import { Layout } from './components/Layout/Layout';
import { MainContent } from './components/MainContent/MainContent';

function App() {
  return (
    <Layout>
      <MainContent />
    </Layout>
  );
}

export default App;
```

**Step 5: Run test (expect PASS)**

Run: `npm test -- src/components/Layout/__tests__/Layout.test.tsx`
Expected: PASS

**Step 6: Commit**

```bash
git add src/components/Layout/ src/App.tsx
git commit -m "feat: add base layout component"
```

---

### Task 5: Create DateSelector Component

**Files:**
- Create: `src/components/DateSelector/DateSelector.tsx`
- Create: `src/components/DateSelector/MonthRow.tsx`
- Create: `src/components/DateSelector/DayRow.tsx`
- Create: `src/components/DateSelector/__tests__/DateSelector.test.tsx`

**Step 1: Write test**

```typescript
// src/components/DateSelector/__tests__/DateSelector.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DateSelector } from '../DateSelector';

// Mock zustand store
vi.mock('../../../store', () => ({
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
```

**Step 2: Run test (expect FAIL)**

**Step 3: Implement components**

```tsx
// src/components/DateSelector/DateSelector.tsx
import { MonthRow } from './MonthRow';
import { DayRow } from './DayRow';

export function DateSelector() {
  return (
    <div data-testid="date-selector" className="py-3 px-4 space-y-2">
      <MonthRow />
      <DayRow />
    </div>
  );
}
```

```tsx
// src/components/DateSelector/MonthRow.tsx
import { useRef, useEffect } from 'react';
import { useAppStore } from '../../store';

const MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];

export function MonthRow() {
  const { selectedMonth, setDate, selectedDate } = useAppStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    // Scroll selected month into view
    if (scrollRef.current) {
      const selectedEl = scrollRef.current.children[selectedMonth] as HTMLElement;
      if (selectedEl) {
        selectedEl.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      }
    }
  }, [selectedMonth]);
  
  const handleMonthClick = (monthIndex: number) => {
    const newDate = new Date(selectedDate);
    newDate.setMonth(monthIndex);
    setDate(newDate);
  };
  
  return (
    <div 
      ref={scrollRef}
      data-testid="month-row"
      className="flex gap-2 overflow-x-auto scrollbar-hide scroll-smooth snap-x snap-mandatory"
    >
      {MONTHS.map((month, index) => (
        <button
          key={month}
          onClick={() => handleMonthClick(index)}
          className={`
            flex-shrink-0 px-4 py-2 rounded-lg font-medium text-sm transition-all
            snap-center
            ${selectedMonth === index 
              ? 'bg-blue-500 text-white shadow-md' 
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }
          `}
        >
          {month}
        </button>
      ))}
    </div>
  );
}
```

```tsx
// src/components/DateSelector/DayRow.tsx
import { useRef, useEffect, useMemo } from 'react';
import { useAppStore } from '../../store';

export function DayRow() {
  const { selectedDate, setDate } = useAppStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  
  const days = useMemo(() => {
    const year = selectedDate.getFullYear();
    const month = selectedDate.getMonth();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    
    return Array.from({ length: daysInMonth }, (_, i) => i + 1);
  }, [selectedDate]);
  
  useEffect(() => {
    if (scrollRef.current) {
      const selectedDay = selectedDate.getDate();
      const selectedEl = scrollRef.current.children[selectedDay - 1] as HTMLElement;
      if (selectedEl) {
        selectedEl.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      }
    }
  }, [selectedDate]);
  
  const handleDayClick = (day: number) => {
    const newDate = new Date(selectedDate);
    newDate.setDate(day);
    setDate(newDate);
  };
  
  const isSelected = (day: number) => selectedDate.getDate() === day;
  const isToday = (day: number) => {
    const today = new Date();
    return today.getDate() === day && 
           today.getMonth() === selectedDate.getMonth() && 
           today.getFullYear() === selectedDate.getFullYear();
  };
  
  return (
    <div 
      ref={scrollRef}
      data-testid="day-row"
      className="flex gap-1 overflow-x-auto scrollbar-hide scroll-smooth snap-x snap-mandatory"
    >
      {days.map((day) => (
        <button
          key={day}
          onClick={() => handleDayClick(day)}
          className={`
            flex-shrink-0 w-10 h-10 rounded-full font-medium text-sm transition-all
            snap-center flex items-center justify-center
            ${isSelected(day)
              ? 'bg-blue-500 text-white shadow-md scale-110'
              : isToday(day)
                ? 'bg-blue-100 text-blue-600 border-2 border-blue-300'
                : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-200'
            }
          `}
        >
          {day}
        </button>
      ))}
    </div>
  );
}
```

Add CSS for scrollbar-hide in `src/index.css`:
```css
.scrollbar-hide {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
.scrollbar-hide::-webkit-scrollbar {
  display: none;
}
```

**Step 4: Run test (expect PASS)**

Run: `npm test -- src/components/DateSelector/__tests__/DateSelector.test.tsx`
Expected: PASS

**Step 5: Verify visually**

Run: `npm run tauri dev`
Expected: Month and day rows visible, can click to select

**Step 6: Commit**

```bash
git add src/components/DateSelector/ src/index.css
git commit -m "feat: add date selector with month and day rows"
```

---

### Task 6: Create MainContent with Panels

**Files:**
- Create: `src/components/MainContent/MainContent.tsx`
- Create: `src/components/LeftPanel/LeftPanel.tsx`
- Create: `src/components/RightPanel/RightPanel.tsx`

**Step 1: Implement MainContent**

```tsx
// src/components/MainContent/MainContent.tsx
import { LeftPanel } from '../LeftPanel/LeftPanel';
import { RightPanel } from '../RightPanel/RightPanel';

export function MainContent() {
  return (
    <div className="flex h-full">
      <LeftPanel />
      <RightPanel />
    </div>
  );
}
```

**Step 2: Implement LeftPanel**

```tsx
// src/components/LeftPanel/LeftPanel.tsx
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
```

**Step 3: Implement RightPanel (placeholder)**

```tsx
// src/components/RightPanel/RightPanel.tsx
export function RightPanel() {
  return (
    <div className="w-1/2 bg-white p-4">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Daily Record</h2>
      <p className="text-gray-500">Select a task to view details...</p>
    </div>
  );
}
```

**Step 4: Create placeholder FridgeBoard and AIChat**

```tsx
// src/components/FridgeBoard/FridgeBoard.tsx
export function FridgeBoard() {
  return (
    <div className="h-full bg-gray-50 p-4 relative overflow-hidden">
      <div className="absolute inset-0 opacity-5 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-gray-400 to-transparent" />
      <h3 className="text-sm font-medium text-gray-600 mb-2">冰箱贴板</h3>
      <p className="text-gray-400 text-sm">点击任务查看文件...</p>
    </div>
  );
}
```

```tsx
// src/components/AIChat/AIChat.tsx
export function AIChat() {
  return (
    <div className="h-full bg-white p-4 flex flex-col">
      <h3 className="text-sm font-medium text-gray-600 mb-2">AI 助手</h3>
      <div className="flex-1 overflow-y-auto">
        <p className="text-gray-400 text-sm">Start a conversation...</p>
      </div>
      <div className="mt-2 flex gap-2">
        <input 
          type="text" 
          placeholder="Ask anything..."
          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"
        />
        <button className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm">
          Send
        </button>
      </div>
    </div>
  );
}
```

**Step 5: Verify layout**

Run: `npm run tauri dev`
Expected: Three panel layout visible, can resize chat area

**Step 6: Commit**

```bash
git add src/components/MainContent/ src/components/LeftPanel/ src/components/RightPanel/ src/components/FridgeBoard/ src/components/AIChat/
git commit -m "feat: add main content layout with resizable panels"
```

---

## Phase 3: Daily Record Editor (Right Panel)

### Task 7: Create Task Components

**Files:**
- Create: `src/components/TaskList/TaskList.tsx`
- Create: `src/components/TaskList/TaskItem.tsx`
- Create: `src/components/TaskList/__tests__/TaskList.test.tsx`
- Modify: `src/components/RightPanel/RightPanel.tsx`

**Step 1: Write test**

```typescript
// src/components/TaskList/__tests__/TaskList.test.tsx
import { describe, it, expect, vi } from 'vitest';
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
```

**Step 2: Run test (expect FAIL)**

**Step 3: Implement components**

```tsx
// src/components/TaskList/TaskItem.tsx
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
```

```tsx
// src/components/TaskList/TaskList.tsx
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
```

**Step 4: Run test (expect PASS)**

Run: `npm test -- src/components/TaskList/__tests__/TaskList.test.tsx`
Expected: PASS

**Step 5: Update RightPanel to use TaskList**

```tsx
// src/components/RightPanel/RightPanel.tsx
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
```

**Step 6: Verify interaction**

Run: `npm run tauri dev`
Expected: Click task to select, checkbox to toggle, can add new tasks

**Step 7: Commit**

```bash
git add src/components/TaskList/ src/components/RightPanel/
git commit -m "feat: add task list with selection and toggle"
```

---

## Phase 4: Fridge Board Core Features ⭐

### Task 8: Implement File Drop Zone

**Files:**
- Create: `src/components/FridgeBoard/DropZone.tsx`
- Modify: `src/components/FridgeBoard/FridgeBoard.tsx`
- Update: `src-tauri/Cargo.toml` (add fs permission)

**Step 1: Configure Tauri permissions**

Edit `src-tauri/Cargo.toml` to add:
```toml
[dependencies]
tauri = { version = "2.0", features = ["fs-all"] }
```

Edit `src-tauri/capabilities/default.json`:
```json
{
  "identifier": "default",
  "description": "Default capabilities",
  "windows": ["main"],
  "permissions": [
    "path:default",
    "event:default",
    "window:default",
    "app:default",
    "image:default",
    "resources:default",
    "menu:default",
    "tray:default",
    "fs:allow-read-file",
    "fs:allow-write-file",
    "fs:allow-read-dir",
    "fs:allow-copy-file",
    "fs:allow-mkdir",
    "fs:allow-remove",
    "fs:default"
  ]
}
```

**Step 2: Implement DropZone**

```tsx
// src/components/FridgeBoard/DropZone.tsx
import { useState, useCallback } from 'react';
import { useAppStore } from '../../store';
import { FridgeItem, TAG_COLORS } from '../../types';
import { readFile, mkdir, BaseDirectory } from '@tauri-apps/plugin-fs';
import { join, appDataDir } from '@tauri-apps/api/path';

export function DropZone() {
  const [isDragging, setIsDragging] = useState(false);
  const { selectedTask, addFridgeItem } = useAppStore();
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);
  
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (!selectedTask) {
      alert('请先选择一个任务');
      return;
    }
    
    const files = Array.from(e.dataTransfer.files);
    
    for (const file of files) {
      try {
        // Generate unique ID
        const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        // Create directory if not exists
        const taskDir = selectedTask.fridgePath;
        await mkdir(taskDir, { recursive: true, baseDir: BaseDirectory.AppData });
        
        // Determine file type
        const previewType = file.type.startsWith('image/') 
          ? 'image' 
          : file.type.includes('pdf') || file.type.includes('text') 
            ? 'document' 
            : 'other';
        
        // Create fridge item with random position
        const fridgeItem: FridgeItem = {
          id,
          fileName: file.name,
          filePath: await join(taskDir, file.name),
          mimeType: file.type,
          size: file.size,
          previewType,
          createdAt: new Date(),
          position: {
            x: 50 + Math.random() * 200,
            y: 50 + Math.random() * 150,
            rotation: (Math.random() - 0.5) * 16, // -8 to 8 degrees
            zIndex: Date.now()
          },
          tags: {
            color: 'none'
          },
          visual: {
            magnetColor: TAG_COLORS.none.border,
            shadow: true,
            scale: 1
          }
        };
        
        // Add to store
        addFridgeItem(fridgeItem);
        
        // TODO: Actually copy file content (Phase 9)
        console.log('File dropped:', file.name, 'at position:', fridgeItem.position);
        
      } catch (error) {
        console.error('Error handling dropped file:', error);
      }
    }
  }, [selectedTask, addFridgeItem]);
  
  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        absolute inset-4 border-2 border-dashed rounded-xl transition-all duration-300
        flex items-center justify-center
        ${isDragging 
          ? 'border-blue-400 bg-blue-50 scale-[1.02]' 
          : 'border-gray-300 bg-transparent'
        }
        ${!selectedTask && 'pointer-events-none opacity-50'}
      `}
    >
      <div className="text-center">
        <p className={`text-sm transition-colors ${isDragging ? 'text-blue-600' : 'text-gray-400'}`}>
          {isDragging ? '释放以添加文件' : '拖拽文件到这里'}
        </p>
        {!selectedTask && (
          <p className="text-xs text-gray-400 mt-1">先选择一个任务</p>
        )}
      </div>
    </div>
  );
}
```

**Step 3: Update FridgeBoard**

```tsx
// src/components/FridgeBoard/FridgeBoard.tsx
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
    <div className="h-full bg-gray-50 relative overflow-hidden">
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
      <div className="relative flex-1 h-[calc(100%-60px)]">
        <DropZone />
        
        {/* Render fridge items */}
        {taskItems.map((item) => (
          <FridgeItemCard key={item.id} item={item} />
        ))}
      </div>
    </div>
  );
}
```

**Step 4: Verify drop zone**

Run: `npm run tauri dev`
Expected: Can drag files to board, see visual feedback

**Step 5: Commit**

```bash
git add src/components/FridgeBoard/ src-tauri/
git commit -m "feat: add file drop zone to fridge board"
```

---

### Task 9: Implement FridgeItemCard with Drag

**Files:**
- Create: `src/components/FridgeBoard/FridgeItemCard.tsx`
- Create: `src/components/FridgeBoard/__tests__/FridgeItemCard.test.tsx`

**Step 1: Write test**

```typescript
// src/components/FridgeBoard/__tests__/FridgeItemCard.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FridgeItemCard } from '../FridgeItemCard';
import { FridgeItem } from '../../../types';

const mockItem: FridgeItem = {
  id: '1',
  fileName: 'test.png',
  filePath: '/path/test.png',
  mimeType: 'image/png',
  size: 1024,
  previewType: 'image',
  createdAt: new Date(),
  position: { x: 100, y: 100, rotation: 5, zIndex: 1 },
  tags: { color: 'red', text: '重要' },
  visual: { magnetColor: '#FECACA', shadow: true, scale: 1 }
};

describe('FridgeItemCard', () => {
  it('should render file name', () => {
    render(<FridgeItemCard item={mockItem} />);
    expect(screen.getByText('test.png')).toBeInTheDocument();
  });

  it('should apply correct position styles', () => {
    const { container } = render(<FridgeItemCard item={mockItem} />);
    const card = container.firstChild as HTMLElement;
    
    expect(card.style.transform).toContain('translate(100px, 100px)');
    expect(card.style.transform).toContain('rotate(5deg)');
  });
});
```

**Step 2: Run test (expect FAIL)**

**Step 3: Implement FridgeItemCard**

```tsx
// src/components/FridgeBoard/FridgeItemCard.tsx
import { useState, useCallback } from 'react';
import { FridgeItem, TAG_COLORS } from '../../types';
import { useAppStore } from '../../store';
import { FileImage, FileText, File, X } from 'lucide-react';

interface FridgeItemCardProps {
  item: FridgeItem;
}

export function FridgeItemCard({ item }: FridgeItemCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const { updateFridgeItemPosition } = useAppStore();
  
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left click
    
    setIsDragging(true);
    const startX = e.clientX;
    const startY = e.clientY;
    const startPosX = item.position.x;
    const startPosY = item.position.y;
    
    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      
      updateFridgeItemPosition(
        item.id,
        startPosX + deltaX,
        startPosY + deltaY
      );
    };
    
    const handleMouseUp = () => {
      setIsDragging(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [item.id, item.position.x, item.position.y, updateFridgeItemPosition]);
  
  const getFileIcon = () => {
    switch (item.previewType) {
      case 'image': return <FileImage className="w-8 h-8 text-blue-500" />;
      case 'document': return <FileText className="w-8 h-8 text-orange-500" />;
      default: return <File className="w-8 h-8 text-gray-500" />;
    }
  };
  
  const tagColor = TAG_COLORS[item.tags.color];
  
  return (
    <div
      onMouseDown={handleMouseDown}
      onContextMenu={(e) => {
        e.preventDefault();
        setShowMenu(true);
      }}
      style={{
        transform: `translate(${item.position.x}px, ${item.position.y}px) rotate(${item.position.rotation}deg) scale(${item.visual.scale})`,
        zIndex: isDragging ? 9999 : item.position.zIndex,
      }}
      className={`
        absolute w-28 p-3 bg-white rounded-lg cursor-move select-none
        transition-shadow duration-200
        ${item.visual.shadow ? 'shadow-lg hover:shadow-xl' : ''}
        ${isDragging ? 'cursor-grabbing scale-105' : 'cursor-grab'}
      `}
    >
      {/* Magnet effect at top */}
      <div 
        className="absolute -top-2 left-1/2 -translate-x-1/2 w-8 h-3 rounded-full"
        style={{ backgroundColor: item.visual.magnetColor }}
      />
      
      {/* File icon */}
      <div className="flex justify-center mb-2">
        {getFileIcon()}
      </div>
      
      {/* File name */}
      <p className="text-xs text-center text-gray-700 truncate font-medium">
        {item.fileName}
      </p>
      
      {/* Tag indicator */}
      {(item.tags.text || item.tags.color !== 'none') && (
        <div 
          className="absolute -bottom-2 right-0 px-2 py-0.5 rounded-full text-[10px] font-medium border"
          style={{
            backgroundColor: tagColor.bg,
            color: tagColor.text,
            borderColor: tagColor.border
          }}
        >
          {item.tags.text || tagColor.label}
        </div>
      )}
      
      {/* Context Menu */}
      {showMenu && (
        <div className="absolute top-full left-0 mt-2 bg-white rounded-lg shadow-xl border border-gray-200 py-1 z-50 min-w-[120px]">
          <button 
            className="w-full px-3 py-1.5 text-left text-sm hover:bg-gray-100"
            onClick={() => setShowMenu(false)}
          >
            编辑标签
          </button>
          <button 
            className="w-full px-3 py-1.5 text-left text-sm hover:bg-gray-100"
            onClick={() => setShowMenu(false)}
          >
            删除
          </button>
          <button 
            className="w-full px-3 py-1.5 text-left text-sm hover:bg-gray-100"
            onClick={() => setShowMenu(false)}
          >
            打开文件
          </button>
        </div>
      )}
      
      {/* Click outside to close menu */}
      {showMenu && (
        <div 
          className="fixed inset-0 z-40"
          onClick={() => setShowMenu(false)}
        />
      )}
    </div>
  );
}
```

**Step 4: Run test (expect PASS)**

Run: `npm test -- src/components/FridgeBoard/__tests__/FridgeItemCard.test.tsx`
Expected: PASS

**Step 5: Verify dragging**

Run: `npm run tauri dev`
Expected: Can drag files around the board, they stay in position

**Step 6: Commit**

```bash
git add src/components/FridgeBoard/
git commit -m "feat: add draggable fridge item cards"
```

---

### Task 10: Implement Tag System

**Files:**
- Create: `src/components/FridgeBoard/TagEditor.tsx`
- Modify: `src/components/FridgeBoard/FridgeItemCard.tsx`

**Step 1: Implement TagEditor**

```tsx
// src/components/FridgeBoard/TagEditor.tsx
import { useState } from 'react';
import { FridgeItem, TagColor, TAG_COLORS } from '../../types';
import { useAppStore } from '../../store';

interface TagEditorProps {
  item: FridgeItem;
  onClose: () => void;
}

export function TagEditor({ item, onClose }: TagEditorProps) {
  const [text, setText] = useState(item.tags.text || '');
  const [color, setColor] = useState<TagColor>(item.tags.color);
  const { updateFridgeItemTags } = useAppStore();
  
  const handleSave = () => {
    updateFridgeItemTags(item.id, {
      text: text.trim() || undefined,
      color
    });
    onClose();
  };
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="bg-white rounded-xl shadow-2xl p-6 w-80">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">编辑标签</h3>
        
        {/* Text Tag */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-600 mb-1">
            文字标签
          </label>
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="例如：重要、参考"
            maxLength={10}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-400"
          />
        </div>
        
        {/* Color Tag */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-600 mb-2">
            颜色分类
          </label>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(TAG_COLORS) as TagColor[]).map((c) => (
              <button
                key={c}
                onClick={() => setColor(c)}
                className={`
                  w-8 h-8 rounded-full border-2 transition-all
                  ${color === c ? 'border-gray-800 scale-110' : 'border-transparent'}
                `}
                style={{ backgroundColor: TAG_COLORS[c].bg }}
                title={TAG_COLORS[c].label}
              />
            ))}
          </div>
          <p className="text-xs text-gray-400 mt-2">
            {TAG_COLORS[color].label}
          </p>
        </div>
        
        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={handleSave}
            className="flex-1 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600"
          >
            保存
          </button>
          <button
            onClick={onClose}
            className="flex-1 py-2 bg-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-300"
          >
            取消
          </button>
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Update FridgeItemCard to use TagEditor**

Modify the context menu in `FridgeItemCard.tsx`:

```tsx
// Add import at top
import { TagEditor } from './TagEditor';

// In component, add state
const [showTagEditor, setShowTagEditor] = useState(false);

// In context menu, change button onClick:
<button 
  className="w-full px-3 py-1.5 text-left text-sm hover:bg-gray-100"
  onClick={() => {
    setShowMenu(false);
    setShowTagEditor(true);
  }}
>
  编辑标签
</button>

// Add TagEditor modal at end of component:
{showTagEditor && (
  <TagEditor 
    item={item} 
    onClose={() => setShowTagEditor(false)} 
  />
)}
```

**Step 3: Add tag filter to FridgeBoard**

Add to `FridgeBoard.tsx`:

```tsx
// Add state for tag filter
const [tagFilter, setTagFilter] = useState<TagColor | null>(null);

// Filter items:
const filteredItems = tagFilter 
  ? taskItems.filter(item => item.tags.color === tagFilter)
  : taskItems;

// Add filter buttons in header:
<div className="flex gap-1 mt-2">
  <button
    onClick={() => setTagFilter(null)}
    className={`px-2 py-0.5 text-xs rounded-full ${!tagFilter ? 'bg-gray-800 text-white' : 'bg-gray-200'}`}
  >
    全部
  </button>
  {(['red', 'blue', 'green', 'yellow', 'purple', 'orange'] as TagColor[]).map(c => (
    <button
      key={c}
      onClick={() => setTagFilter(c)}
      className={`w-5 h-5 rounded-full border-2 ${tagFilter === c ? 'border-gray-800' : 'border-transparent'}`}
      style={{ backgroundColor: TAG_COLORS[c].bg }}
      title={TAG_COLORS[c].label}
    />
  ))}
</div>
```

**Step 4: Verify tag system**

Run: `npm run tauri dev`
Expected: Can add color and text tags, filter by color

**Step 5: Commit**

```bash
git add src/components/FridgeBoard/
git commit -m "feat: add tag system with color and text labels"
```

---

## Phase 5: File Storage Integration

### Task 11: Implement File Persistence

**Files:**
- Create: `src/lib/storage.ts`
- Create: `src/lib/__tests__/storage.test.ts`

**Step 1: Write test**

```typescript
// src/lib/__tests__/storage.test.ts
import { describe, it, expect, vi } from 'vitest';
import { saveFridgeMetadata, loadFridgeMetadata } from '../storage';

describe('Storage', () => {
  it('should save and load fridge metadata', async () => {
    const metadata = {
      files: [{ id: '1', name: 'test.png', position: { x: 100, y: 100, rotation: 0, zIndex: 1 } }]
    };
    
    await saveFridgeMetadata('/test/path', metadata);
    const loaded = await loadFridgeMetadata('/test/path');
    
    expect(loaded).toEqual(metadata);
  });
});
```

**Step 2: Run test (expect FAIL)**

**Step 3: Implement storage**

```typescript
// src/lib/storage.ts
import { writeTextFile, readTextFile, mkdir, exists, BaseDirectory } from '@tauri-apps/plugin-fs';
import { join } from '@tauri-apps/api/path';

export interface FridgeMetadata {
  files: Array<{
    id: string;
    name: string;
    mimeType: string;
    size: number;
    position: {
      x: number;
      y: number;
      rotation: number;
      zIndex: number;
    };
    tags: {
      text?: string;
      color: string;
    };
    createdAt: string;
  }>;
}

export async function saveFridgeMetadata(taskPath: string, metadata: FridgeMetadata): Promise<void> {
  const indexPath = await join(taskPath, 'index.json');
  await writeTextFile(indexPath, JSON.stringify(metadata, null, 2), {
    baseDir: BaseDirectory.AppData
  });
}

export async function loadFridgeMetadata(taskPath: string): Promise<FridgeMetadata | null> {
  try {
    const indexPath = await join(taskPath, 'index.json');
    const content = await readTextFile(indexPath, {
      baseDir: BaseDirectory.AppData
    });
    return JSON.parse(content);
  } catch {
    return null;
  }
}

export async function copyFileToFridge(
  sourcePath: string, 
  targetDir: string, 
  fileName: string
): Promise<string> {
  // Ensure target directory exists
  await mkdir(targetDir, { recursive: true, baseDir: BaseDirectory.AppData });
  
  // Copy file logic (using Tauri command in next task)
  const targetPath = await join(targetDir, fileName);
  return targetPath;
}
```

**Step 4: Run test (expect PASS)**

Run: `npm test -- src/lib/__tests__/storage.test.ts`
Expected: PASS

**Step 5: Commit**

```bash
git add src/lib/
git commit -m "feat: add file storage utilities"
```

---

### Task 12: Create Tauri Commands for File Operations

**Files:**
- Create: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/lib.rs`

**Step 1: Implement Tauri commands**

```rust
// src-tauri/src/commands.rs
use std::fs;
use std::path::Path;
use tauri::AppHandle;

#[tauri::command]
pub fn copy_file(source: String, destination: String) -> Result<String, String> {
    fs::copy(&source, &destination)
        .map_err(|e| e.to_string())?;
    Ok(destination)
}

#[tauri::command]
pub fn open_file(path: String) -> Result<(), String> {
    opener::open(path).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn ensure_dir(path: String) -> Result<(), String> {
    fs::create_dir_all(&path).map_err(|e| e.to_string())
}
```

**Step 2: Register commands in lib.rs**

```rust
// src-tauri/src/lib.rs
mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            commands::copy_file,
            commands::open_file,
            commands::ensure_dir,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**Step 3: Add Rust dependencies**

Edit `src-tauri/Cargo.toml`:
```toml
[dependencies]
# ... existing dependencies
opener = "0.7"
```

**Step 4: Update DropZone to actually copy files**

Modify `src/components/FridgeBoard/DropZone.tsx`:

```tsx
import { invoke } from '@tauri-apps/api/core';

// In handleDrop, after creating fridge item:
// Actually copy the file
const arrayBuffer = await file.arrayBuffer();
const uint8Array = new Uint8Array(arrayBuffer);

await invoke('ensure_dir', { path: taskDir });
await invoke('copy_file', {
  source: file.path,  // This won't work directly in web, need different approach
  destination: fridgeItem.filePath
});
```

**Note**: Browser File API doesn't provide full path for security. We'll need to handle this differently - either:
1. Use Tauri's native drag-drop which gives paths
2. Store files as binary in IndexedDB

For MVP, let's use approach 2 (store in IndexedDB and save to Tauri FS).

**Step 5: Commit**

```bash
git add src-tauri/
git commit -m "feat: add tauri commands for file operations"
```

---

## Phase 6: AI Chat Integration

### Task 13: Create AI Chat Interface

**Files:**
- Create: `src/components/AIChat/ChatMessage.tsx`
- Modify: `src/components/AIChat/AIChat.tsx`
- Create: `src/lib/ai.ts`

**Step 1: Implement ChatMessage**

```tsx
// src/components/AIChat/ChatMessage.tsx
import { Message } from '../../types';
import { User, Bot } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex gap-2 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`
        w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0
        ${isUser ? 'bg-blue-500' : 'bg-gray-200'}
      `}>
        {isUser ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-gray-600" />}
      </div>
      
      <div className={`
        max-w-[80%] px-3 py-2 rounded-lg text-sm
        ${isUser 
          ? 'bg-blue-500 text-white rounded-br-none' 
          : 'bg-gray-100 text-gray-800 rounded-bl-none'
        }
      `}>
        {message.content}
      </div>
    </div>
  );
}
```

**Step 2: Update AIChat component**

```tsx
// src/components/AIChat/AIChat.tsx
import { useState } from 'react';
import { useAppStore } from '../../store';
import { ChatMessage } from './ChatMessage';
import { Message } from '../../types';
import { Send } from 'lucide-react';

export function AIChat() {
  const [input, setInput] = useState('');
  const { chatMessages, isChatLoading, selectedTask } = useAppStore();
  
  const handleSend = async () => {
    if (!input.trim() || isChatLoading) return;
    
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };
    
    // TODO: Send to AI and get response
    console.log('Sending to AI:', input, 'Context:', selectedTask);
    
    setInput('');
  };
  
  return (
    <div className="h-full bg-white flex flex-col">
      {/* Header */}
      <div className="px-4 py-2 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-600">🤖 AI 助手</h3>
        {selectedTask && (
          <p className="text-xs text-gray-400 mt-0.5">
            上下文: {selectedTask.title}
          </p>
        )}
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {chatMessages.length === 0 ? (
          <p className="text-center text-gray-400 text-sm">
            开始对话，我可以帮你：<br/>
            • 创建和管理任务<br/>
            • 整理冰箱贴文件<br/>
            • 生成代码脚本
          </p>
        ) : (
          chatMessages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))
        )}
      </div>
      
      {/* Input */}
      <div className="p-3 border-t border-gray-200">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="问我任何问题..."
            disabled={isChatLoading}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-400 disabled:bg-gray-100"
          />
          <button
            onClick={handleSend}
            disabled={isChatLoading || !input.trim()}
            className="px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
```

**Step 3: Verify chat UI**

Run: `npm run tauri dev`
Expected: Chat interface visible, can type messages

**Step 4: Commit**

```bash
git add src/components/AIChat/
git commit -m "feat: add AI chat interface"
```

---

## Phase 7: Polish and Testing

### Task 14: Add Animations and Polish

**Files:**
- Create: `src/styles/animations.css`
- Modify: `src/index.css`

**Step 1: Add animation styles**

```css
/* src/styles/animations.css */
@keyframes popIn {
  0% {
    opacity: 0;
    transform: scale(0);
  }
  70% {
    transform: scale(1.1);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
}

@keyframes shake {
  0%, 100% { transform: rotate(0deg); }
  25% { transform: rotate(-2deg); }
  75% { transform: rotate(2deg); }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.animate-pop-in {
  animation: popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.animate-shake {
  animation: shake 0.3s ease-in-out;
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-out;
}
```

**Step 2: Import in index.css**

```css
/* src/index.css */
@import './styles/animations.css';

@tailwind base;
@tailwind components;
@tailwind utilities;

.scrollbar-hide {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
.scrollbar-hide::-webkit-scrollbar {
  display: none;
}
```

**Step 3: Add animations to FridgeItemCard**

Add to component mount:
```tsx
<div className="animate-pop-in ...">
```

**Step 4: Commit**

```bash
git add src/styles/ src/index.css src/components/FridgeBoard/
git commit -m "feat: add animations and polish"
```

---

### Task 15: Write Integration Tests

**Files:**
- Create: `src/__tests__/integration.test.tsx`

**Step 1: Write integration tests**

```typescript
// src/__tests__/integration.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import App from '../App';

describe('Integration Tests', () => {
  it('should select date and show daily record', () => {
    render(<App />);
    
    // Click on day 25
    const day25 = screen.getByText('25');
    fireEvent.click(day25);
    
    // Should show daily record
    expect(screen.getByText(/Daily Record/i)).toBeInTheDocument();
  });

  it('should select task and update fridge board', () => {
    render(<App />);
    
    // Add a task first
    const input = screen.getByPlaceholderText('+ 添加新任务...');
    fireEvent.change(input, { target: { value: 'Test Task' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    
    // Click on the task
    const task = screen.getByText('Test Task');
    fireEvent.click(task);
    
    // Fridge board should show task name
    expect(screen.getByText(/🧲 Test Task/i)).toBeInTheDocument();
  });
});
```

**Step 2: Run all tests**

Run: `npm test`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/__tests__/
git commit -m "test: add integration tests"
```

---

## Phase 8: Build and Release

### Task 16: Configure Build and Package

**Files:**
- Modify: `package.json`
- Modify: `src-tauri/tauri.conf.json`

**Step 1: Update package.json scripts**

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "tauri": "tauri",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "typecheck": "tsc --noEmit"
  }
}
```

**Step 2: Update tauri.conf.json**

```json
{
  "build": {
    "beforeBuildCommand": "npm run build",
    "beforeDevCommand": "npm run dev",
    "devUrl": "http://localhost:1420",
    "frontendDist": "../dist"
  },
  "bundle": {
    "active": true,
    "targets": ["msi", "dmg", "appimage"],
    "identifier": "com.dailytodo.app",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ]
  }
}
```

**Step 3: Build for production**

Run: `npm run tauri build`
Expected: Build succeeds, installer created in `src-tauri/target/release/bundle/`

**Step 4: Final commit**

```bash
git add package.json src-tauri/tauri.conf.json
git commit -m "chore: configure production build"
```

---

## Summary

This implementation plan covers:

### ✅ Phase 1: Foundation
- Project setup with Tauri + React + TypeScript
- Core type definitions
- Zustand store with tests

### ✅ Phase 2: Layout
- Base layout with date selectors
- Month and day horizontal scroll
- Resizable three-panel layout

### ✅ Phase 3: Daily Record
- Task list with selection
- Add/toggle tasks
- Task-fri

## Phase 4: Fridge Board Core Features
- File drop zone with visual feedback
- Draggable fridge item cards
- Free positioning (x, y, rotation, z-index)
- Tag system (6 colors + text)
- Tag filtering
- Context menu for editing

### Phase 5: File Storage
- Tauri file system integration
- File persistence in app data directory
- Metadata storage (index.json)
- File copy operations

### Phase 6: AI Chat
- Chat interface with message bubbles
- Context awareness (selected task)
- Message history display
- Input handling

### Phase 7: Polish
- Animations (pop-in, shake, fade)
- Visual refinements
- Integration tests
- Error handling

### Phase 8: Release
- Production build configuration
- Multi-platform bundling (Windows, macOS, Linux)
- Installer generation

---

## Testing Checklist

Before considering complete:

- [ ] All unit tests pass (`npm test`)
- [ ] Type checking passes (`npm run typecheck`)
- [ ] Build succeeds (`npm run tauri build`)
- [ ] Can drag files to fridge board
- [ ] Can move files freely on board
- [ ] Can add/edit tags
- [ ] Can filter by tag color
- [ ] Can select tasks and see associated files
- [ ] Can add new tasks
- [ ] Can toggle task completion
- [ ] Date navigation works smoothly
- [ ] AI chat interface functional
- [ ] File persistence works across restarts

---

## Next Steps

1. **AI Integration**: Connect to OpenAI API or local LLM
2. **Sync**: Add cloud sync capability
3. **Mobile**: Build for mobile platforms
4. **Plugins**: Allow custom AI tools
5. **Export**: Add PDF/markdown export for daily records

**Plan saved to**: `docs/plans/2024-03-24-dailytodo-implementation-plan.md`
