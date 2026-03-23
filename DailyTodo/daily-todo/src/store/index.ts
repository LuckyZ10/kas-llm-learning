import { create } from 'zustand';
import { Task, DailyRecord, FridgeItem, Message, TAG_COLORS, TagColor } from '../types';

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
  updateFridgeItemTags: (id: string, tags: { text?: string; color: TagColor }) => void;
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