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