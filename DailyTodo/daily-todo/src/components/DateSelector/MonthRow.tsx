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