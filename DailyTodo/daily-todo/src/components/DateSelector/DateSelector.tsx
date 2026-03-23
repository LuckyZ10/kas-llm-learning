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