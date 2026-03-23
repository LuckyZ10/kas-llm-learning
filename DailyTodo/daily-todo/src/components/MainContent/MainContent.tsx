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