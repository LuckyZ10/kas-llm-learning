import { useState, useCallback } from 'react';
import { FridgeItem, TAG_COLORS } from '../../types';
import { useAppStore } from '../../store';
import { FileImage, FileText, File } from 'lucide-react';

interface FridgeItemCardProps {
  item: FridgeItem;
}

export function FridgeItemCard({ item }: FridgeItemCardProps) {
  const [isDragging, setIsDragging] = useState(false);
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
    </div>
  );
}