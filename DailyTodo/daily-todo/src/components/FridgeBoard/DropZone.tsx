import { useState, useCallback } from 'react';
import { useAppStore } from '../../store';
import { FridgeItem, TAG_COLORS } from '../../types';

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
          filePath: `${selectedTask.fridgePath}/${file.name}`,
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