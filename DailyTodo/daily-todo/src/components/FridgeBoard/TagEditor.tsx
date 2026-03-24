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