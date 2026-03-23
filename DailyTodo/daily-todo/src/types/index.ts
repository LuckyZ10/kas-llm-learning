// src/types/index.ts

export type TagColor = 'red' | 'blue' | 'green' | 'yellow' | 'purple' | 'orange' | 'none';

export interface Task {
  id: string;              // 唯一标识
  title: string;           // 任务标题
  description?: string;    // 详细描述（富文本）
  completed: boolean;      // 完成状态
  createdAt: Date;
  updatedAt: Date;
  fridgePath: string;      // 该任务的冰箱贴目录路径
}

export interface DailyRecord {
  date: string;            // YYYY-MM-DD
  tasks: Task[];           // 该日所有任务
  aiContext: string;       // 对话上下文摘要
  lastModified: Date;
}

export interface Position {
  x: number;            // 绝对X坐标（px）
  y: number;            // 绝对Y坐标（px）
  rotation: number;     // 旋转角度 (-8° ~ 8°)
  zIndex: number;      // 层级（可叠放）
}

export interface Tags {
  text?: string;        // 文字标签（如"重要"、"参考"、"待处理"）
  color: TagColor;     // 颜色标签分类
}

export interface VisualStyle {
  magnetColor: string;  // 磁贴颜色（根据tags.color自动设置）
  shadow: boolean;      // 是否显示投影（可开关）
  scale: number;       // 缩放比例（0.5 ~ 1.5）
}

export interface FridgeItem {
  id: string;
  fileName: string;
  filePath: string;        // Tauri FS中的存储路径
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