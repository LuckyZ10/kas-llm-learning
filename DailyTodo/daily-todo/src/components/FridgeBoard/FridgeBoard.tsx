export function FridgeBoard() {
  return (
    <div className="h-full bg-gray-50 p-4 relative overflow-hidden">
      <div className="absolute inset-0 opacity-5 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-gray-400 to-transparent" />
      <h3 className="text-sm font-medium text-gray-600 mb-2">冰箱贴板</h3>
      <p className="text-gray-400 text-sm">点击任务查看文件...</p>
    </div>
  );
}