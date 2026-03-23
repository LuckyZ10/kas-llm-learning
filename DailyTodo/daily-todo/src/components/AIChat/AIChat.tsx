export function AIChat() {
  return (
    <div className="h-full bg-white p-4 flex flex-col">
      <h3 className="text-sm font-medium text-gray-600 mb-2">AI 助手</h3>
      <div className="flex-1 overflow-y-auto">
        <p className="text-gray-400 text-sm">Start a conversation...</p>
      </div>
      <div className="mt-2 flex gap-2">
        <input 
          type="text" 
          placeholder="Ask anything..."
          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"
        />
        <button className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm">
          Send
        </button>
      </div>
    </div>
  );
}