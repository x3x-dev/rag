import { useState, useRef, useEffect } from 'react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  selectedModel: 'simple' | 'enhanced' | 'no_rag';
  onModelChange: (model: 'simple' | 'enhanced' | 'no_rag') => void;
}

const ChatInput = ({ onSendMessage, disabled = false, selectedModel, onModelChange }: ChatInputProps) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
      if (textareaRef.current) {
        textareaRef.current.style.height = '2.5rem';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '2.5rem';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [message]);

  return (
    <div className="border-t border-gray-200 bg-white p-4">
      {/* Model selector bar */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-gray-700">Model:</span>
                   <div className="flex bg-gray-50 rounded-lg p-1 border border-gray-200">
           <button
             type="button"
             onClick={() => onModelChange('no_rag')}
             disabled={disabled}
             className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200 ${
               selectedModel === 'no_rag'
                 ? 'bg-blue-500 text-white shadow-sm'
                 : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
             } disabled:opacity-50 disabled:cursor-not-allowed`}
           >
             No RAG
           </button>
           <button
             type="button"
             onClick={() => onModelChange('simple')}
             disabled={disabled}
             className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200 ${
               selectedModel === 'simple'
                 ? 'bg-blue-500 text-white shadow-sm'
                 : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
             } disabled:opacity-50 disabled:cursor-not-allowed`}
           >
             Simple
           </button>
           <button
             type="button"
             onClick={() => onModelChange('enhanced')}
             disabled={disabled}
             className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200 ${
               selectedModel === 'enhanced'
                 ? 'bg-blue-500 text-white shadow-sm'
                 : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
             } disabled:opacity-50 disabled:cursor-not-allowed`}
           >
             Enhanced
           </button>
          </div>
        </div>
        
        <div className="text-xs text-gray-500">
          {selectedModel === 'no_rag' && 'GPT-3.5 without RAG context'}
          {selectedModel === 'simple' && 'Fast, concise responses with RAG'}
          {selectedModel === 'enhanced' && 'Detailed, comprehensive responses with RAG'}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-3">
        <div className="flex-1 min-w-0">
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about MingLib functions, examples, or documentation..."
              disabled={disabled}
              className="w-full resize-none rounded-lg border border-gray-300 px-4 py-3 pr-12 text-sm placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-50 disabled:cursor-not-allowed"
              style={{ minHeight: '2.5rem', maxHeight: '7.5rem' }}
              rows={1}
            />
            
            {/* Character count */}
            <div className="absolute bottom-1 right-1 text-xs text-gray-400">
              {message.length > 100 && `${message.length}/1000`}
            </div>
          </div>
          
          {/* Helper text */}
          <div className="mt-1 text-xs text-gray-500">
            Press Enter to send, Shift+Enter for new line
          </div>
        </div>

        {/* Send button - aligned with textarea */}
        <div className="flex items-start pt-0">
          <button
            type="submit"
            disabled={!message.trim() || disabled}
            className="flex-shrink-0 inline-flex items-center justify-center w-10 h-10 rounded-lg bg-blue-500 text-white hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                      >
            {disabled ? (
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
            </svg>
            )}
          </button>
        </div>
      </form>


    </div>
  );
};

export default ChatInput;
