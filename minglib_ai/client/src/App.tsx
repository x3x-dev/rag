import { useState, useEffect } from 'react';
import ChatHeader from './components/chat-header';
import ChatMessage from './components/chat-message';
import ChatInput from './components/chat-input';

// Welcome message with examples
const initialMessages = [
  {
    id: 'welcome',
    content: `Hello! I'm your MingLib AI assistant. I can help you with quantitative finance functions, risk management, portfolio optimization, and more.

Try these example questions:

• **How do I calculate portfolio VaR using minglib?**
• **List all optimization methods in the minglib.portfolio module?**
• **What's risk parity optimization in minglib and How is it implemented?**
• **What functions are in minglib.risk for stress testing?**

Select a model above and ask me anything about MingLib!`,
    isUser: false,
    timestamp: new Date(),
  },
];

function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'simple' | 'enhanced' | 'no_rag'>('simple');
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);

  // Load chat history on component mount
  useEffect(() => {
    const loadChatHistory = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/chat/history');
        if (response.ok) {
          const data = await response.json();
          if (data.messages && data.messages.length > 0) {
            // Convert backend format to frontend format
            const convertedMessages = data.messages
              .map((msg: any) => ({
                id: msg.id,
                content: msg.content,
                isUser: msg.is_user,
                timestamp: new Date(msg.timestamp),
                reference: msg.reference || (msg.is_user ? undefined : 'No reference'),
              }))
              // Sort by timestamp to ensure correct order
              .sort((a: any, b: any) => a.timestamp.getTime() - b.timestamp.getTime());
            
            // If we have history, replace initial messages with welcome + history
            if (convertedMessages.length > 0) {
              setMessages([initialMessages[0], ...convertedMessages]);
            }
          }
        }
      } catch (error) {
        console.error('Failed to load chat history:', error);
        // Keep initial messages if loading fails
      } finally {
        setIsLoadingHistory(false);
      }
    };

    loadChatHistory();
  }, []);

  const handleClearHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/chat/history', {
        method: 'DELETE',
      });
      
      if (response.ok) {
        // Reset to initial welcome message only
        setMessages(initialMessages);
      } else {
        console.error('Failed to clear chat history');
      }
    } catch (error) {
      console.error('Error clearing chat history:', error);
    }
  };

  const handleSendMessage = async (content: string) => {
    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      content,
      isUser: true,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Call backend API
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: content,
          model: selectedModel
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        // Add AI response
        const aiMessage = {
          id: result.data.id,
          content: result.data.content,
          isUser: false,
          timestamp: new Date(result.data.timestamp),
          reference: result.data.reference,
        };
        
        setMessages(prev => [...prev, aiMessage]);
      } else {
        // Handle API error
        const errorMessage = {
          id: Date.now().toString() + '_error',
          content: `Error: ${result.error?.message || 'Something went wrong'}`,
          isUser: false,
          timestamp: new Date(),
          reference: 'Error',
        };
        
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Add error message
      const errorMessage = {
        id: Date.now().toString() + '_error',
        content: `Connection error: ${error instanceof Error ? error.message : 'Unable to connect to server'}`,
        isUser: false,
        timestamp: new Date(),
        reference: 'Error',
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };



  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <ChatHeader onClearHistory={handleClearHistory} />
      
      <div className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto px-6 py-4 space-y-4 chat-container">
          {isLoadingHistory ? (
            <div className="flex justify-center items-center py-8">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
              <span className="ml-2 text-gray-600">Loading chat history...</span>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <ChatMessage 
                  key={message.id} 
                  message={message} 
                  onSendMessage={handleSendMessage}
                />
              ))}
              
              {isLoading && (
            <div className="flex justify-start mb-6">
              <div className="flex max-w-4xl">
                <div className="flex-shrink-0 mr-3">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium text-white bg-gray-600">
                    AI
                  </div>
                </div>
                <div className="flex flex-col items-start">
                  <div className="rounded-2xl px-4 py-3 bg-white border border-gray-200 text-gray-800 shadow-sm">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <span className="text-sm text-gray-500">Thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
              )}
            </>
          )}
        </div>
      </div>

      <ChatInput 
        onSendMessage={handleSendMessage} 
        disabled={isLoading || isLoadingHistory}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
      />
    </div>
  );
}

export default App;