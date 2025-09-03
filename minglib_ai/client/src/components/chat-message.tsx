import { useState } from 'react';

interface CodeBlockProps {
  code: string;
  language?: string;
}

const CodeBlock = ({ code, language = 'python' }: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative bg-gray-900 rounded-lg overflow-hidden my-4">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 text-gray-300 text-sm">
        <span className="font-mono">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-2 px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 transition-colors"
        >
          {copied ? (
            <>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              Copied!
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
      <pre className="p-4 text-sm text-gray-100 overflow-x-auto">
        <code className="code-block">{code}</code>
      </pre>
    </div>
  );
};

interface MessageProps {
  message: {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
    codeBlocks?: { code: string; language?: string }[];
    reference?: string;
  };
  onSendMessage?: (content: string) => void;
}

const ChatMessage = ({ message, onSendMessage }: MessageProps) => {
  const formatContent = (content: string) => {
    // Split content by code blocks (```...```)
    const parts = content.split(/(```[\s\S]*?```)/g);
    
    return parts.map((part, index) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        const lines = part.slice(3, -3).split('\n');
        const language = lines[0].trim() || 'python';
        const code = lines.slice(1).join('\n');
        return (
          <CodeBlock 
            key={index} 
            code={code} 
            language={language}
          />
        );
      }
      
      // Regular text with enhanced markdown formatting
      return (
        <div key={index} className="prose prose-sm max-w-none">
          {part.split('\n').map((line, lineIndex) => {
            const trimmedLine = line.trim();
            
            // Skip empty lines but preserve spacing
            if (!trimmedLine) {
              return <div key={lineIndex} className="h-2" />;
            }
            
            // Check if this line is an example question (starts with •)
            const isExampleQuestion = trimmedLine.startsWith('• **') && trimmedLine.includes('?**');
            
            if (isExampleQuestion && onSendMessage) {
              // Extract the question text (remove • ** and **)
              const questionText = line.replace(/^•\s*\*\*(.*?)\*\*.*$/, '$1').trim();
              
              return (
                <button
                  key={lineIndex}
                  onClick={() => onSendMessage(questionText)}
                  className="block w-full text-left my-2 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg hover:from-blue-100 hover:to-indigo-100 hover:border-blue-300 transition-all duration-200 group shadow-sm hover:shadow-md"
                >
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full group-hover:bg-blue-600 transition-colors"></div>
                    <span className="text-blue-700 group-hover:text-blue-800 font-medium text-sm">
                      {questionText}
                    </span>
                    <svg className="w-4 h-4 text-blue-400 group-hover:text-blue-600 transition-colors ml-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </div>
                </button>
              );
            }
            
            // Headers (### Header)
            if (trimmedLine.startsWith('###')) {
              const headerText = trimmedLine.replace(/^###\s*/, '');
              return (
                <h3 key={lineIndex} className="text-lg font-semibold text-gray-900 mt-6 mb-3 border-b border-gray-200 pb-2">
                  {headerText}
                </h3>
              );
            }
            
            // Headers (## Header)
            if (trimmedLine.startsWith('##')) {
              const headerText = trimmedLine.replace(/^##\s*/, '');
              return (
                <h2 key={lineIndex} className="text-xl font-bold text-gray-900 mt-6 mb-4">
                  {headerText}
                </h2>
              );
            }
            
            // Numbered lists (1. Item)
            if (/^\d+\.\s/.test(trimmedLine)) {
              const listText = trimmedLine.replace(/^\d+\.\s*/, '');
              const formattedText = formatInlineMarkdown(listText);
              return (
                <div key={lineIndex} className="flex items-start gap-3 my-2">
                  <span className="inline-flex items-center justify-center w-6 h-6 text-xs font-medium text-white bg-blue-500 rounded-full flex-shrink-0 mt-0.5">
                    {trimmedLine.match(/^(\d+)/)?.[1]}
                  </span>
                  <div 
                    className="leading-relaxed flex-1"
                    dangerouslySetInnerHTML={{ __html: formattedText }}
                  />
                </div>
              );
            }
            
            // Bullet points (- Item or * Item)
            if (trimmedLine.startsWith('- ') || trimmedLine.startsWith('* ')) {
              const listText = trimmedLine.replace(/^[-*]\s*/, '');
              const formattedText = formatInlineMarkdown(listText);
              return (
                <div key={lineIndex} className="flex items-start gap-3 my-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full flex-shrink-0 mt-2" />
                  <div 
                    className="leading-relaxed flex-1"
                    dangerouslySetInnerHTML={{ __html: formattedText }}
                  />
                </div>
              );
            }
            
            // Reference lines (REFERENCE:)
            if (trimmedLine.startsWith('REFERENCE:')) {
              const refText = trimmedLine.replace(/^REFERENCE:\s*/, '');
              return (
                <div key={lineIndex} className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center gap-2 text-sm text-blue-800">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                    </svg>
                    <span className="font-medium">Reference:</span>
                    <span className="font-mono text-xs">{refText}</span>
                  </div>
                </div>
              );
            }
            
            // Regular paragraphs
            const formattedText = formatInlineMarkdown(line);
            return (
              <div 
                key={lineIndex} 
                dangerouslySetInnerHTML={{ __html: formattedText }}
                className="leading-relaxed my-1"
              />
            );
          })}
        </div>
      );
    });
  };

  const formatInlineMarkdown = (text: string): string => {
    return text
      // Handle **bold** text
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
      // Handle `inline code`
      .replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800">$1</code>')
      // Handle Copy text
      .replace(/\bCopy\b/g, '<span class="text-blue-600 font-medium">Copy</span>');
  };

  return (
    <div className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`flex max-w-4xl ${message.isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 ${message.isUser ? 'ml-3' : 'mr-3'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium text-white ${
            message.isUser ? 'bg-blue-500' : 'bg-gray-600'
          }`}>
            {message.isUser ? 'U' : 'AI'}
          </div>
        </div>

        {/* Message Content */}
        <div className={`flex flex-col ${message.isUser ? 'items-end' : 'items-start'}`}>
          <div className={`rounded-2xl px-4 py-3 ${
            message.isUser 
              ? 'bg-blue-500 text-white' 
              : 'bg-white border border-gray-200 text-gray-800 shadow-sm'
          }`}>
            <div className="text-sm leading-relaxed">
              {formatContent(message.content)}
            </div>
            
            {/* Reference */}
            {message.reference && !message.isUser && (
              <div className="mt-3 pt-3 border-t border-gray-200">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                  </svg>
                  <span>Reference: {message.reference}</span>
                </div>
              </div>
            )}
          </div>
          
          {/* Timestamp */}
          <div className={`text-xs text-gray-500 mt-1 ${message.isUser ? 'text-right' : 'text-left'}`}>
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
