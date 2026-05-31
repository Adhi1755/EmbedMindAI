import React, { useEffect, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface MarkdownRendererProps {
  content: string;
  typingSpeed?: number;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  typingSpeed = 1,
}) => {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    let index = 0;
    let frameId: number;

    const type = () => {
      const batchSize = 5;
      const nextIndex = Math.min(index + batchSize, content.length);
      setDisplayedText(content.slice(0, nextIndex));
      index = nextIndex;

      if (index < content.length) {
        frameId = requestAnimationFrame(type);
      }
    };

    frameId = requestAnimationFrame(type);
    return () => cancelAnimationFrame(frameId);
  }, [content]);

  return (
    <div className="prose dark:prose-invert max-w-none">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            // In react-markdown v9, `inline` prop was removed.
            // Detect inline code by absence of a language class and no newlines.
            const isInline = !match && !String(children).includes("\n");
            return !isInline && match ? (
              <SyntaxHighlighter
                style={oneDark}
                language={match[1]}
                PreTag="div"
                wrapLines={true} 
                 lineProps={() => {
    return {
      style: {
        backgroundColor: "transparent", // ✅ removes grey line background
        display: "block",               // ✅ ensures full width lines
      },
    };
  }}
                customStyle={{
                  background: "transparent",
                  border: "none",        
                  boxShadow: "none",         
                  borderRadius: "0.5rem",
                  fontSize: "0.875rem",
                  padding: "1rem",
                  margin: 0,                 
                }}
                {...props}
              >
                {String(children).replace(/\n$/, "")}
              </SyntaxHighlighter>
            ) : (
              <code
                className="bg-gray-100 dark:bg-gray-800 rounded px-1 py-0.5 text-gray-400"
                {...props}
              >
                {children}
              </code>
            );
          },
        }}
      >
        {displayedText}
      </Markdown>
    </div>
  );
};

export default MarkdownRenderer;
