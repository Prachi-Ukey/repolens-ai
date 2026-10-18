function renderInlineText(text, onSelectCitation) {
  if (!text) return null;

  const tokenRegex = /(\*\*[^*]+\*\*|`[^`]+`|\[[a-zA-Z0-9_\-./\\]+:\d+(?:[-–—]\d+)?\])/g;
  const parts = text.split(tokenRegex);

  return parts.map((part, idx) => {
    if (!part) return null;

    if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
      return (
        <strong key={idx} className="font-semibold text-slate-100">
          {part.slice(2, -2)}
        </strong>
      );
    }

    if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
      return (
        <code
          key={idx}
          className="px-1.5 py-0.5 rounded bg-[#161B22] text-blue-300 border border-[#30363D] font-mono text-[11px]"
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    const citeMatch = part.match(/^\[([a-zA-Z0-9_\-./\\]+):(\d+)(?:[-–—](\d+))?\]$/);
    if (citeMatch) {
      const filePath = citeMatch[1];
      const startLine = parseInt(citeMatch[2], 10);
      const endLine = citeMatch[3] ? parseInt(citeMatch[3], 10) : startLine;

      if (onSelectCitation) {
        return (
          <button
            key={idx}
            type="button"
            onClick={() => onSelectCitation(filePath, startLine, endLine)}
            className="inline-flex items-center text-blue-400 hover:text-blue-300 hover:underline font-mono text-[11px] mx-0.5 transition"
            title={`View ${filePath} lines ${startLine}-${endLine}`}
          >
            {part}
          </button>
        );
      }
      return (
        <span key={idx} className="font-mono text-blue-400 text-[11px] mx-0.5">
          {part}
        </span>
      );
    }

    return part;
  });
}

function renderFormattedMessage(rawContent, onSelectCitation) {
  if (!rawContent) return null;

  // Unescape backslash-escaped Markdown characters: \### -> ###, \** -> **, \- -> -, etc.
  const content = rawContent.replace(/\\([#*_`[\]()\-+!])/g, '$1');

  const lines = content.split('\n');
  const elements = [];
  let currentList = null;

  const flushList = () => {
    if (!currentList) return;
    if (currentList.type === 'ul') {
      elements.push(
        <ul key={`ul-${elements.length}`} className="space-y-1.5 my-2 list-none">
          {currentList.items.map((item, i) => (
            <li key={i} className="flex items-start gap-2 text-slate-300">
              <span className={`text-slate-500 select-none mt-0.5 text-xs ${item.indent ? 'ml-4' : ''}`}>
                •
              </span>
              <div className="flex-1">
                {renderInlineText(item.text, onSelectCitation)}
              </div>
            </li>
          ))}
        </ul>
      );
    } else if (currentList.type === 'ol') {
      elements.push(
        <ol key={`ol-${elements.length}`} className="space-y-1.5 my-2 list-none">
          {currentList.items.map((item, i) => (
            <li key={i} className="flex items-start gap-2 text-slate-300">
              <span className="font-mono text-[11px] text-slate-500 select-none mt-0.5 shrink-0">
                {item.num}.
              </span>
              <div className="flex-1">
                {renderInlineText(item.text, onSelectCitation)}
              </div>
            </li>
          ))}
        </ol>
      );
    }
    currentList = null;
  };

  lines.forEach((line, lineIdx) => {
    const trimmed = line.trim();

    if (!trimmed) {
      flushList();
      elements.push(<div key={`sp-${lineIdx}`} className="h-2" />);
      return;
    }

    // Headings: ###, ##, #
    const h3Match = line.match(/^###\s+(.+)$/);
    if (h3Match) {
      flushList();
      elements.push(
        <h4 key={`h3-${lineIdx}`} className="text-xs font-semibold text-slate-100 mt-3 mb-1.5 tracking-wide">
          {renderInlineText(h3Match[1], onSelectCitation)}
        </h4>
      );
      return;
    }

    const h2Match = line.match(/^##\s+(.+)$/);
    if (h2Match) {
      flushList();
      elements.push(
        <h3 key={`h2-${lineIdx}`} className="text-sm font-bold text-slate-100 mt-3.5 mb-1.5">
          {renderInlineText(h2Match[1], onSelectCitation)}
        </h3>
      );
      return;
    }

    const h1Match = line.match(/^#\s+(.+)$/);
    if (h1Match) {
      flushList();
      elements.push(
        <h2 key={`h1-${lineIdx}`} className="text-sm font-bold text-slate-100 mt-4 mb-2">
          {renderInlineText(h1Match[1], onSelectCitation)}
        </h2>
      );
      return;
    }

    // Unordered list item: - or *
    const ulMatch = line.match(/^(\s*)[-*]\s+(.+)$/);
    if (ulMatch) {
      const isIndented = ulMatch[1].length >= 2;
      if (!currentList || currentList.type !== 'ul') {
        flushList();
        currentList = { type: 'ul', items: [] };
      }
      currentList.items.push({ text: ulMatch[2], indent: isIndented });
      return;
    }

    // Ordered list item: 1. 2. etc.
    const olMatch = line.match(/^(\s*)(\d+)\.\s+(.+)$/);
    if (olMatch) {
      if (!currentList || currentList.type !== 'ol') {
        flushList();
        currentList = { type: 'ol', items: [] };
      }
      currentList.items.push({ num: olMatch[2], text: olMatch[3] });
      return;
    }

    // Regular paragraph
    flushList();
    elements.push(
      <p key={`p-${lineIdx}`} className="leading-relaxed my-1">
        {renderInlineText(line, onSelectCitation)}
      </p>
    );
  });

  flushList();
  return elements;
}

export const MessageItem = ({ message, onSelectCitation }) => {
  const isUser = message.sender === 'user';

  let sources = [];

  if (Array.isArray(message.sources)) {
    sources = message.sources;
  } else if (message.sources_json) {
    try {
      sources = JSON.parse(message.sources_json);
    } catch (e) {
      sources = [];
    }
  }

  const isStreaming = Boolean(message.isStreaming);
  const isGrounded = Boolean(message.grounded);

  // Remove duplicate source ranges
  const uniqueSources = sources.filter((src, index, self) => {
    return (
      index ===
      self.findIndex(
        (item) =>
          item.file_path === src.file_path &&
          item.start_line === src.start_line &&
          item.end_line === src.end_line
      )
    );
  });

  return (
    <div
      className={`flex space-x-3 p-4 rounded-xl transition ${
        isUser
          ? 'bg-[#161B22]/60 border border-[#30363D]'
          : 'bg-[#0D1117] border border-[#21262D]'
      }`}
    >

      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
          isUser
            ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
            : 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30'
        }`}
      >
        <span className="text-xs font-bold">
          {isUser ? 'U' : 'AI'}
        </span>
      </div>

      <div className="flex-1 space-y-3 overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between">

          <span className="font-mono text-xs font-bold text-slate-200">
            {isUser ? 'You' : 'RepoLens AI Assistant'}
          </span>

          {!isUser && (
            (message.isError || message.error) ? (
              <span className="text-[10px] font-mono text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20">
                Service Error
              </span>
            ) : isStreaming ? (
              <span className="text-[10px] font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                Analyzing Evidence...
              </span>
            ) : isGrounded ? (
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                Grounded Evidence
              </span>
            ) : (
              <span className="text-[10px] font-mono text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                Ungrounded / Missing Evidence
              </span>
            )
          )}

        </div>

        {/* Answer */}
        <div className={`text-xs leading-relaxed font-sans break-words ${message.isError || message.error ? 'text-rose-300' : 'text-slate-300'}`}>
          {isUser ? (
            <div className="whitespace-pre-wrap">{message.content}</div>
          ) : (
            renderFormattedMessage(message.content, onSelectCitation)
          )}
        </div>

        {/* Evidence */}
        {!isUser && uniqueSources.length > 0 && (
          <div className="pt-2">

            <div className="flex items-center gap-2 mb-2">

              <span className="text-emerald-400 text-xs">
                ✓
              </span>

              <span className="text-[10px] font-mono text-slate-500 uppercase font-bold tracking-wider">
                Evidence
              </span>

            </div>

            <div className="flex flex-wrap gap-2">

              {uniqueSources.map((src, idx) => (
                <button
                  key={`${src.file_path}-${src.start_line}-${src.end_line}-${idx}`}
                  onClick={() => {
                    if (onSelectCitation) {
                      onSelectCitation(
                        src.file_path,
                        src.start_line,
                        src.end_line
                      );
                    }
                  }}
                  title={`Open ${src.file_path} lines ${src.start_line}-${src.end_line}`}
                  className="group bg-[#161B22] hover:bg-blue-600/10 hover:border-blue-500/40 text-blue-400 border border-[#30363D] px-2.5 py-1.5 rounded-md text-[11px] font-mono flex items-center gap-1.5 transition"
                >
                  <span>
                    {src.file_path}
                    <span className="text-slate-500">
                      : L{src.start_line}–L{src.end_line}
                    </span>
                  </span>

                  <span className="text-slate-500 group-hover:text-blue-400">
                    →
                  </span>
                </button>
              ))}

            </div>
          </div>
        )}

      </div>
    </div>
  );
};