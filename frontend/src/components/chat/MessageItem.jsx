import React from 'react';

export const MessageItem = ({ message, onSelectCitation }) => {
  const isUser = message.sender === 'user';

  let sources = [];

if (Array.isArray(message.sources)) {
  sources = message.sources;
}
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
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              Grounded Evidence
            </span>
          )}

        </div>

        {/* Answer */}
        <div className="text-xs text-slate-300 leading-relaxed font-sans whitespace-pre-wrap break-words">
          {message.content}
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