import React, { useState } from 'react';
import { FileCode, Copy, Check, Search, X } from 'lucide-react';

export const CodeViewer = ({ file, highlightedLines, onClose }) => {
  const [copied, setCopied] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  if (!file) {
    return (
      <div className="h-full bg-[#0D1117] border-l border-[#30363D] flex flex-col items-center justify-center p-6 text-center text-slate-500 font-mono text-xs">
        <FileCode className="w-10 h-10 mb-3 opacity-30 text-blue-400" />
        <p>Select a file from the repository tree or click an AI source citation to view code.</p>
      </div>
    );
  }

  const lines = (file.content || '').splitlines ? file.content.splitlines() : (file.content || '').split('\n');

  const handleCopy = () => {
    navigator.clipboard.writeText(file.content || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const startHighlight = highlightedLines?.start || 0;
  const endHighlight = highlightedLines?.end || 0;

  return (
    <div className="h-full flex flex-col bg-[#0D1117] border-l border-[#30363D] font-mono text-xs overflow-hidden">
      {/* File Header */}
      <div className="h-10 bg-[#161B22] border-b border-[#30363D] px-4 flex items-center justify-between">
        <div className="flex items-center space-x-2 truncate">
          <FileCode className="w-4 h-4 text-blue-400 shrink-0" />
          <span className="font-bold text-slate-200 truncate">{file.file_path}</span>
          <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 uppercase">
            {file.language}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2 top-2" />
            <input
              type="text"
              placeholder="Find in file..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-[#0D1117] border border-[#30363D] focus:border-blue-500 rounded px-2 pl-7 py-1 text-[11px] text-slate-200 outline-none w-32 focus:w-48 transition-all"
            />
          </div>

          <button
            onClick={handleCopy}
            className="p-1.5 rounded hover:bg-[#21262D] text-slate-400 hover:text-white transition flex items-center gap-1"
            title="Copy Code"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>

          {onClose && (
            <button onClick={onClose} className="p-1.5 rounded hover:bg-[#21262D] text-slate-400 hover:text-white transition">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Code Editor Body */}
      <div className="flex-1 overflow-auto p-2 font-mono">
        <table className="w-full border-collapse">
          <tbody>
            {lines.map((line, idx) => {
              const lineNum = idx + 1;
              const isHighlighted = lineNum >= startHighlight && lineNum <= endHighlight;
              const isSearchMatch = searchTerm && line.toLowerCase().includes(searchTerm.toLowerCase());

              return (
                <tr
                  key={idx}
                  className={`hover:bg-[#161B22] ${
                    isHighlighted ? 'bg-blue-900/30 text-blue-200 border-l-2 border-blue-500' : ''
                  } ${isSearchMatch ? 'bg-amber-900/30' : ''}`}
                >
                  <td className="w-12 text-right pr-4 text-slate-600 select-none text-[11px]">
                    {lineNum}
                  </td>
                  <td className="whitespace-pre text-slate-300 leading-relaxed py-0.5 px-2 font-mono text-xs">
                    {line}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
