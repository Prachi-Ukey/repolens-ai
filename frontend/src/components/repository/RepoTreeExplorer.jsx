import React, { useState } from 'react';
import { Folder, FolderOpen, FileCode, ChevronRight, ChevronDown, Search } from 'lucide-react';

const FileTreeNode = ({ node, onSelectFile, selectedFileId, searchTerm }) => {
  const [isOpen, setIsOpen] = useState(true);

  const isDirectory = node.type === 'directory';

  // Search filter matching
  const matchesSearch = !searchTerm || node.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    (isDirectory && node.children?.some(c => c.name.toLowerCase().includes(searchTerm.toLowerCase())));

  if (!matchesSearch) return null;

  if (isDirectory) {
    return (
      <div className="select-none text-xs">
        <div
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center space-x-1.5 px-2 py-1 hover:bg-[#21262D] rounded cursor-pointer text-slate-300 font-mono"
        >
          {isOpen ? <ChevronDown className="w-3.5 h-3.5 text-slate-400" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-400" />}
          {isOpen ? <FolderOpen className="w-4 h-4 text-blue-400" /> : <Folder className="w-4 h-4 text-blue-400" />}
          <span className="truncate">{node.name}</span>
        </div>
        {isOpen && (
          <div className="pl-3.5 border-l border-[#30363D] ml-2 mt-0.5 space-y-0.5">
            {node.children.map((child, idx) => (
              <FileTreeNode
                key={child.path || idx}
                node={child}
                onSelectFile={onSelectFile}
                selectedFileId={selectedFileId}
                searchTerm={searchTerm}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  const isSelected = selectedFileId === node.id;

  return (
    <div
      onClick={() => onSelectFile(node)}
      className={`flex items-center space-x-2 px-2 py-1 rounded cursor-pointer text-xs font-mono transition ${
        isSelected
          ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
          : 'text-slate-400 hover:bg-[#21262D] hover:text-slate-200'
      }`}
    >
      <FileCode className="w-4 h-4 text-slate-400 shrink-0" />
      <span className="truncate">{node.name}</span>
    </div>
  );
};

export const RepoTreeExplorer = ({ tree, onSelectFile, selectedFileId }) => {
  const [searchTerm, setSearchTerm] = useState('');

  return (
    <div className="h-full flex flex-col bg-[#161B22] border-r border-[#30363D] select-none">
      <div className="p-3 border-b border-[#30363D]">
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" />
          <input
            type="text"
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-[#0D1117] border border-[#30363D] focus:border-blue-500 rounded-md pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 font-mono outline-none"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {tree && tree.length > 0 ? (
          tree.map((node, idx) => (
            <FileTreeNode
              key={node.path || idx}
              node={node}
              onSelectFile={onSelectFile}
              selectedFileId={selectedFileId}
              searchTerm={searchTerm}
            />
          ))
        ) : (
          <p className="text-xs text-slate-500 p-2 font-mono italic">No files found.</p>
        )}
      </div>
    </div>
  );
};
