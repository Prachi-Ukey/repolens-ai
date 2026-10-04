import React from 'react';
import { Link } from 'react-router-dom';
import { FolderGit2, FileCode2, Layers, Clock, ArrowRight, Trash2 } from 'lucide-react';

export const RepositoryCard = ({ repo, onDelete }) => {
  const isCompleted = repo.status === 'completed';

  return (
    <div className="bg-[#161B22] border border-[#30363D] hover:border-[#484F58] rounded-xl p-5 flex flex-col justify-between transition group shadow-sm hover:shadow-md">
      <div>
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <FolderGit2 className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[11px] font-mono text-slate-500">{repo.owner}</span>
              <h4 className="text-base font-bold text-white group-hover:text-blue-400 transition font-mono truncate max-w-[200px]">
                {repo.name}
              </h4>
            </div>
          </div>
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
            isCompleted 
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' 
              : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
          }`}>
            {repo.status}
          </span>
        </div>

        <p className="text-xs text-slate-400 line-clamp-2 mb-4 h-8">
          {repo.description || "No description provided."}
        </p>

        <div className="grid grid-cols-3 gap-2 py-2.5 px-3 bg-[#0D1117] rounded-lg border border-[#30363D] text-[11px] font-mono mb-4">
          <div>
            <span className="text-slate-500 block text-[10px]">Lang</span>
            <span className="text-slate-300 font-medium truncate block">{repo.primary_language || 'Text'}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">Files</span>
            <span className="text-slate-300 font-medium block">{repo.file_count}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">Chunks</span>
            <span className="text-slate-300 font-medium block">{repo.chunk_count}</span>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between pt-3 border-t border-[#21262D]">
        <span className="text-[10px] text-slate-500 flex items-center gap-1 font-mono">
          <Clock className="w-3 h-3" />
          {new Date(repo.created_at).toLocaleDateString()}
        </span>

        <div className="flex items-center space-x-2">
          {onDelete && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(repo.id);
              }}
              className="p-1.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition"
              title="Delete Repository"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}

          <Link
            to={`/repositories/${repo.id}`}
            className="bg-blue-600/20 hover:bg-blue-600 text-blue-400 hover:text-white px-3 py-1.5 rounded-md text-xs font-medium flex items-center space-x-1.5 transition border border-blue-500/30"
          >
            <span>Open Chat</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>
    </div>
  );
};
