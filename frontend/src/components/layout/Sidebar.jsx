import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, FolderGit2, PlusCircle, MessageSquare, Settings, PlayCircle } from 'lucide-react';

export const Sidebar = ({ onOpenAddModal, onLoadDemo }) => {
  const navItems = [
    { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-56 bg-[#161B22] border-r border-[#30363D] flex flex-col justify-between p-3 select-none">
      <div className="space-y-4">
        <div className="px-2 pt-2">
          <button
            onClick={onOpenAddModal}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium px-3 py-2 rounded-md flex items-center justify-center space-x-2 text-xs transition shadow-md shadow-blue-600/20"
          >
            <PlusCircle className="w-4 h-4" />
            <span>Analyze Repository</span>
          </button>

          {onLoadDemo && (
            <button
              onClick={onLoadDemo}
              className="w-full mt-2 bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 font-medium px-3 py-1.5 rounded-md flex items-center justify-center space-x-2 text-xs transition"
            >
              <PlayCircle className="w-3.5 h-3.5" />
              <span>Load Demo Repo</span>
            </button>
          )}
        </div>

        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-3 py-2 rounded-md text-xs font-medium transition ${
                    isActive
                      ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30'
                      : 'text-slate-400 hover:bg-[#21262D] hover:text-slate-200'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>

      <div className="p-2 bg-[#0D1117] rounded-md border border-[#30363D] text-[11px] text-slate-400">
        <p className="font-mono text-slate-300 font-medium mb-1">RepoLens AI v1.0</p>
        <p className="text-[10px] leading-relaxed text-slate-500">
          AST-aware Code Chunker & Vector Search for GitHub Repositories.
        </p>
      </div>
    </aside>
  );
};
