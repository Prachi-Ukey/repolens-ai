import React from 'react';
import { Sparkles, Shield, Cpu, Database, Network, Key, FileText, Lightbulb, Compass } from 'lucide-react';

const ACTIONS = [
  { label: 'Explain Project', query: 'What is this project about and what are its key capabilities?', icon: Compass },
  { label: 'Explain Architecture', query: 'Explain the project architecture and main layers.', icon: Cpu },
  { label: 'Find Authentication', query: 'Where is authentication implemented and how does it work?', icon: Key },
  { label: 'Find Database Logic', query: 'Which files handle database connection and ORM schemas?', icon: Database },
  { label: 'Find API Endpoints', query: 'Where are the main API endpoints defined?', icon: Network },
  { label: 'Find Entry Point', query: 'Identify the main entry point files for frontend and backend.', icon: Sparkles },
  { label: 'Find Security Risks', query: 'What are potential security issues or sensitive configurations in this codebase?', icon: Shield },
  { label: 'Suggest Refactors', query: 'Suggest architectural and code quality improvements for this repository.', icon: Lightbulb },
];

export const QuickActions = ({ onSelectAction }) => {
  return (
    <div className="flex items-center gap-2 overflow-x-auto py-2 px-4 bg-[#161B22] border-b border-[#30363D] no-scrollbar select-none">
      <span className="text-[11px] font-mono text-slate-500 flex items-center gap-1 shrink-0">
        <Sparkles className="w-3 h-3 text-blue-400" /> Quick Actions:
      </span>
      {ACTIONS.map((act) => {
        const Icon = act.icon;
        return (
          <button
            key={act.label}
            onClick={() => onSelectAction(act.query)}
            className="shrink-0 bg-[#0D1117] hover:bg-blue-600/15 hover:text-blue-300 hover:border-blue-500/40 text-slate-300 border border-[#30363D] px-2.5 py-1 rounded-full text-xs font-mono flex items-center space-x-1.5 transition"
          >
            <Icon className="w-3 h-3 text-blue-400" />
            <span>{act.label}</span>
          </button>
        );
      })}
    </div>
  );
};
