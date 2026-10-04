import React from 'react';
import { AlertTriangle, Info, ShieldAlert, CheckCircle2 } from 'lucide-react';

const SEVERITY_CONFIG = {
  high: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', icon: ShieldAlert, badge: 'High Severity' },
  medium: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', icon: AlertTriangle, badge: 'Medium Severity' },
  low: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', icon: Info, badge: 'Low Severity' },
};

export const CodeInsightsCard = ({ insights, onSelectFile }) => {
  if (!insights || insights.length === 0) {
    return (
      <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-8 text-center text-slate-400 font-mono text-xs">
        <CheckCircle2 className="w-10 h-10 text-emerald-400 mx-auto mb-2 opacity-80" />
        <p className="font-bold text-slate-200 text-sm">Clean Codebase Audit</p>
        <p className="text-slate-500 mt-1">No major code quality or security concerns detected in analyzed files.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3 font-mono">
      {insights.map((item, idx) => {
        const config = SEVERITY_CONFIG[item.severity] || SEVERITY_CONFIG.medium;
        const Icon = config.icon;

        return (
          <div key={idx} className={`p-4 rounded-xl border ${config.bg} ${config.border} space-y-2`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Icon className={`w-4 h-4 ${config.text}`} />
                <h5 className="text-xs font-bold text-white">{item.title}</h5>
              </div>
              <span className={`text-[10px] px-2 py-0.5 rounded ${config.bg} ${config.text} border ${config.border}`}>
                {config.badge}
              </span>
            </div>

            <p className="text-xs text-slate-300 font-sans">{item.explanation}</p>

            <div className="flex items-center justify-between pt-2 border-t border-[#30363D]/40 text-[11px]">
              <button
                onClick={() => onSelectFile(item.file_path, item.line_number)}
                className="text-blue-400 hover:underline font-mono"
              >
                {item.file_path}:{item.line_number}
              </button>
              <span className="text-slate-400 italic font-sans">{item.suggestion}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
};
