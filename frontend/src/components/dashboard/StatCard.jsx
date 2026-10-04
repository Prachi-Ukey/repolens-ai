import React from 'react';

export const StatCard = ({ title, value, icon: Icon, description, color = "blue" }) => {
  return (
    <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-4 flex items-center justify-between shadow-sm">
      <div>
        <p className="text-xs font-medium text-slate-400 mb-1">{title}</p>
        <h3 className="text-2xl font-bold font-mono text-white tracking-tight">{value}</h3>
        {description && <p className="text-[11px] text-slate-500 mt-1">{description}</p>}
      </div>
      <div className={`p-3 rounded-lg bg-${color}-500/10 text-${color}-400 border border-${color}-500/20`}>
        <Icon className="w-5 h-5" />
      </div>
    </div>
  );
};
