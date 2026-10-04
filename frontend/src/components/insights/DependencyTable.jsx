import React from 'react';
import { Package, Layers } from 'lucide-react';

export const DependencyTable = ({ data }) => {
  if (!data || !data.dependencies || data.dependencies.length === 0) {
    return (
      <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-6 text-center text-slate-500 font-mono text-xs">
        <Package className="w-8 h-8 mx-auto mb-2 opacity-30 text-blue-400" />
        <p>No package manifest files (package.json, requirements.txt) discovered.</p>
      </div>
    );
  }

  return (
    <div className="bg-[#161B22] border border-[#30363D] rounded-xl overflow-hidden font-mono text-xs">
      <div className="p-3 bg-[#0D1117] border-b border-[#30363D] flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Package className="w-4 h-4 text-blue-400" />
          <span className="font-bold text-slate-200">Discovered Package Dependencies ({data.total_dependencies})</span>
        </div>
        <div className="flex gap-1">
          {data.ecosystems_found.map(eco => (
            <span key={eco} className="text-[10px] uppercase px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              {eco}
            </span>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[#30363D] bg-[#161B22] text-slate-400 text-[11px]">
              <th className="p-3 font-semibold">Package Name</th>
              <th className="p-3 font-semibold">Version</th>
              <th className="p-3 font-semibold">Type</th>
              <th className="p-3 font-semibold">Ecosystem</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#21262D]">
            {data.dependencies.map((dep, idx) => (
              <tr key={idx} className="hover:bg-[#0D1117] text-slate-300">
                <td className="p-3 font-bold text-blue-300">{dep.name}</td>
                <td className="p-3 text-slate-400">{dep.version}</td>
                <td className="p-3">
                  <span className={`text-[10px] px-2 py-0.5 rounded ${
                    dep.type === 'production' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-slate-700 text-slate-300'
                  }`}>
                    {dep.type}
                  </span>
                </td>
                <td className="p-3 text-slate-500 uppercase">{dep.ecosystem}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
