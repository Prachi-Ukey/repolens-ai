import React, { useState } from 'react';
import { Cpu, Database, Shield, Layout, Server, ArrowRight, ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';

const CATEGORY_COLORS = {
  frontend: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', icon: Layout },
  backend: { bg: 'bg-indigo-500/10', border: 'border-indigo-500/30', text: 'text-indigo-400', icon: Server },
  auth: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', icon: Shield },
  database: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-400', icon: Database },
};

export const ArchDiagram = ({ graphData }) => {
  const [zoom, setZoom] = useState(1);
  const [selectedNode, setSelectedNode] = useState(null);

  if (!graphData || !graphData.nodes) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 text-center text-slate-500 font-mono text-xs">
        <Cpu className="w-10 h-10 mb-2 opacity-30 text-blue-400" />
        <p>Analyzing architecture diagram for repository...</p>
      </div>
    );
  }

  const nodes = graphData.nodes || [];
  const edges = graphData.edges || [];

  return (
    <div className="h-full flex flex-col bg-[#0D1117] border border-[#30363D] rounded-xl overflow-hidden">
      <div className="h-10 bg-[#161B22] border-b border-[#30363D] px-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Cpu className="w-4 h-4 text-blue-400" />
          <span className="font-mono text-xs font-bold text-slate-200">Architecture Component Graph</span>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setZoom(prev => Math.min(prev + 0.2, 1.8))}
            className="p-1 rounded hover:bg-[#21262D] text-slate-400 hover:text-white"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => setZoom(prev => Math.max(prev - 0.2, 0.6))}
            className="p-1 rounded hover:bg-[#21262D] text-slate-400 hover:text-white"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={() => setZoom(1)}
            className="p-1 rounded hover:bg-[#21262D] text-slate-400 hover:text-white"
            title="Reset Zoom"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-auto relative flex flex-col items-center justify-center bg-[#0B0F17]">
        <div
          className="transition-transform duration-300 flex flex-wrap items-center justify-center gap-8 max-w-4xl"
          style={{ transform: `scale(${zoom})` }}
        >
          {nodes.map((node) => {
            const style = CATEGORY_COLORS[node.type] || CATEGORY_COLORS.backend;
            const Icon = style.icon;

            return (
              <div
                key={node.id}
                onClick={() => setSelectedNode(node)}
                className={`p-4 rounded-xl border ${style.bg} ${style.border} cursor-pointer hover:scale-105 transition shadow-lg w-52 text-left`}
              >
                <div className="flex items-center space-x-2.5 mb-2">
                  <div className={`p-2 rounded-lg bg-[#0D1117] ${style.text}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <h5 className="font-mono text-xs font-bold text-white truncate">{node.label}</h5>
                    <span className="text-[10px] text-slate-400 font-mono">{node.category}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Edges List Legend */}
        <div className="w-full mt-8 pt-4 border-t border-[#21262D] grid grid-cols-2 md:grid-cols-3 gap-3 text-xs font-mono">
          {edges.map((edge, idx) => (
            <div key={idx} className="flex items-center space-x-2 text-slate-400 bg-[#161B22] p-2 rounded border border-[#30363D]">
              <span className="text-blue-400 font-bold">{edge.source}</span>
              <ArrowRight className="w-3.5 h-3.5 text-slate-500 shrink-0" />
              <span className="text-indigo-400 font-bold">{edge.target}</span>
              <span className="text-[10px] text-slate-500 ml-auto">({edge.label})</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
