import React, { useState, useMemo } from 'react';
import { 
  Cpu, Database, Shield, Layout, Server, ArrowDown, ArrowRight,
  ZoomIn, ZoomOut, RefreshCw, FileCode, CheckCircle, Brain, 
  Network, Eye, EyeOff
} from 'lucide-react';

export const ArchDiagram = ({ graphData }) => {
  const [zoom, setZoom] = useState(1);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 text-center text-slate-500 font-mono text-xs">
        <Cpu className="w-10 h-10 mb-2 opacity-30 text-blue-400" />
        <p>Analyzing architecture diagram for repository...</p>
      </div>
    );
  }

  const nodes = graphData.nodes || [];
  const edges = graphData.edges || [];

  const getIcon = (type, category) => {
    const c = (category || "").toLowerCase();
    if (c.includes("presentation") || type === "frontend") return Layout;
    if (c.includes("security") || type === "auth") return Shield;
    if (c.includes("data") || type === "database") return Database;
    if (c.includes("ai") || c.includes("ml") || c.includes("model")) return Brain;
    if (c.includes("processing") || c.includes("logic")) return Cpu;
    if (c.includes("api") || c.includes("service")) return Network;
    return Server;
  };

  const getColor = (type, category) => {
    const c = (category || "").toLowerCase();
    if (c.includes("presentation") || type === "frontend") return "blue";
    if (c.includes("security") || type === "auth") return "emerald";
    if (c.includes("data") || type === "database") return "amber";
    if (c.includes("ai") || c.includes("ml")) return "rose";
    if (c.includes("processing") || c.includes("logic")) return "cyan";
    if (c.includes("api") || c.includes("service")) return "purple";
    return "indigo";
  };

  const colorStyles = {
    blue: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
    emerald: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
    amber: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
    rose: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
    cyan: 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400',
    purple: 'bg-purple-500/10 border-purple-500/30 text-purple-400',
    indigo: 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400',
  };

  // Build connection maps for hover highlights
  const relatedNodes = useMemo(() => {
    if (!hoveredNode) return new Set();
    const related = new Set([hoveredNode]);
    edges.forEach(e => {
      if (e.source === hoveredNode) related.add(e.target);
      if (e.target === hoveredNode) related.add(e.source);
    });
    return related;
  }, [hoveredNode, edges]);

  // Group nodes by level
  const levels = {};
  [1, 2, 3, 4, 5, 6, 7].forEach(lvl => levels[lvl] = []);
  
  nodes.forEach(n => {
    const lvl = n.level || 4;
    if (levels[lvl]) levels[lvl].push(n);
    else levels[4].push(n); // fallback
  });

  const levelNames = {
    1: "Presentation / Client",
    2: "Application / Entry Points",
    3: "API & Security",
    4: "Core Services / Modules",
    5: "Business Logic / Processing",
    6: "AI / ML & Domain Models",
    7: "Data & Storage"
  };

  const activeLevels = Object.keys(levels).filter(l => levels[l].length > 0).sort();

  return (
    <div className="h-full flex flex-col bg-[#0D1117] border border-[#30363D] rounded-xl overflow-hidden font-sans">
      
      {/* Top Bar */}
      <div className="h-14 bg-[#161B22] border-b border-[#30363D] px-4 flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3">
          <Cpu className="w-5 h-5 text-blue-400" />
          <div>
            <div className="font-mono text-sm font-bold text-slate-200">Repository Architecture Analyzer</div>
            <div className="text-[10px] text-slate-500 font-mono">Dynamically inferred from code relationships</div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button onClick={() => setZoom(prev => Math.min(prev + 0.1, 1.5))} className="p-1.5 rounded hover:bg-[#21262D] text-slate-400">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button onClick={() => setZoom(prev => Math.max(prev - 0.1, 0.5))} className="p-1.5 rounded hover:bg-[#21262D] text-slate-400">
            <ZoomOut className="w-4 h-4" />
          </button>
          <button onClick={() => setZoom(1)} className="p-1.5 rounded hover:bg-[#21262D] text-slate-400">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        
        {/* Main Graph Area */}
        <div className="flex-1 overflow-auto bg-[#0B0F17] relative p-8 select-none" onClick={() => setSelectedNode(null)}>
          <div 
            className="flex flex-col items-center max-w-6xl mx-auto origin-top transition-transform duration-300 pb-20 space-y-8"
            style={{ transform: `scale(${zoom})` }}
          >
            {activeLevels.map((lvl, index) => {
              const layerNodes = levels[lvl];
              const isLast = index === activeLevels.length - 1;
              
              return (
                <React.Fragment key={lvl}>
                  {/* Layer Container */}
                  <div className="w-full bg-[#161B22] border border-[#30363D] rounded-2xl p-6 shadow-lg">
                    <div className="font-mono text-xs font-bold tracking-widest uppercase text-slate-400 mb-6 border-b border-white/5 pb-2">
                      Level {lvl} — {levelNames[lvl]}
                    </div>
                    
                    <div className="flex flex-wrap justify-center gap-6">
                      {layerNodes.map(node => {
                        const Icon = getIcon(node.type, node.category);
                        const cKey = getColor(node.type, node.category);
                        const cStyle = colorStyles[cKey];
                        
                        // Highlighting logic
                        const isHovered = hoveredNode === node.id;
                        const isRelated = hoveredNode && relatedNodes.has(node.id);
                        const dim = hoveredNode && !isHovered && !isRelated;
                        
                        return (
                          <div
                            key={node.id}
                            onMouseEnter={() => setHoveredNode(node.id)}
                            onMouseLeave={() => setHoveredNode(null)}
                            onClick={(e) => { e.stopPropagation(); setSelectedNode(node); }}
                            className={`p-4 w-64 rounded-xl border transition-all duration-300 cursor-pointer flex items-start space-x-3 
                              ${cStyle} 
                              ${isHovered ? 'scale-110 shadow-xl shadow-black/50 z-10 border-white/50' : ''}
                              ${dim ? 'opacity-20 grayscale scale-95' : 'opacity-100 shadow-md'}
                            `}
                          >
                            <div className="p-2 rounded-lg bg-[#0D1117] shrink-0">
                              <Icon className="w-5 h-5" />
                            </div>
                            <div className="overflow-hidden">
                              <h5 className="font-mono text-sm font-bold text-white truncate" title={node.label}>{node.label}</h5>
                              <span className="text-[10px] text-slate-400 font-sans block mt-1 truncate">{node.category}</span>
                              {node.file && (
                                <span className="text-[9px] text-slate-500 font-mono block mt-1 truncate opacity-70">
                                  {node.file.split('/').pop()} {node.lines ? `:${node.lines}` : ''}
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  
                  {/* Flow Arrow */}
                  {!isLast && (
                    <div className="flex justify-center text-slate-600">
                      <ArrowDown className="w-6 h-6 animate-pulse" />
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Details Panel Sidebar */}
        {selectedNode && (
          <div className="w-80 border-l border-[#30363D] bg-[#161B22] p-6 flex flex-col shrink-0 overflow-y-auto z-20 shadow-2xl">
            <div className="flex items-center space-x-3 mb-6">
              <div className={`p-2 rounded-lg bg-[#0D1117] ${colorStyles[getColor(selectedNode.type, selectedNode.category)]}`}>
                {React.createElement(getIcon(selectedNode.type, selectedNode.category), { className: "w-6 h-6" })}
              </div>
              <div>
                <h3 className="font-mono text-sm font-bold text-white leading-tight">{selectedNode.label}</h3>
                <span className="text-xs text-slate-400">{selectedNode.category}</span>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Details</h4>
                <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-3 text-xs font-mono text-slate-300">
                  <div className="mb-2"><span className="text-slate-500">ID:</span> {selectedNode.id}</div>
                  <div className="mb-2"><span className="text-slate-500">Type:</span> {selectedNode.type}</div>
                  <div className="mb-2"><span className="text-slate-500">File:</span> {selectedNode.file || "N/A"}</div>
                </div>
              </div>

              {selectedNode.functions && selectedNode.functions.length > 0 && (
                <div>
                  <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Encapsulated Logic</h4>
                  <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-3 text-xs font-mono text-slate-300">
                    <ul className="list-disc list-inside space-y-1">
                      {selectedNode.functions.map((f, i) => (
                        <li key={i} className="text-slate-400">{f}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* Connections List */}
              <div>
                <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2">Relationships</h4>
                
                <div className="space-y-2">
                  {edges.filter(e => e.source === selectedNode.id).length > 0 && (
                    <div className="bg-blue-900/10 border border-blue-500/20 rounded-lg p-3">
                      <div className="text-xs font-bold text-blue-400 mb-2 flex items-center"><ArrowRight className="w-3 h-3 mr-1"/> Outgoing (Calls/Uses)</div>
                      <ul className="text-xs space-y-1">
                        {edges.filter(e => e.source === selectedNode.id).map((e, idx) => {
                          const targetNode = nodes.find(n => n.id === e.target);
                          return (
                            <li key={idx} className="flex items-center text-slate-300">
                              <span className="text-slate-500 mr-2 text-[10px] font-mono">[{e.label}]</span>
                              {targetNode ? targetNode.label : e.target}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}

                  {edges.filter(e => e.target === selectedNode.id).length > 0 && (
                    <div className="bg-amber-900/10 border border-amber-500/20 rounded-lg p-3">
                      <div className="text-xs font-bold text-amber-400 mb-2 flex items-center"><ArrowRight className="w-3 h-3 mr-1 rotate-180"/> Incoming (Called By)</div>
                      <ul className="text-xs space-y-1">
                        {edges.filter(e => e.target === selectedNode.id).map((e, idx) => {
                          const sourceNode = nodes.find(n => n.id === e.source);
                          return (
                            <li key={idx} className="flex items-center text-slate-300">
                              <span className="text-slate-500 mr-2 text-[10px] font-mono">[{e.label}]</span>
                              {sourceNode ? sourceNode.label : e.source}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                  
                  {edges.filter(e => e.source === selectedNode.id || e.target === selectedNode.id).length === 0 && (
                    <div className="text-xs text-slate-500 italic p-3 bg-[#0D1117] rounded-lg border border-[#30363D]">
                      No direct relationships detected in AST.
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            <button 
              onClick={() => setSelectedNode(null)}
              className="mt-6 w-full py-2 bg-[#21262D] hover:bg-[#30363D] text-slate-300 rounded-md text-xs font-bold transition-colors"
            >
              Close Details
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
