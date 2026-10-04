import React from 'react';
import { Link } from 'react-router-dom';
import { Code2, Sparkles, ShieldCheck, Cpu, ArrowRight, Layers, FileCode2, Terminal, CheckCircle2 } from 'lucide-react';

export const LandingPage = () => {
  return (
    <div className="min-h-screen bg-[#0D1117] text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="h-16 border-b border-[#30363D] px-6 flex items-center justify-between sticky top-0 bg-[#0D1117]/80 backdrop-blur-md z-40">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Code2 className="w-5 h-5 text-white" />
          </div>
          <span className="text-lg font-bold font-mono">RepoLens<span className="text-blue-400">.AI</span></span>
        </div>

        <div className="flex items-center space-x-4">
          <Link to="/login" className="text-xs font-mono text-slate-300 hover:text-white transition">Sign In</Link>
          <Link
            to="/register"
            className="bg-blue-600 hover:bg-blue-500 text-white text-xs font-mono px-4 py-2 rounded-lg transition shadow-md shadow-blue-600/20"
          >
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main className="flex-1 max-w-6xl mx-auto px-6 py-16 flex flex-col items-center text-center">
        <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-mono mb-6">
          <Sparkles className="w-4 h-4" />
          <span>AST-Aware RAG Codebase Assistant</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight max-w-4xl text-white font-mono leading-tight mb-6">
          Understand Any Codebase with <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400">AI</span>
        </h1>

        <p className="text-slate-400 text-base sm:text-lg max-w-2xl leading-relaxed mb-8">
          Connect a GitHub repository and ask questions about its architecture, code, dependencies, and implementation.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4 mb-16">
          <Link
            to="/register"
            className="bg-blue-600 hover:bg-blue-500 text-white font-mono text-sm px-6 py-3 rounded-xl font-medium flex items-center space-x-2 shadow-xl shadow-blue-600/25 transition"
          >
            <span>Analyze Repository</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            to="/login"
            className="bg-[#161B22] hover:bg-[#21262D] text-slate-200 border border-[#30363D] font-mono text-sm px-6 py-3 rounded-xl font-medium transition"
          >
            View Demo
          </Link>
        </div>

        {/* Visual Interface Preview Mockup */}
        <div className="w-full max-w-5xl bg-[#161B22] border border-[#30363D] rounded-2xl p-2 shadow-2xl overflow-hidden mb-20">
          <div className="bg-[#0D1117] rounded-xl border border-[#21262D] overflow-hidden text-left font-mono text-xs">
            <div className="h-9 bg-[#161B22] border-b border-[#30363D] px-4 flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500/80" />
              <div className="w-3 h-3 rounded-full bg-amber-500/80" />
              <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
              <span className="text-slate-400 ml-4 font-mono text-[11px]">expressjs/express — RepoLens Workspace</span>
            </div>

            <div className="grid grid-cols-12 h-80">
              {/* File Tree */}
              <div className="col-span-3 border-r border-[#30363D] p-3 space-y-2 bg-[#161B22]">
                <div className="text-slate-500 text-[10px] uppercase font-bold">Files</div>
                <div className="text-blue-400 flex items-center gap-1.5"><FileCode2 className="w-3.5 h-3.5" /> lib/express.js</div>
                <div className="text-slate-400 pl-3 flex items-center gap-1.5"><FileCode2 className="w-3.5 h-3.5" /> lib/router/index.js</div>
                <div className="text-slate-400 pl-3 flex items-center gap-1.5"><FileCode2 className="w-3.5 h-3.5" /> lib/middleware/init.js</div>
              </div>

              {/* Chat preview */}
              <div className="col-span-9 p-4 space-y-3 bg-[#0D1117] flex flex-col justify-between">
                <div className="space-y-3">
                  <div className="bg-[#161B22] p-3 rounded-lg border border-[#30363D]">
                    <p className="text-blue-400 font-bold mb-1">User Question:</p>
                    <p className="text-slate-200">Where is router middleware initialized in this repository?</p>
                  </div>
                  <div className="bg-[#161B22]/60 p-3 rounded-lg border border-[#30363D] space-y-2">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-emerald-400" />
                      <span className="text-emerald-400 font-bold">Grounded RAG Answer:</span>
                    </div>
                    <p className="text-slate-300 text-[11px] leading-relaxed">
                      Router middleware is created in <span className="text-blue-400 underline font-mono">lib/express.js</span> at lines 42–68 during application setup.
                    </p>
                  </div>
                </div>

                <div className="bg-[#161B22] p-2.5 rounded-lg border border-[#30363D] text-slate-500">
                  Ask questions about code structure, auth flow, security...
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Feature Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full text-left font-mono">
          <div className="bg-[#161B22] border border-[#30363D] p-6 rounded-2xl space-y-3">
            <div className="p-3 w-fit rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <Cpu className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Understand Your Codebase</h3>
            <p className="text-xs text-slate-400 leading-relaxed font-sans">
              AI analyzes full repository tree structure, source code, and AST definitions.
            </p>
          </div>

          <div className="bg-[#161B22] border border-[#30363D] p-6 rounded-2xl space-y-3">
            <div className="p-3 w-fit rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Source-Aware Answers</h3>
            <p className="text-xs text-slate-400 leading-relaxed font-sans">
              Every technical answer cites exact file paths and line ranges with line-level code viewing.
            </p>
          </div>

          <div className="bg-[#161B22] border border-[#30363D] p-6 rounded-2xl space-y-3">
            <div className="p-3 w-fit rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Architecture Insights</h3>
            <p className="text-xs text-slate-400 leading-relaxed font-sans">
              Visualize how frontend, API routes, database schemas, and auth layers interact.
            </p>
          </div>
        </div>
      </main>

      <footer className="border-t border-[#30363D] py-6 px-6 text-center text-xs font-mono text-slate-500">
        RepoLens AI — Intelligent Developer Codebase Assistant
      </footer>
    </div>
  );
};
