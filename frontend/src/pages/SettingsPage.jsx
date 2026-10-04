import React from 'react';
import { Navbar } from '../components/layout/Navbar';
import { Sidebar } from '../components/layout/Sidebar';
import { useAuth } from '../context/AuthContext';
import { Settings, Shield, Key, Database, Cpu, CheckCircle2 } from 'lucide-react';

export const SettingsPage = () => {
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-[#0D1117] text-slate-100 flex flex-col font-sans">
      <Navbar />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar />

        <main className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl font-mono text-xs">
          <div className="border-b border-[#30363D] pb-4">
            <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
              <Settings className="w-5 h-5 text-blue-400" />
              <span>Application Settings & Configuration</span>
            </h1>
            <p className="text-slate-400 font-sans mt-1">
              Manage developer account credentials and inspect RAG embedding configuration.
            </p>
          </div>

          {/* User Account Info */}
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-5 space-y-3">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <Shield className="w-4 h-4 text-emerald-400" />
              <span>Developer Account Profile</span>
            </h4>
            <div className="grid grid-cols-2 gap-4 text-slate-300">
              <div>
                <span className="text-slate-500 block text-[11px]">Username</span>
                <span className="font-bold text-blue-400">{user?.username}</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[11px]">Email</span>
                <span className="font-bold text-slate-200">{user?.email}</span>
              </div>
            </div>
          </div>

          {/* RAG & AI Configuration */}
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-5 space-y-4">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <span>AI Provider & Embeddings Settings</span>
            </h4>

            <div className="space-y-3">
              <div className="p-3 bg-[#0D1117] rounded-lg border border-[#30363D] flex items-center justify-between">
                <div>
                  <p className="font-bold text-slate-200">LLM Provider</p>
                  <p className="text-[11px] text-slate-500 font-sans">Configured via backend .env (`LLM_PROVIDER`)</p>
                </div>
                <span className="px-2.5 py-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 uppercase font-bold">
                  Mock / OpenAI / Gemini
                </span>
              </div>

              <div className="p-3 bg-[#0D1117] rounded-lg border border-[#30363D] flex items-center justify-between">
                <div>
                  <p className="font-bold text-slate-200">Embedding Vector Storage</p>
                  <p className="text-[11px] text-slate-500 font-sans">ChromaDB Isolated Repository Namespace Collections</p>
                </div>
                <span className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
                  ChromaDB Native
                </span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-300 text-[11px] leading-relaxed font-sans">
              To configure your external OpenAI or Gemini API keys, edit <code className="text-white font-mono">backend/.env</code> and restart the backend server.
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};
