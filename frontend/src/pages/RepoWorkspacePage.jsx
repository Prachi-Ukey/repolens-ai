import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../services/api';
import { Navbar } from '../components/layout/Navbar';
import { RepoTreeExplorer } from '../components/repository/RepoTreeExplorer';
import { ChatWindow } from '../components/chat/ChatWindow';
import { CodeViewer } from '../components/repository/CodeViewer';
import { ArchDiagram } from '../components/repository/ArchDiagram';
import { CodeInsightsCard } from '../components/insights/CodeInsightsCard';
import { DependencyTable } from '../components/insights/DependencyTable';
import { DocGeneratorModal } from '../components/docgen/DocGeneratorModal';
import { MessageSquare, Cpu, ShieldAlert, Package, FileText, FolderGit2, Loader2, ArrowLeft } from 'lucide-react';

export const RepoWorkspacePage = () => {
  const { repoId } = useParams();
  const [repo, setRepo] = useState(null);
  const [fileTree, setFileTree] = useState([]);
  const [flatFiles, setFlatFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [highlightedLines, setHighlightedLines] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('chat'); // 'chat', 'architecture', 'insights', 'dependencies'
  const [isDocModalOpen, setIsDocModalOpen] = useState(false);

  // Insights & Architecture data
  const [archGraph, setArchGraph] = useState(null);
  const [insightsData, setInsightsData] = useState([]);
  const [depsData, setDepsData] = useState(null);

  useEffect(() => {
    const loadRepoData = async () => {
      setLoading(true);
      try {
        const repoRes = await api.get(`/repositories/${repoId}`);
        setRepo(repoRes.data);

        const filesRes = await api.get(`/repositories/${repoId}/files`);
        setFileTree(filesRes.data.tree || []);
        setFlatFiles(filesRes.data.flat_files || []);

        if (filesRes.data.flat_files && filesRes.data.flat_files.length > 0) {
          setSelectedFile(filesRes.data.flat_files[0]);
        }

        // Fetch auxiliary data
        const archRes = await api.get(`/repositories/${repoId}/architecture`);
        setArchGraph(archRes.data);

        const insightsRes = await api.get(`/repositories/${repoId}/insights`);
        setInsightsData(insightsRes.data.insights || []);

        const depsRes = await api.get(`/repositories/${repoId}/dependencies`);
        setDepsData(depsRes.data);

      } catch (err) {
        console.error("Error loading workspace data:", err);
      } finally {
        setLoading(false);
      }
    };
    loadRepoData();
  }, [repoId]);

  const handleSelectCitation = async (filePath, startLine, endLine) => {
    // Find file object
    const target = flatFiles.find(f => f.file_path === filePath);
    if (target) {
      // Fetch full content if missing
      if (!target.content) {
        try {
          const res = await api.get(`/repositories/${repoId}/files/${target.id}`);
          setSelectedFile(res.data);
        } catch (e) {
          setSelectedFile(target);
        }
      } else {
        setSelectedFile(target);
      }
      setHighlightedLines({ start: startLine, end: endLine });
    }
  };

  const handleSelectFileFromTree = async (node) => {
    const target = flatFiles.find(f => f.id === node.id || f.file_path === node.path);
    if (target) {
      if (!target.content) {
        try {
          const res = await api.get(`/repositories/${repoId}/files/${target.id}`);
          setSelectedFile(res.data);
        } catch (e) {
          setSelectedFile(target);
        }
      } else {
        setSelectedFile(target);
      }
      setHighlightedLines(null);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0D1117] text-slate-100 flex flex-col font-mono">
        <Navbar />
        <div className="flex-1 flex items-center justify-center p-12 text-slate-500 text-xs space-x-2">
          <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
          <span>Opening repository workspace RAG index...</span>
        </div>
      </div>
    );
  }

  if (!repo) {
    return (
      <div className="min-h-screen bg-[#0D1117] text-slate-100 flex flex-col font-mono">
        <Navbar />
        <div className="flex-1 flex flex-col items-center justify-center p-12 text-slate-400 text-xs space-y-4">
          <FolderGit2 className="w-12 h-12 text-red-400 opacity-50" />
          <p>Repository not found or access denied.</p>
          <Link to="/dashboard" className="text-blue-400 underline">Back to Dashboard</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0D1117] text-slate-100 flex flex-col font-sans overflow-hidden">
      <Navbar />

      {/* Workspace Header Bar */}
      <div className="h-12 bg-[#161B22] border-b border-[#30363D] px-4 flex items-center justify-between text-xs font-mono select-none">
        <div className="flex items-center space-x-3">
          <Link to="/dashboard" className="p-1 rounded hover:bg-[#21262D] text-slate-400 hover:text-white transition">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div className="flex items-center space-x-2">
            <FolderGit2 className="w-4 h-4 text-blue-400" />
            <span className="font-bold text-white">{repo.owner} / {repo.name}</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              {repo.primary_language}
            </span>
          </div>
        </div>

        {/* View Tabs */}
        <div className="flex items-center space-x-1 bg-[#0D1117] p-1 rounded-lg border border-[#30363D]">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-3 py-1 rounded-md text-xs font-medium flex items-center space-x-1.5 transition ${
              activeTab === 'chat' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>AI Workspace Chat</span>
          </button>

          <button
            onClick={() => setActiveTab('architecture')}
            className={`px-3 py-1 rounded-md text-xs font-medium flex items-center space-x-1.5 transition ${
              activeTab === 'architecture' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>Architecture Graph</span>
          </button>

          <button
            onClick={() => setActiveTab('insights')}
            className={`px-3 py-1 rounded-md text-xs font-medium flex items-center space-x-1.5 transition ${
              activeTab === 'insights' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>Code Insights ({insightsData.length})</span>
          </button>

          <button
            onClick={() => setActiveTab('dependencies')}
            className={`px-3 py-1 rounded-md text-xs font-medium flex items-center space-x-1.5 transition ${
              activeTab === 'dependencies' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Package className="w-3.5 h-3.5" />
            <span>Dependencies</span>
          </button>
        </div>

        {/* Documentation Generator Button */}
        <button
          onClick={() => setIsDocModalOpen(true)}
          className="bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 px-3 py-1 rounded-md text-xs font-medium flex items-center space-x-1.5 transition"
        >
          <FileText className="w-3.5 h-3.5" />
          <span>Generate Docs</span>
        </button>
      </div>

      {/* Main Workspace Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Panel 1 (Left): Repository Explorer Tree */}
        <div className="w-64 shrink-0 h-full">
          <RepoTreeExplorer
            tree={fileTree}
            onSelectFile={handleSelectFileFromTree}
            selectedFileId={selectedFile?.id}
          />
        </div>

        {/* Panel 2 (Center) & Tab Content */}
        <div className="flex-1 flex flex-col h-full overflow-hidden">
          {activeTab === 'chat' && (
            <div className="grid grid-cols-12 h-full">
              {/* Center Chat Panel */}
              <div className="col-span-7 h-full border-r border-[#30363D]">
                <ChatWindow repoId={repo.id} onSelectCitation={handleSelectCitation} />
              </div>

              {/* Right Code Viewer Panel */}
              <div className="col-span-5 h-full">
                <CodeViewer
                  file={selectedFile}
                  highlightedLines={highlightedLines}
                  onClose={() => setSelectedFile(null)}
                />
              </div>
            </div>
          )}

          {activeTab === 'architecture' && (
            <div className="p-6 h-full">
              <ArchDiagram graphData={archGraph} />
            </div>
          )}

          {activeTab === 'insights' && (
            <div className="p-6 h-full overflow-y-auto max-w-4xl mx-auto w-full">
              <h3 className="text-base font-bold font-mono text-white mb-4">Repository Code Quality & Risk Audit</h3>
              <CodeInsightsCard
                insights={insightsData}
                onSelectFile={(fPath, line) => {
                  setActiveTab('chat');
                  handleSelectCitation(fPath, line, line);
                }}
              />
            </div>
          )}

          {activeTab === 'dependencies' && (
            <div className="p-6 h-full overflow-y-auto max-w-4xl mx-auto w-full">
              <h3 className="text-base font-bold font-mono text-white mb-4">Package & Framework Dependencies</h3>
              <DependencyTable data={depsData} />
            </div>
          )}
        </div>
      </div>

      <DocGeneratorModal
        repoId={repo.id}
        isOpen={isDocModalOpen}
        onClose={() => setIsDocModalOpen(false)}
      />
    </div>
  );
};
