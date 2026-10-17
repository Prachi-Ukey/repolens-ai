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
import { MessageSquare, BookOpen, Cpu, ShieldAlert, Package, FileText, FolderGit2, Loader2, ArrowLeft, ExternalLink, RefreshCw } from 'lucide-react';

export const RepoWorkspacePage = () => {
  const { repoId } = useParams();
  const [repo, setRepo] = useState(null);
  const [repoOverview, setRepoOverview] = useState(null);
  const [fileTree, setFileTree] = useState([]);
  const [flatFiles, setFlatFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [highlightedLines, setHighlightedLines] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('chat'); // 'chat', 'overview', 'architecture', 'insights', 'dependencies'
  const [isDocModalOpen, setIsDocModalOpen] = useState(false);

  // Insights & Architecture data
  const [archGraph, setArchGraph] = useState(null);
  const [insightsData, setInsightsData] = useState([]);
  const [depsData, setDepsData] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [isReanalyzing, setIsReanalyzing] = useState(false);

  const loadRepoData = async () => {
    setLoading(true);
    try {
      const repoRes = await api.get(`/repositories/${repoId}`);
      setRepo(repoRes.data);

      if (repoRes.data.status === 'completed') {
        // Fetch workspace resources concurrently instead of sequential waterfall
        const [filesResult, overviewResult, archResult, insightsResult, depsResult] = await Promise.allSettled([
          api.get(`/repositories/${repoId}/files`),
          api.get(`/repositories/${repoId}/overview`),
          api.get(`/repositories/${repoId}/architecture`),
          api.get(`/repositories/${repoId}/insights`),
          api.get(`/repositories/${repoId}/dependencies`)
        ]);

        if (overviewResult.status === 'fulfilled') {
          setRepoOverview(overviewResult.value.data);
        }

        if (archResult.status === 'fulfilled') {
          setArchGraph(archResult.value.data);
        }

        if (insightsResult.status === 'fulfilled') {
          setInsightsData(insightsResult.value.data.insights || []);
        }

        if (depsResult.status === 'fulfilled') {
          setDepsData(depsResult.value.data);
        }

        if (filesResult.status === 'fulfilled') {
          const files = filesResult.value.data.flat_files || [];
          setFileTree(filesResult.value.data.tree || []);
          setFlatFiles(files);

          // Prefer README file as initial code viewer preview, otherwise first file
          const defaultFile = files.find(f => f.file_path && f.file_path.toLowerCase().includes('readme')) || files[0];
          if (defaultFile) {
            if (!defaultFile.content) {
              try {
                const fileContentRes = await api.get(`/repositories/${repoId}/files/${defaultFile.id}`);
                setSelectedFile(fileContentRes.data);
              } catch (e) {
                setSelectedFile(defaultFile);
              }
            } else {
              setSelectedFile(defaultFile);
            }
          }
        }
      }
    } catch (err) {
      console.error("Error loading workspace data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRepoData();
  }, [repoId]);

  // Live polling if repository is still in analyzing/pending state
  useEffect(() => {
    let timer;
    let isCancelled = false;

    if (repo && (repo.status === 'analyzing' || repo.status === 'pending')) {
      const poll = async () => {
        try {
          const res = await api.get(`/repositories/${repoId}/analysis-status`);
          if (isCancelled) return;
          setAnalysisStatus(res.data);

          if (res.data.status === 'completed') {
            loadRepoData();
            return;
          } else if (res.data.status === 'failed') {
            setRepo(prev => ({ ...prev, status: 'failed' }));
            return;
          }
        } catch (e) {
          console.error("Error polling status in workspace:", e);
        }

        if (!isCancelled) {
          timer = setTimeout(poll, 2000);
        }
      };
      poll();
    }

    return () => {
      isCancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [repo?.status, repoId]);

  const handleReanalyze = async () => {
    if (!repo) return;
    setIsReanalyzing(true);
    try {
      await api.post('/repositories', { url: repo.url, reanalyze: true });
      setRepo(prev => ({ ...prev, status: 'analyzing' }));
      setAnalysisStatus({
        status: 'analyzing',
        current_stage: 'validating',
        progress_percent: 10,
        message: 'Reanalysis started...'
      });
    } catch (err) {
      console.error("Failed to start reanalysis:", err);
    } finally {
      setIsReanalyzing(false);
    }
  };

  const handleSelectCitation = async (filePath, startLine, endLine) => {
    // Find file object
    const target = flatFiles.find(f => f.file_path === filePath || f.file_path.endsWith('/' + filePath) || filePath.endsWith('/' + f.file_path));
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
      setActiveTab('chat');
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
    <div className="h-screen bg-[#0D1117] text-slate-100 flex flex-col font-sans overflow-hidden">
      <Navbar />

      {/* Workspace Header Bar */}
      <div className="h-12 bg-[#161B22] border-b border-[#30363D] px-4 flex items-center justify-between text-xs font-mono select-none shrink-0">
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
            onClick={() => setActiveTab('overview')}
            className={`px-3 py-1 rounded-md text-xs font-medium flex items-center space-x-1.5 transition ${
              activeTab === 'overview' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Project Overview</span>
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

      {/* Real-time Analysis Progress Banner */}
      {(repo.status === 'analyzing' || repo.status === 'pending') && (
        <div className="bg-blue-950/50 border-b border-blue-500/30 px-4 py-2.5 flex items-center justify-between text-xs font-mono text-blue-200 shrink-0">
          <div className="flex items-center space-x-3">
            <Loader2 className="w-4 h-4 text-blue-400 animate-spin shrink-0" />
            <span>
              Analyzing repository: <span className="font-bold text-white">{analysisStatus?.current_stage || 'Processing'}</span>
              {analysisStatus?.message ? ` — ${analysisStatus.message}` : ''}
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-36 h-2 bg-[#0D1117] rounded-full overflow-hidden border border-[#30363D]">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${analysisStatus?.progress_percent || 15}%` }}
              />
            </div>
            <span className="font-bold text-blue-400">{analysisStatus?.progress_percent || 10}%</span>
          </div>
        </div>
      )}

      {/* Analysis Failure Banner */}
      {repo.status === 'failed' && (
        <div className="bg-red-950/50 border-b border-red-500/30 px-4 py-2.5 flex items-center justify-between text-xs font-mono text-red-200 shrink-0">
          <div className="flex items-center space-x-3">
            <ShieldAlert className="w-4 h-4 text-red-400 shrink-0" />
            <span>
              Analysis failed: {analysisStatus?.error_details || 'An error occurred during repository ingestion.'}
            </span>
          </div>
          <button
            onClick={handleReanalyze}
            disabled={isReanalyzing}
            className="px-3 py-1 bg-red-600/30 hover:bg-red-600/50 text-red-200 border border-red-500/40 rounded text-xs transition flex items-center space-x-1"
          >
            {isReanalyzing ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
            <span>Retry Analysis</span>
          </button>
        </div>
      )}

      {/* Main Workspace Body */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Panel 1 (Left): Repository Explorer Tree */}
        <div className="w-64 shrink-0 h-full overflow-hidden">
          <RepoTreeExplorer
            tree={fileTree}
            onSelectFile={handleSelectFileFromTree}
            selectedFileId={selectedFile?.id}
          />
        </div>

        {/* Panel 2 (Center) & Tab Content */}
        <div className="flex-1 min-w-0 min-h-0 flex flex-col h-full overflow-hidden">
          {activeTab === 'chat' && (
            <div className="grid grid-cols-12 h-full min-h-0 overflow-hidden">
              {/* Center Chat Panel */}
              <div className="col-span-7 h-full min-w-0 min-h-0 overflow-hidden border-r border-[#30363D]">
                <ChatWindow repoId={repo.id} onSelectCitation={handleSelectCitation} />
              </div>

              {/* Right Code Viewer Panel */}
              <div className="col-span-5 h-full min-w-0 min-h-0 overflow-hidden">
                <CodeViewer
                  file={selectedFile}
                  highlightedLines={highlightedLines}
                  onClose={() => setSelectedFile(null)}
                />
              </div>
            </div>
          )}

          {activeTab === 'overview' && (
            <div className="p-6 h-full overflow-y-auto max-w-5xl mx-auto w-full space-y-6">
              {/* Overview Header Card */}
              <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-6">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center space-x-3">
                      <FolderGit2 className="w-6 h-6 text-blue-400" />
                      <h2 className="text-xl font-bold font-mono text-white">
                        {repo.owner} / {repo.name}
                      </h2>
                      <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono">
                        {repo.primary_language || 'Codebase'}
                      </span>
                      <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                        {repo.status || 'Indexed'}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 max-w-2xl">
                      {repo.description || `Public GitHub repository ${repo.owner}/${repo.name} analyzed with AST-aware RAG vector search.`}
                    </p>
                  </div>

                  <div className="flex items-center space-x-3">
                    {repo.url && (
                      <a
                        href={repo.url}
                        target="_blank"
                        rel="noreferrer"
                        className="bg-[#21262D] hover:bg-[#30363D] text-slate-300 border border-[#30363D] px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 transition"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                        <span>View on GitHub</span>
                      </a>
                    )}
                    <button
                      onClick={() => setActiveTab('chat')}
                      className="bg-blue-600 hover:bg-blue-500 text-white px-3.5 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 shadow transition"
                    >
                      <MessageSquare className="w-3.5 h-3.5" />
                      <span>Start Chat</span>
                    </button>
                  </div>
                </div>

                {/* Quick Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6 pt-6 border-t border-[#30363D]">
                  <div className="bg-[#0D1117] p-3 rounded-lg border border-[#21262D]">
                    <div className="text-[11px] font-mono text-slate-500">Indexed Files</div>
                    <div className="text-lg font-bold font-mono text-slate-200 mt-1">{repo.file_count || flatFiles.length}</div>
                  </div>
                  <div className="bg-[#0D1117] p-3 rounded-lg border border-[#21262D]">
                    <div className="text-[11px] font-mono text-slate-500">Vector Chunks</div>
                    <div className="text-lg font-bold font-mono text-slate-200 mt-1">{repo.chunk_count || 0}</div>
                  </div>
                  <div className="bg-[#0D1117] p-3 rounded-lg border border-[#21262D]">
                    <div className="text-[11px] font-mono text-slate-500">Primary Language</div>
                    <div className="text-lg font-bold font-mono text-blue-400 mt-1">{repo.primary_language || 'N/A'}</div>
                  </div>
                  <div className="bg-[#0D1117] p-3 rounded-lg border border-[#21262D]">
                    <div className="text-[11px] font-mono text-slate-500">Code Audit Insights</div>
                    <div className="text-lg font-bold font-mono text-emerald-400 mt-1">{insightsData.length}</div>
                  </div>
                </div>
              </div>

              {/* README Preview Card */}
              <div className="bg-[#161B22] border border-[#30363D] rounded-xl overflow-hidden">
                <div className="h-10 bg-[#21262D] px-4 flex items-center justify-between border-b border-[#30363D]">
                  <div className="flex items-center space-x-2">
                    <BookOpen className="w-4 h-4 text-blue-400" />
                    <span className="text-xs font-bold font-mono text-slate-200">README.md Preview</span>
                  </div>
                  <button
                    onClick={() => {
                      const readme = flatFiles.find(f => f.file_path && f.file_path.toLowerCase().includes('readme'));
                      if (readme) {
                        handleSelectFileFromTree({ id: readme.id, path: readme.file_path });
                      }
                      setActiveTab('chat');
                    }}
                    className="text-[11px] font-mono text-blue-400 hover:text-blue-300 transition flex items-center space-x-1"
                  >
                    <span>Open in Code Viewer</span>
                    <span>→</span>
                  </button>
                </div>

                <div className="p-6 overflow-y-auto max-h-[500px] text-xs font-mono text-slate-300 leading-relaxed bg-[#0D1117] whitespace-pre-wrap">
                  {repoOverview?.readme_preview ||
                    flatFiles.find(f => f.file_path && f.file_path.toLowerCase().includes('readme'))?.content ||
                    "No README.md preview available for this repository."}
                </div>
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
