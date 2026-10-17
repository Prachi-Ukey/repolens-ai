import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { Navbar } from '../components/layout/Navbar';
import { Sidebar } from '../components/layout/Sidebar';
import { StatCard } from '../components/dashboard/StatCard';
import { RepositoryCard } from '../components/dashboard/RepositoryCard';
import { AddRepoModal } from '../components/dashboard/AddRepoModal';
import { FolderGit2, FileCode2, Layers, MessageSquare, Plus, PlayCircle, Loader2 } from 'lucide-react';

export const DashboardPage = () => {
  const [repositories, setRepositories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [modalInitialRepoId, setModalInitialRepoId] = useState(null);
  const [modalInitialUrl, setModalInitialUrl] = useState('');
  const navigate = useNavigate();
  const pollTimerRef = useRef(null);

  const fetchRepositories = async (isBackground = false) => {
    if (!isBackground) setLoading(true);
    try {
      const res = await api.get('/repositories');
      setRepositories(res.data || []);

      // Check for an ongoing analysis session to restore
      const activeRepoId = sessionStorage.getItem('repolens_active_analysis_repo_id');
      if (activeRepoId && res.data) {
        const activeRepo = res.data.find(r => r.id === activeRepoId);
        if (activeRepo && (activeRepo.status === 'analyzing' || activeRepo.status === 'pending')) {
          setModalInitialRepoId(activeRepo.id);
          setModalInitialUrl(activeRepo.url);
          setIsAddModalOpen(true);
        } else if (activeRepo && activeRepo.status === 'completed') {
          sessionStorage.removeItem('repolens_active_analysis_repo_id');
        }
      }
    } catch (err) {
      console.error("Error loading repositories:", err);
    } finally {
      if (!isBackground) setLoading(false);
    }
  };

  useEffect(() => {
    fetchRepositories();
  }, []);

  // Periodic background check if any repository is currently analyzing
  useEffect(() => {
    const hasActiveAnalyzing = repositories.some(r => r.status === 'analyzing' || r.status === 'pending');
    if (hasActiveAnalyzing) {
      pollTimerRef.current = setTimeout(() => {
        fetchRepositories(true);
      }, 4000);
    }
    return () => {
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    };
  }, [repositories]);

  const handleLoadDemo = async () => {
    setLoading(true);
    try {
      const res = await api.post('/demo/load-sample');
      await fetchRepositories();
      if (res.data.repository_id) {
        navigate(`/repositories/${res.data.repository_id}`);
      }
    } catch (err) {
      console.error("Error loading demo repository:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteRepo = async (repoId) => {
    try {
      await api.delete(`/repositories/${repoId}`);
      setRepositories(prev => prev.filter(r => r.id !== repoId));
      if (sessionStorage.getItem('repolens_active_analysis_repo_id') === repoId) {
        sessionStorage.removeItem('repolens_active_analysis_repo_id');
      }
    } catch (err) {
      console.error("Error deleting repository:", err);
    }
  };

  const handleOpenAddModal = (repo = null) => {
    if (repo) {
      setModalInitialRepoId(repo.id);
      setModalInitialUrl(repo.url);
    } else {
      setModalInitialRepoId(null);
      setModalInitialUrl('');
    }
    setIsAddModalOpen(true);
  };

  const handleCloseAddModal = () => {
    setIsAddModalOpen(false);
    setModalInitialRepoId(null);
    setModalInitialUrl('');
    fetchRepositories(true);
  };

  // Aggregated Stats
  const totalRepos = repositories.length;
  const totalFiles = repositories.reduce((sum, r) => sum + (r.file_count || 0), 0);
  const totalChunks = repositories.reduce((sum, r) => sum + (r.chunk_count || 0), 0);

  return (
    <div className="min-h-screen bg-[#0D1117] text-slate-100 flex flex-col font-sans">
      <Navbar />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar onOpenAddModal={() => handleOpenAddModal()} onLoadDemo={handleLoadDemo} />

        <main className="flex-1 overflow-y-auto p-6 space-y-8">
          {/* Top Header */}
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#30363D] pb-6">
            <div>
              <h1 className="text-2xl font-bold font-mono text-white tracking-tight">Developer Dashboard</h1>
              <p className="text-xs text-slate-400 font-sans mt-1">
                Analyze public GitHub repositories with AST-aware RAG vector search.
              </p>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={handleLoadDemo}
                className="bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 px-4 py-2 rounded-xl text-xs font-mono font-medium flex items-center space-x-2 transition"
              >
                <PlayCircle className="w-4 h-4" />
                <span>Load Sample Demo Repo</span>
              </button>

              <button
                onClick={() => handleOpenAddModal()}
                className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-xl text-xs font-mono font-medium flex items-center space-x-2 shadow-lg shadow-blue-600/20 transition"
              >
                <Plus className="w-4 h-4" />
                <span>Analyze New Repo</span>
              </button>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard title="Total Repositories" value={totalRepos} icon={FolderGit2} color="blue" description="Connected GitHub Repos" />
            <StatCard title="Total Files Analyzed" value={totalFiles} icon={FileCode2} color="indigo" description="Parsed Source Files" />
            <StatCard title="Total Code Chunks" value={totalChunks} icon={Layers} color="emerald" description="Indexed Vector Vectors" />
            <StatCard title="RAG Engine Status" value="Active" icon={MessageSquare} color="purple" description="ChromaDB Vector Store" />
          </div>

          {/* Repositories Section */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold font-mono text-white">Analyzed Repositories</h3>
              <span className="text-xs text-slate-500 font-mono">{repositories.length} Repositories</span>
            </div>

            {loading ? (
              <div className="flex items-center justify-center p-12 text-slate-500 font-mono text-xs space-x-2">
                <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
                <span>Loading repositories...</span>
              </div>
            ) : repositories.length === 0 ? (
              <div className="bg-[#161B22] border border-[#30363D] rounded-2xl p-12 text-center text-slate-400 font-mono text-xs space-y-4">
                <FolderGit2 className="w-12 h-12 text-blue-400 mx-auto opacity-40" />
                <div>
                  <h4 className="text-sm font-bold text-slate-200">No Repositories Analyzed Yet</h4>
                  <p className="text-slate-500 font-sans mt-1">Connect a public GitHub repository URL or load our sample dataset to start.</p>
                </div>
                <div className="flex justify-center gap-3 pt-2">
                  <button
                    onClick={handleLoadDemo}
                    className="bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 px-4 py-2 rounded-lg transition"
                  >
                    Load Sample Demo
                  </button>
                  <button
                    onClick={() => handleOpenAddModal()}
                    className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg transition shadow-md shadow-blue-600/20"
                  >
                    Add GitHub URL
                  </button>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                {repositories.map(repo => (
                  <RepositoryCard
                    key={repo.id}
                    repo={repo}
                    onDelete={handleDeleteRepo}
                    onViewProgress={(r) => handleOpenAddModal(r)}
                  />
                ))}
              </div>
            )}
          </div>
        </main>
      </div>

      <AddRepoModal
        isOpen={isAddModalOpen}
        initialRepoId={modalInitialRepoId}
        initialUrl={modalInitialUrl}
        onClose={handleCloseAddModal}
        onSuccess={(repoId) => {
          fetchRepositories();
          navigate(`/repositories/${repoId}`);
        }}
      />
    </div>
  );
};
