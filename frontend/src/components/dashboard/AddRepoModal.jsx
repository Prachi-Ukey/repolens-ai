import React, { useState, useEffect, useRef } from 'react';
import api from '../../services/api';
import { X, GitPullRequest, CheckCircle2, Loader2, AlertCircle, RefreshCw } from 'lucide-react';

const STAGES = [
  { id: 'validating', activeLabel: 'Validating repository URL...', doneLabel: 'Repository validated' },
  { id: 'downloading', activeLabel: 'Cloning repository...', doneLabel: 'Repository downloaded' },
  { id: 'detecting_structure', activeLabel: 'Detecting project structure...', doneLabel: 'Project structure detected' },
  { id: 'parsing_files', activeLabel: 'Processing source code files...', doneLabel: 'Source code files processed' },
  { id: 'chunking', activeLabel: 'Creating AST & code chunks...', doneLabel: 'AST & code chunks created' },
  { id: 'embedding', activeLabel: 'Generating embeddings & indexing in ChromaDB...', doneLabel: 'Embeddings generated & indexed in ChromaDB' },
  { id: 'completed', activeLabel: 'Finalizing analysis...', doneLabel: 'Analysis complete' }
];

export const AddRepoModal = ({ isOpen, onClose, onSuccess, initialRepoId = null, initialUrl = '' }) => {
  const [url, setUrl] = useState(initialUrl || '');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentJob, setCurrentJob] = useState(null);
  const [stageProgress, setStageProgress] = useState({
    stage: '',
    percent: 0,
    message: ''
  });

  const pollTimeoutRef = useRef(null);
  const failureCountRef = useRef(0);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, []);

  // Initialize if initialRepoId is provided
  useEffect(() => {
    if (isOpen && initialRepoId) {
      setCurrentJob({
        repository_id: initialRepoId,
        status: 'processing'
      });
      setLoading(true);
      setError(null);
      setStageProgress({
        stage: 'validating',
        percent: 10,
        message: 'Resuming analysis status tracking...'
      });
      sessionStorage.setItem('repolens_active_analysis_repo_id', initialRepoId);
    } else if (isOpen && initialUrl) {
      setUrl(initialUrl);
    }
  }, [isOpen, initialRepoId, initialUrl]);

  // Non-overlapping polling loop
  useEffect(() => {
    if (!currentJob || currentJob.status !== 'processing') {
      return;
    }

    let isPollingActive = true;

    const pollStatus = async () => {
      if (!isPollingActive || !isMountedRef.current) return;

      try {
        const res = await api.get(`/repositories/${currentJob.repository_id}/analysis-status`);
        if (!isPollingActive || !isMountedRef.current) return;

        failureCountRef.current = 0;
        const statusData = res.data;

        setStageProgress({
          stage: statusData.current_stage || 'validating',
          percent: typeof statusData.progress_percent === 'number' ? statusData.progress_percent : 0,
          message: statusData.message || ''
        });

        if (statusData.status === 'completed') {
          sessionStorage.removeItem('repolens_active_analysis_repo_id');
          setStageProgress(prev => ({
            ...prev,
            stage: 'completed',
            percent: 100,
            message: statusData.message || 'Analysis complete!'
          }));
          pollTimeoutRef.current = setTimeout(() => {
            if (isMountedRef.current) {
              onSuccess(statusData.repository_id);
              handleClose();
            }
          }, 1200);
          return;
        } else if (statusData.status === 'failed') {
          sessionStorage.removeItem('repolens_active_analysis_repo_id');
          setError(statusData.error_details || statusData.message || 'Analysis failed on backend.');
          setLoading(false);
          return;
        }
      } catch (err) {
        if (!isPollingActive || !isMountedRef.current) return;

        console.error("Error polling job status:", err);
        const statusCode = err.response?.status;

        if (statusCode === 401 || statusCode === 403) {
          setError("Your authentication session has expired. Please log in again to monitor repository analysis.");
          setLoading(false);
          sessionStorage.removeItem('repolens_active_analysis_repo_id');
          return;
        }

        failureCountRef.current += 1;
        if (failureCountRef.current >= 6) {
          setError("Lost connection to backend server after multiple attempts. Please check if the server is running and retry.");
          setLoading(false);
          return;
        }
      }

      // Schedule next poll only AFTER current poll completes
      if (isPollingActive && isMountedRef.current) {
        pollTimeoutRef.current = setTimeout(pollStatus, 1500);
      }
    };

    pollStatus();

    return () => {
      isPollingActive = false;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, [currentJob]);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setError(null);
    setLoading(true);
    failureCountRef.current = 0;

    const trimmedUrl = url.trim();
    if (!trimmedUrl) {
      setError("Please provide a valid GitHub repository URL.");
      setLoading(false);
      return;
    }

    try {
      const res = await api.post('/repositories', { url: trimmedUrl });
      const repoId = res.data.repository?.id;
      const jobId = res.data.job_id;

      if (!repoId) {
        throw new Error("Backend response missing repository ID.");
      }

      // If repository is already completed, open workspace immediately without redundant polling
      if (res.data.is_already_completed || res.data.repository?.status === 'completed') {
        sessionStorage.removeItem('repolens_active_analysis_repo_id');
        setStageProgress({
          stage: 'completed',
          percent: 100,
          message: 'Repository already analyzed! Opening workspace...'
        });
        pollTimeoutRef.current = setTimeout(() => {
          if (isMountedRef.current) {
            onSuccess(repoId);
            handleClose();
          }
        }, 500);
        return;
      }

      sessionStorage.setItem('repolens_active_analysis_repo_id', repoId);

      setCurrentJob({
        job_id: jobId,
        repository_id: repoId,
        status: 'processing'
      });
      setStageProgress({
        stage: 'validating',
        percent: 10,
        message: 'Validating GitHub URL...'
      });
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to submit repository.');
      setLoading(false);
      sessionStorage.removeItem('repolens_active_analysis_repo_id');
    }
  };

  const handleClose = () => {
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
    }
    setUrl('');
    setLoading(false);
    setError(null);
    setCurrentJob(null);
    setStageProgress({ stage: '', percent: 0, message: '' });
    onClose();
  };

  const handleRetry = () => {
    setError(null);
    setLoading(false);
    setCurrentJob(null);
    setStageProgress({ stage: '', percent: 0, message: '' });
  };

  if (!isOpen) return null;

  const currentStageIndex = STAGES.findIndex(s => s.id === stageProgress.stage);

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#161B22] border border-[#30363D] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between p-5 border-b border-[#21262D]">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <GitPullRequest className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-white font-mono">
              {loading ? "Repository Ingestion Pipeline" : "Analyze GitHub Repository"}
            </h3>
          </div>
          <button onClick={handleClose} className="p-1 rounded-md text-slate-400 hover:text-white transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6">
          {!loading ? (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-mono text-slate-400 mb-1.5">
                  GitHub Repository URL
                </label>
                <input
                  type="url"
                  required
                  placeholder="https://github.com/username/repository"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] focus:border-blue-500 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 font-mono outline-none transition"
                />
                <p className="text-[11px] text-slate-500 mt-1">
                  Public GitHub repositories are supported (e.g. https://github.com/psf/requests).
                </p>
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-medium">{error}</p>
                  </div>
                </div>
              )}

              <div className="pt-2 flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={handleClose}
                  className="px-4 py-2 rounded-lg text-xs font-medium text-slate-400 hover:text-white transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium px-5 py-2 rounded-lg transition shadow-md shadow-blue-600/20 flex items-center gap-1.5"
                >
                  <span>Start Analysis</span>
                </button>
              </div>
            </form>
          ) : (
            <div className="space-y-5">
              <div className="flex items-center justify-between font-mono text-xs text-slate-300">
                <span className="flex items-center gap-2">
                  {stageProgress.stage === 'completed' ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                  )}
                  {stageProgress.stage === 'completed' ? 'Analysis Complete' : 'Analyzing repository...'}
                </span>
                <span className="text-blue-400 font-bold">{stageProgress.percent}%</span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-2 bg-[#0D1117] rounded-full overflow-hidden border border-[#30363D]">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 transition-all duration-500"
                  style={{ width: `${Math.max(5, stageProgress.percent)}%` }}
                />
              </div>

              {/* Progress timeline */}
              <div className="space-y-2 pt-2">
                {STAGES.map((s, idx) => {
                  const isDone = currentStageIndex > idx || stageProgress.stage === 'completed';
                  const isCurrent = currentStageIndex === idx && stageProgress.stage !== 'completed';

                  return (
                    <div key={s.id} className="flex items-center space-x-3 text-xs font-mono">
                      {isDone ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      ) : isCurrent ? (
                        <Loader2 className="w-4 h-4 text-blue-400 animate-spin shrink-0" />
                      ) : (
                        <div className="w-4 h-4 rounded-full border border-[#30363D] shrink-0" />
                      )}
                      <span className={isDone ? 'text-emerald-300' : isCurrent ? 'text-blue-400 font-bold' : 'text-slate-500'}>
                        {isDone ? s.doneLabel : isCurrent ? s.activeLabel : s.doneLabel}
                      </span>
                    </div>
                  );
                })}
              </div>

              {stageProgress.message && (
                <p className="text-[11px] font-mono text-slate-400 italic text-center pt-2">
                  {stageProgress.message}
                </p>
              )}

              {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center justify-between gap-2 mt-3">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>{error}</span>
                  </div>
                  <button
                    onClick={handleRetry}
                    className="flex items-center gap-1 text-[11px] font-mono px-2 py-1 rounded bg-red-500/20 hover:bg-red-500/30 transition shrink-0"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Retry
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
