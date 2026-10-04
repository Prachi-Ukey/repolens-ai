import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { X, GitPullRequest, CheckCircle2, Loader2, AlertCircle } from 'lucide-react';

const STAGES = [
  { id: 'validating', label: 'Repository validated' },
  { id: 'downloading', label: 'Repository downloaded' },
  { id: 'detecting_structure', label: 'Project structure detected' },
  { id: 'parsing_files', label: 'Source code files processed' },
  { id: 'chunking', label: 'AST & code chunks created' },
  { id: 'embedding', label: 'Embeddings generated & indexed in ChromaDB' },
  { id: 'completed', label: 'Analysis complete' }
];

export const AddRepoModal = ({ isOpen, onClose, onSuccess }) => {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentJob, setCurrentJob] = useState(null);
  const [stageProgress, setStageProgress] = useState({
    stage: '',
    percent: 0,
    message: ''
  });

  useEffect(() => {
    let interval;
    if (currentJob && currentJob.status === 'processing') {
      interval = setInterval(async () => {
        try {
          const res = await api.get(`/repositories/${currentJob.repository_id}/analysis-status`);
          const statusData = res.data;
          setStageProgress({
            stage: statusData.current_stage,
            percent: statusData.progress_percent,
            message: statusData.message
          });

          if (statusData.status === 'completed') {
            clearInterval(interval);
            setTimeout(() => {
              onSuccess(statusData.repository_id);
              handleClose();
            }, 1000);
          } else if (statusData.status === 'failed') {
            clearInterval(interval);
            setError(statusData.error_details || statusData.message || 'Analysis failed.');
            setLoading(false);
          }
        } catch (err) {
          console.error("Error polling job status:", err);
        }
      }, 1500);
    }

    return () => clearInterval(interval);
  }, [currentJob]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const res = await api.post('/repositories', { url });
      setCurrentJob({
        job_id: res.data.job_id,
        repository_id: res.data.repository.id,
        status: 'processing'
      });
      setStageProgress({
        stage: 'validating',
        percent: 10,
        message: 'Validating GitHub URL...'
      });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to submit repository.');
      setLoading(false);
    }
  };

  const handleClose = () => {
    setUrl('');
    setLoading(false);
    setError(null);
    setCurrentJob(null);
    setStageProgress({ stage: '', percent: 0, message: '' });
    onClose();
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
            <h3 className="text-base font-bold text-white font-mono">Analyze GitHub Repository</h3>
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
                  Public GitHub repositories are supported.
                </p>
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{error}</span>
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
                  className="bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium px-5 py-2 rounded-lg transition shadow-md shadow-blue-600/20"
                >
                  Start Analysis
                </button>
              </div>
            </form>
          ) : (
            <div className="space-y-5">
              <div className="flex items-center justify-between font-mono text-xs text-slate-300">
                <span className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                  Analyzing repository...
                </span>
                <span className="text-blue-400 font-bold">{stageProgress.percent}%</span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-2 bg-[#0D1117] rounded-full overflow-hidden border border-[#30363D]">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 transition-all duration-500"
                  style={{ width: `${stageProgress.percent}%` }}
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
                        {s.label}
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
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
