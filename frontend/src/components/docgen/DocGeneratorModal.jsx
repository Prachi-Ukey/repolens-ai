import React, { useState } from 'react';
import api from '../../services/api';
import { FileText, Download, Copy, Check, Loader2, X, Sparkles } from 'lucide-react';

const DOC_TYPES = [
  { id: 'readme', label: 'README.md', desc: 'Standard project landing & installation guide' },
  { id: 'api', label: 'API Reference', desc: 'Detailed endpoint documentation & schemas' },
  { id: 'architecture', label: 'Architecture Doc', desc: 'High-level component & system design overview' },
  { id: 'onboarding', label: 'Developer Onboarding', desc: 'Quickstart guide for new developers' },
];

export const DocGeneratorModal = ({ repoId, isOpen, onClose }) => {
  const [selectedType, setSelectedType] = useState('readme');
  const [loading, setLoading] = useState(false);
  const [generatedDoc, setGeneratedDoc] = useState(null);
  const [copied, setCopied] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    setGeneratedDoc(null);
    try {
      const res = await api.post(`/repositories/${repoId}/generate-documentation`, {
        doc_type: selectedType
      });
      setGeneratedDoc(res.data.content);
    } catch (err) {
      console.error("Doc gen error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (generatedDoc) {
      navigator.clipboard.writeText(generatedDoc);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#161B22] border border-[#30363D] rounded-2xl w-full max-w-3xl h-[80vh] flex flex-col shadow-2xl overflow-hidden font-mono">
        <div className="h-14 px-5 border-b border-[#21262D] flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <FileText className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-white">AI Documentation Generator</h3>
          </div>
          <button onClick={onClose} className="p-1 rounded-md text-slate-400 hover:text-white transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Left panel: selection */}
          <div className="w-64 border-r border-[#30363D] p-4 space-y-3 bg-[#0D1117]">
            <span className="text-xs font-bold text-slate-400 block mb-2">Select Document Type:</span>
            {DOC_TYPES.map(doc => (
              <button
                key={doc.id}
                onClick={() => setSelectedType(doc.id)}
                className={`w-full text-left p-3 rounded-lg border text-xs transition ${
                  selectedType === doc.id
                    ? 'bg-blue-600/20 text-blue-400 border-blue-500/40'
                    : 'bg-[#161B22] text-slate-300 border-[#30363D] hover:border-slate-500'
                }`}
              >
                <div className="font-bold">{doc.label}</div>
                <div className="text-[10px] text-slate-500 font-sans mt-0.5">{doc.desc}</div>
              </button>
            ))}

            <button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full mt-4 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white py-2.5 rounded-lg text-xs font-medium flex items-center justify-center space-x-2 transition shadow-md shadow-blue-600/20"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Generate Doc</span>
                </>
              )}
            </button>
          </div>

          {/* Right panel: preview */}
          <div className="flex-1 flex flex-col bg-[#0D1117] overflow-hidden">
            <div className="h-10 bg-[#161B22] border-b border-[#30363D] px-4 flex items-center justify-between text-xs">
              <span className="text-slate-400">Preview: {selectedType.toUpperCase()}.md</span>
              {generatedDoc && (
                <button
                  onClick={handleCopy}
                  className="flex items-center space-x-1.5 px-3 py-1 rounded bg-blue-600/20 text-blue-400 border border-blue-500/30 hover:bg-blue-600 hover:text-white transition"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied!' : 'Copy Markdown'}</span>
                </button>
              )}
            </div>

            <div className="flex-1 p-4 overflow-y-auto font-sans text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">
              {generatedDoc ? (
                generatedDoc
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 font-mono">
                  <FileText className="w-10 h-10 mb-2 opacity-30 text-blue-400" />
                  <p>Select a document type on the left and click "Generate Doc".</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
