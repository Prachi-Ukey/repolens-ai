import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Code2, UserPlus, AlertCircle, Loader2 } from 'lucide-react';

export const RegisterPage = () => {
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await register(email, username, password);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0D1117] flex items-center justify-center p-4 font-mono">
      <div className="w-full max-w-md bg-[#161B22] border border-[#30363D] rounded-2xl p-8 shadow-2xl space-y-6">
        <div className="text-center space-y-2">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center mx-auto shadow-lg shadow-blue-500/20">
            <Code2 className="w-7 h-7 text-white" />
          </div>
          <h2 className="text-xl font-bold text-white tracking-wide">Create Developer Account</h2>
          <p className="text-xs text-slate-400 font-sans">Get started analyzing GitHub codebases with AI.</p>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          <div>
            <label className="block text-slate-400 mb-1">Email Address</label>
            <input
              type="email"
              required
              placeholder="developer@repolens.ai"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-[#0D1117] border border-[#30363D] focus:border-blue-500 rounded-lg px-3.5 py-2.5 text-slate-100 placeholder-slate-500 outline-none transition"
            />
          </div>

          <div>
            <label className="block text-slate-400 mb-1">Username</label>
            <input
              type="text"
              required
              placeholder="dev_user"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-[#0D1117] border border-[#30363D] focus:border-blue-500 rounded-lg px-3.5 py-2.5 text-slate-100 placeholder-slate-500 outline-none transition"
            />
          </div>

          <div>
            <label className="block text-slate-400 mb-1">Password</label>
            <input
              type="password"
              required
              minLength={6}
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-[#0D1117] border border-[#30363D] focus:border-blue-500 rounded-lg px-3.5 py-2.5 text-slate-100 placeholder-slate-500 outline-none transition"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded-lg transition shadow-md shadow-blue-600/20 flex items-center justify-center space-x-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <UserPlus className="w-4 h-4" />}
            <span>{loading ? 'Creating Account...' : 'Register'}</span>
          </button>
        </form>

        <p className="text-center text-xs text-slate-500 font-sans">
          Already have an account?{' '}
          <Link to="/login" className="text-blue-400 hover:underline font-mono">
            Sign In
          </Link>
        </p>
      </div>
    </div>
  );
};
