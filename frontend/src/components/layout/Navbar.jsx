import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { Code2, LogOut, User as UserIcon, Sparkles } from 'lucide-react';

export const Navbar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="h-14 bg-[#161B22] border-b border-[#30363D] px-4 flex items-center justify-between text-sm sticky top-0 z-40">
      <div className="flex items-center space-x-3">
        <Link to="/" className="flex items-center space-x-2.5 font-bold text-white tracking-wide">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Code2 className="w-5 h-5 text-white" />
          </div>
          <span className="text-base font-mono">RepoLens<span className="text-blue-400">.AI</span></span>
        </Link>
        <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono flex items-center gap-1">
          <Sparkles className="w-3 h-3" /> Grounded RAG
        </span>
      </div>

      <div className="flex items-center space-x-4">
        {user ? (
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 bg-[#0D1117] border border-[#30363D] px-3 py-1.5 rounded-md">
              <UserIcon className="w-4 h-4 text-blue-400" />
              <span className="font-mono text-xs text-slate-300">{user.username}</span>
            </div>
            <button
              onClick={handleLogout}
              className="p-1.5 rounded-md hover:bg-red-500/10 hover:text-red-400 text-slate-400 transition"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="flex items-center space-x-3">
            <Link to="/login" className="text-slate-300 hover:text-white transition">Sign In</Link>
            <Link to="/register" className="bg-blue-600 hover:bg-blue-500 text-white px-3.5 py-1.5 rounded-md font-medium transition shadow-md shadow-blue-600/20">
              Get Started
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
};
