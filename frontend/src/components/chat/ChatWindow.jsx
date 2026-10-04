import React, { useState, useEffect, useRef } from 'react';
import api from '../../services/api';
import { MessageItem } from './MessageItem';
import { QuickActions } from '../repository/QuickActions';
import {
  Send,
  Loader2,
  MessageSquare,
  Bot,
  Plus
} from 'lucide-react';

export const ChatWindow = ({ repoId, onSelectCitation }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
  // Start with a fresh chat whenever a repository is opened.
  setMessages([]);
  setConversationId(null);
  setInput('');
}, [repoId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth'
    });
  }, [messages]);

  // ---------------------------------------------------------
  // Start a completely new conversation
  // ---------------------------------------------------------

  const handleNewChat = () => {
    if (loading) return;

    setMessages([]);
    setConversationId(null);
    setInput('');
  };

  // ---------------------------------------------------------
  // Send message
  // ---------------------------------------------------------

  const handleSendMessage = async (
    textToSend = input
  ) => {
    const questionText = textToSend.trim();

    if (!questionText || loading) {
      return;
    }

    setInput('');
    setLoading(true);

    // Optimistic user message
    const tempUserMsg = {
      id: Date.now().toString(),
      sender: 'user',
      content: questionText
    };

    setMessages(prev => [
      ...prev,
      tempUserMsg
    ]);

    try {
      const res = await api.post(
        `/repositories/${repoId}/chat`,
        {
          conversation_id: conversationId,
          question: questionText,
          top_k: 5
        }
      );

      setConversationId(
        res.data.conversation_id
      );

      const assistantMsg = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        content: res.data.answer,
        sources: res.data.sources
      };

      setMessages(prev => [
        ...prev,
        assistantMsg
      ]);

    } catch (err) {

      console.error(
        "Chat error:",
        err
      );

      const errorMsg = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        content:
          "Sorry, an error occurred while querying the codebase RAG engine. Please check your network connection or API settings."
      };

      setMessages(prev => [
        ...prev,
        errorMsg
      ]);

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#0D1117]">

      {/* ---------------------------------------------------
          Chat Header
      --------------------------------------------------- */}

      <div className="flex items-center justify-between px-4 py-3 bg-[#161B22] border-b border-[#30363D]">

        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-blue-400" />

          <span className="text-xs font-mono font-bold text-slate-200">
            Repository Chat
          </span>
        </div>

        <button
          type="button"
          onClick={handleNewChat}
          disabled={loading}
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-mono text-slate-300 bg-[#0D1117] border border-[#30363D] hover:border-blue-500 hover:text-blue-400 rounded-lg transition disabled:opacity-40"
        >
          <Plus className="w-3.5 h-3.5" />
          New Chat
        </button>

      </div>

      {/* ---------------------------------------------------
          Quick Action Buttons
      --------------------------------------------------- */}

      <QuickActions
        onSelectAction={(q) =>
          handleSendMessage(q)
        }
      />

      {/* ---------------------------------------------------
          Messages Scroll Area
      --------------------------------------------------- */}

      <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono">

        {messages.length === 0 ? (

          <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-500 text-xs">

            <Bot
              className="w-12 h-12 mb-3 text-blue-400 opacity-40 animate-pulse"
            />

            <h4 className="text-base font-bold text-slate-200 mb-1">
              RepoLens AI Developer Assistant
            </h4>

            <p className="max-w-md text-slate-400">
              Ask natural-language questions about this repository architecture, authentication, database setup, or implementation details.
            </p>

          </div>

        ) : (

          messages.map((msg, idx) => (
            <MessageItem
              key={msg.id || idx}
              message={msg}
              onSelectCitation={onSelectCitation}
            />
          ))

        )}

        {loading && (

          <div className="flex items-center space-x-2 text-xs font-mono text-blue-400 p-4 bg-[#161B22] rounded-xl border border-[#30363D]">

            <Loader2 className="w-4 h-4 animate-spin" />

            <span>
              Searching vector embeddings & generating grounded response...
            </span>

          </div>

        )}

        <div ref={messagesEndRef} />

      </div>

      {/* ---------------------------------------------------
          Input Form
      --------------------------------------------------- */}

      <div className="p-3 bg-[#161B22] border-t border-[#30363D]">

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="flex items-center space-x-2 bg-[#0D1117] border border-[#30363D] focus-within:border-blue-500 rounded-xl p-1.5 transition"
        >

          <input
            type="text"
            placeholder="Ask a question about this repository... (e.g., Where is authentication implemented?)"
            value={input}
            onChange={(e) =>
              setInput(e.target.value)
            }
            className="flex-1 bg-transparent border-none px-3 py-2 text-xs text-slate-100 placeholder-slate-500 font-mono outline-none"
          />

          <button
            type="submit"
            disabled={
              !input.trim() || loading
            }
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white p-2 rounded-lg transition shadow-md shadow-blue-600/20"
          >
            <Send className="w-4 h-4" />
          </button>

        </form>

      </div>

    </div>
  );
};