import React, { createContext, useContext, useState, useEffect } from 'react';
import api from '../services/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('repolens_token') || null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUser = async () => {
      if (token) {
        try {
          const res = await api.get('/auth/me');
          setUser(res.data);
        } catch (err) {
          console.error("Error loading user session:", err);
          logout();
        }
      }
      setLoading(false);
    };
    fetchUser();
  }, [token]);

  const login = async (username_or_email, password) => {
    const res = await api.post('/auth/login', { username_or_email, password });
    const newToken = res.data.access_token;
    localStorage.setItem('repolens_token', newToken);
    setToken(newToken);
    const userRes = await api.get('/auth/me');
    setUser(userRes.data);
    return userRes.data;
  };

  const register = async (email, username, password) => {
    await api.post('/auth/register', { email, username, password });
    return await login(username, password);
  };

  const logout = () => {
    localStorage.removeItem('repolens_token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
