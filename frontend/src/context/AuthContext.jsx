import React, { createContext, useContext, useState, useEffect } from 'react';
import api from '../services/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem('repolens_token') || null);
  const [loading, setLoading] = useState(true);

  // Validate existing stored token once on mount
  useEffect(() => {
    let isMounted = true;
    const initAuth = async () => {
      const storedToken = localStorage.getItem('repolens_token');
      if (storedToken) {
        try {
          const res = await api.get('/auth/me', {
            headers: { Authorization: `Bearer ${storedToken}` },
            timeout: 10000 // 10s bounded timeout
          });
          if (isMounted) {
            setUser(res.data);
            setToken(storedToken);
          }
        } catch (err) {
          console.warn("Stored session token expired or invalid:", err?.message || err);
          if (isMounted) {
            localStorage.removeItem('repolens_token');
            setToken(null);
            setUser(null);
          }
        }
      } else {
        if (isMounted) {
          setUser(null);
          setToken(null);
        }
      }
      if (isMounted) {
        setLoading(false);
      }
    };

    initAuth();
    return () => {
      isMounted = false;
    };
  }, []);

  const login = async (username_or_email, password) => {
    const res = await api.post('/auth/login', { username_or_email, password });
    const newToken = res.data.access_token;
    localStorage.setItem('repolens_token', newToken);
    setToken(newToken);
    const userRes = await api.get('/auth/me', {
      headers: { Authorization: `Bearer ${newToken}` }
    });
    setUser(userRes.data);
    return userRes.data;
  };

  const register = async (email, username, password) => {
    await api.post('/auth/register', { email, username, password });
    return await login(username, password);
  };

  const loginDemo = async () => {
    const demoRes = await api.get('/auth/demo-token');
    const demoToken = demoRes.data.access_token;
    localStorage.setItem('repolens_token', demoToken);
    setToken(demoToken);
    const userRes = await api.get('/auth/me', {
      headers: { Authorization: `Bearer ${demoToken}` }
    });
    setUser(userRes.data);
    return userRes.data;
  };

  const logout = () => {
    localStorage.removeItem('repolens_token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, loginDemo, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
