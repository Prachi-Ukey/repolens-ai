import axios from 'axios';

const apiBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const baseURL = apiBase ? `${apiBase}` : '/api';

const api = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('repolens_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('repolens_token');
    }
    return Promise.reject(error);
  }
);


export default api;
