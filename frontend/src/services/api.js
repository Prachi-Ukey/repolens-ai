import axios from 'axios';

let apiBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

// Automatic fallback for Vercel deployment if environment variable is not configured
if (!apiBase && typeof window !== 'undefined' && window.location.hostname.includes('vercel.app')) {
  apiBase = 'https://repolens-backend-7ui1.onrender.com';
}

const baseURL = apiBase ? `${apiBase}` : '/api';

const api = axios.create({
  baseURL,
  timeout: 30000, // 30-second bounded request timeout
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

    // Enhance timeout and network error messages for better user experience
    if (error.code === 'ECONNABORTED' || (error.message && error.message.toLowerCase().includes('timeout'))) {
      error.message = 'The server took too long to respond. The backend may be spinning up from idle state. Please try again.';
      if (error.response?.data) {
        error.response.data.detail = error.message;
      }
    } else if (!error.response && (error.code === 'ERR_NETWORK' || error.message === 'Network Error')) {
      error.message = 'Unable to reach the backend server. The server may be booting or unreachable. Please try again in a few moments.';
      if (error.response?.data) {
        error.response.data.detail = error.message;
      }
    }

    return Promise.reject(error);
  }
);

export default api;
