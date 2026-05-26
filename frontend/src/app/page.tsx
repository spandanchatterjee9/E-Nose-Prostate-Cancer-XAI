'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Activity, Lock, User, AlertTriangle, CheckCircle } from 'lucide-react';
import { API_BASE } from '../config';

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isRegister, setIsRegister] = useState(false);
  const [role, setRole] = useState('clinician'); // clinician or researcher
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    try {
      if (isRegister) {
        // Register API Call
        const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ username, password, role }),
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Registration failed.');
        }

        setSuccess('Account created successfully! Switching to Login...');
        setIsRegister(false);
        setPassword('');
      } else {
        // Login API Call (requires Form URL Encoded body)
        const params = new URLSearchParams();
        params.append('username', username);
        params.append('password', password);

        const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: params,
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Invalid username or password.');
        }

        // Store session tokens
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user_role', data.role);
        localStorage.setItem('username', data.username);

        router.push('/dashboard');
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex items-center justify-center relative bg-[#05070c] px-4 overflow-hidden">
      {/* Visual background gradient lights */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-900/20 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-900/10 rounded-full blur-[100px] pointer-events-none" />

      <div className="w-full max-w-md glass-panel p-8 rounded-2xl border border-slate-800/80 shadow-2xl relative z-10">
        <div className="flex flex-col items-center mb-8">
          <div className="w-16 h-16 bg-indigo-500/10 rounded-2xl flex items-center justify-center border border-indigo-500/20 mb-4 animate-pulse">
            <Activity className="w-8 h-8 text-cyan-400" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white text-center">
            MOOSY-32 Clinical Portal
          </h1>
          <p className="text-sm text-slate-400 mt-2 text-center">
            Explainable AI Prostate Cancer Diagnostic System
          </p>
        </div>

        {error && (
          <div className="bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-lg p-3 flex items-center gap-2 mb-6 text-sm">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 rounded-lg p-3 flex items-center gap-2 mb-6 text-sm">
            <CheckCircle className="w-4 h-4 shrink-0 text-emerald-400" />
            <span>{success}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
              Username
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <User className="w-4 h-4 text-slate-500" />
              </span>
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. clinician_jane"
                className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2.5 pl-10 pr-4 text-sm text-slate-100 placeholder-slate-600 focus:outline-none transition-all"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
              Password
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="w-4 h-4 text-slate-500" />
              </span>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2.5 pl-10 pr-4 text-sm text-slate-100 placeholder-slate-600 focus:outline-none transition-all"
              />
            </div>
          </div>

          {isRegister && (
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                System Role
              </label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2.5 px-3 text-sm text-slate-100 focus:outline-none transition-all"
              >
                <option value="clinician" className="bg-slate-950">Clinician (Diagnosis Dashboard)</option>
                <option value="researcher" className="bg-slate-950">Researcher (Ablation & Benchmarking)</option>
              </select>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 focus:ring-2 focus:ring-indigo-500 focus:outline-none rounded-lg text-sm font-semibold text-white transition-all shadow-lg shadow-indigo-900/30 flex items-center justify-center"
          >
            {loading ? 'Processing...' : isRegister ? 'Register Account' : 'Clinician Sign In'}
          </button>
        </form>

        <div className="mt-6 pt-6 border-t border-slate-900 text-center">
          <button
            onClick={() => {
              setIsRegister(!isRegister);
              setError('');
              setSuccess('');
            }}
            className="text-xs font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
          >
            {isRegister
              ? 'Already registered? Access local workspace'
              : 'Create new local researcher/clinician credential'}
          </button>
        </div>
      </div>
    </main>
  );
}
