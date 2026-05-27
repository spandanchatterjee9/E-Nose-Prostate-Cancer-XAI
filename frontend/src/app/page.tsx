'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { 
  Activity, 
  Lock, 
  User, 
  AlertTriangle, 
  CheckCircle, 
  LayoutDashboard, 
  Users, 
  LineChart, 
  ShieldAlert, 
  Cpu, 
  ArrowRight,
  ShieldCheck
} from 'lucide-react';
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
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

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

  const handleBypass = () => {
    // Demo bypass authentication for local testing
    localStorage.setItem('access_token', 'demo_bypass_token');
    localStorage.setItem('user_role', 'clinician');
    localStorage.setItem('username', 'Demo Clinician');
    router.push('/dashboard');
  };

  const placeholderNavItems = [
    { name: 'Diagnosis Overview', icon: LayoutDashboard },
    { name: 'Patient Files', icon: Users },
    { name: 'Predict & Explain', icon: Activity },
    { name: 'Research & Benchmarking', icon: LineChart },
  ];

  if (!mounted) return null;

  return (
    <div className="min-h-screen flex bg-[#030509]">
      {/* Sidebar Placeholder */}
      <aside className="w-64 bg-[#090e18] border-r border-slate-900 flex flex-col justify-between shrink-0">
        <div>
          {/* Sidebar Header */}
          <div className="h-16 flex items-center gap-2.5 px-6 border-b border-slate-900">
            <Activity className="w-6 h-6 text-indigo-400" />
            <span className="font-semibold text-lg tracking-wide text-white">
              MOOSY-32 AI
            </span>
          </div>

          {/* Navigation Links Placeholder */}
          <nav className="mt-6 px-4 space-y-1.5">
            {placeholderNavItems.map((item) => {
              const Icon = item.icon;
              return (
                <div
                  key={item.name}
                  className="flex items-center justify-between px-4 py-2.5 rounded-lg text-sm font-medium text-slate-500 cursor-not-allowed select-none hover:bg-slate-950/20"
                >
                  <div className="flex items-center gap-3">
                    <Icon className="w-4 h-4 text-slate-600" />
                    <span>{item.name}</span>
                  </div>
                  <Lock className="w-3.5 h-3.5 text-slate-700" />
                </div>
              );
            })}
          </nav>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-slate-900">
          <div className="text-center text-[10px] text-slate-600">
            Authentication Required to Access Portal Pages
          </div>
        </div>
      </aside>

      {/* Main Workspace Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header Bar */}
        <header className="h-16 bg-[#090e18]/45 border-b border-slate-900/60 flex items-center justify-between px-8 relative z-20">
          <div className="flex items-center gap-3">
            <div className="h-2 w-2 rounded-full bg-slate-500 animate-pulse" />
            <span className="text-xs font-semibold text-slate-400">
              MOOSY-32 Sensor Array Status:
            </span>
            <span className="text-xs font-bold text-slate-500">
              AWAITING SESSION / OFFLINE
            </span>
          </div>
          
          <div className="text-xs text-slate-500">
            Clinical System Workspace v1.0.0
          </div>
        </header>

        {/* Main Landing / Dashboard Overview Page */}
        <main className="flex-1 overflow-y-auto p-8 relative z-10 bg-[#05070c]">
          {/* Visual background gradient lights */}
          <div className="absolute top-1/4 left-1/3 w-96 h-96 bg-indigo-900/10 rounded-full blur-[100px] pointer-events-none" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-900/5 rounded-full blur-[100px] pointer-events-none" />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 relative z-10">
            {/* Left 2/3 - Research System Overview */}
            <div className="lg:col-span-2 space-y-6">
              <div className="space-y-2">
                <h1 className="text-3xl font-black text-white tracking-tight">
                  MOOSY-32 Clinical Portal
                </h1>
                <p className="text-slate-400 text-sm">
                  Explainable AI Clinical Support Dashboard for Prostate Cancer Screening
                </p>
              </div>

              {/* Research System Overview Panel */}
              <div className="glass-panel p-6 rounded-xl border border-slate-800/80 space-y-6">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Cpu className="w-5 h-5 text-cyan-400" />
                    <span>E-Nose VOC Research Overview</span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                    This clinical decision support system leverages an Electronic Nose (MOOSY-32) containing an array of 32 Metal Oxide Semiconductor (MOS) sensors. By analyzing the headspace of urine samples, the system detects Volatile Organic Compounds (VOCs) that serve as biomarkers for prostate disease.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  <div className="bg-slate-950/40 border border-slate-900 p-4 rounded-lg space-y-2">
                    <span className="font-semibold text-indigo-400 block">Signal Temporal Features</span>
                    <p className="text-slate-400 leading-relaxed text-[11px]">
                      The sensor array captures voltage responses over time. The signal pre-processing pipeline extracts 31 temporal features per sensor channel, describing peak values, response slopes, area under curve, and differential gas concentrations.
                    </p>
                  </div>
                  <div className="bg-slate-950/40 border border-slate-900 p-4 rounded-lg space-y-2">
                    <span className="font-semibold text-cyan-400 block">CNN-GRU-Attention Network</span>
                    <p className="text-slate-400 leading-relaxed text-[11px]">
                      The sequence of 32 sensor responses is processed by a hybrid deep learning model. Convolutional layers capture spatial/channel features, Gated Recurrent Units map sequential signal flow, and a custom Attention mechanism highlights the most diagnostic sensors.
                    </p>
                  </div>
                </div>

                <div className="border-t border-slate-900 pt-4 space-y-3">
                  <h3 className="font-bold text-white text-xs">Clinical Significance</h3>
                  <p className="text-[11px] leading-relaxed text-slate-400">
                    Traditional screening relies heavily on Serum PSA, which has high false-positive rates leading to unnecessary invasive biopsies. MOOSY-32 VOC biomarker profiles act as a non-invasive adjunct test to distinguish Prostate Cancer (CaP) from Benign Prostatic Hyperplasia (HBP) with high sensitivity and specificity.
                  </p>
                </div>
              </div>
            </div>

            {/* Right 1/3 - Authentication Portal */}
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-slate-800/80 shadow-2xl relative z-10 bg-[#090f1a]/85">
                <div className="flex flex-col items-center mb-6">
                  <h2 className="text-lg font-bold text-white text-center">
                    Secure Workspace Sign In
                  </h2>
                  <p className="text-[11px] text-slate-400 mt-1 text-center">
                    Sign in with your clinician or researcher credential
                  </p>
                </div>

                {error && (
                  <div className="bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-lg p-2.5 flex items-center gap-2 mb-4 text-xs">
                    <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
                    <span>{error}</span>
                  </div>
                )}

                {success && (
                  <div className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 rounded-lg p-2.5 flex items-center gap-2 mb-4 text-xs">
                    <CheckCircle className="w-4 h-4 shrink-0 text-emerald-400" />
                    <span>{success}</span>
                  </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label className="block text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                      Username
                    </label>
                    <div className="relative">
                      <span className="absolute inset-y-0 left-0 pl-2.5 flex items-center pointer-events-none">
                        <User className="w-3.5 h-3.5 text-slate-500" />
                      </span>
                      <input
                        type="text"
                        required
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="e.g. clinician_jane"
                        className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2 pl-8 pr-3 text-xs text-slate-100 placeholder-slate-600 focus:outline-none transition-all"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                      Password
                    </label>
                    <div className="relative">
                      <span className="absolute inset-y-0 left-0 pl-2.5 flex items-center pointer-events-none">
                        <Lock className="w-3.5 h-3.5 text-slate-500" />
                      </span>
                      <input
                        type="password"
                        required
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="••••••••"
                        className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2 pl-8 pr-3 text-xs text-slate-100 placeholder-slate-600 focus:outline-none transition-all"
                      />
                    </div>
                  </div>

                  {isRegister && (
                    <div>
                      <label className="block text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                        System Role
                      </label>
                      <select
                        value={role}
                        onChange={(e) => setRole(e.target.value)}
                        className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2 px-2 text-xs text-slate-100 focus:outline-none transition-all"
                      >
                        <option value="clinician" className="bg-slate-950">Clinician (Diagnosis Dashboard)</option>
                        <option value="researcher" className="bg-slate-950">Researcher (Ablation & Benchmarking)</option>
                      </select>
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full py-2 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 focus:ring-2 focus:ring-indigo-500 focus:outline-none rounded-lg text-xs font-semibold text-white transition-all shadow-lg flex items-center justify-center"
                  >
                    {loading ? 'Processing...' : isRegister ? 'Register Account' : 'Sign In'}
                  </button>
                </form>

                <div className="mt-4 pt-4 border-t border-slate-900 text-center space-y-3">
                  <button
                    onClick={() => {
                      setIsRegister(!isRegister);
                      setError('');
                      setSuccess('');
                    }}
                    className="text-[10px] font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
                  >
                    {isRegister
                      ? 'Already registered? Sign in here'
                      : 'Create new local researcher/clinician credential'}
                  </button>

                  <div className="relative flex py-1 items-center">
                    <div className="flex-grow border-t border-slate-900"></div>
                    <span className="flex-shrink mx-2 text-[9px] text-slate-600 uppercase">OR</span>
                    <div className="flex-grow border-t border-slate-900"></div>
                  </div>

                  <button
                    onClick={handleBypass}
                    className="w-full py-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-300 rounded-lg transition-colors flex items-center justify-center gap-1.5"
                  >
                    <span>Bypass Authentication (Local Demo)</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
