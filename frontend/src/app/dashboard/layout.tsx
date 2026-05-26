'use client';

import React, { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { 
  LayoutDashboard, 
  Users, 
  Activity, 
  LineChart, 
  LogOut, 
  Activity as SensorIcon, 
  ShieldCheck 
} from 'lucide-react';
import Link from 'next/link';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [username, setUsername] = useState('Clinician');
  const [role, setRole] = useState('clinician');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const token = localStorage.getItem('access_token');
    if (!token) {
      router.push('/');
      return;
    }
    setUsername(localStorage.getItem('username') || 'Clinician');
    setRole(localStorage.getItem('user_role') || 'clinician');
  }, [router]);

  const handleSignOut = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_role');
    localStorage.removeItem('username');
    router.push('/');
  };

  const navItems = [
    { name: 'Diagnosis Overview', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Patient Files', path: '/dashboard/patients', icon: Users },
    { name: 'Predict & Explain', path: '/dashboard/predict', icon: Activity },
    { name: 'Research & Benchmarking', path: '/dashboard/research', icon: LineChart },
  ];

  if (!mounted) return null;

  return (
    <div className="min-h-screen flex bg-[#030509]">
      {/* Sidebar */}
      <aside className="w-64 bg-[#090e18] border-r border-slate-900 flex flex-col justify-between shrink-0">
        <div>
          {/* Sidebar Header */}
          <div className="h-16 flex items-center gap-2.5 px-6 border-b border-slate-900">
            <SensorIcon className="w-6 h-6 text-cyan-400" />
            <span className="font-semibold text-lg tracking-wide text-white">
              MOOSY-32 AI
            </span>
          </div>

          {/* Navigation Links */}
          <nav className="mt-6 px-4 space-y-1.5">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.path;
              return (
                <Link
                  key={item.name}
                  href={item.path}
                  className={`flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-indigo-600/10 border-l-2 border-indigo-500 text-indigo-200'
                      : 'text-slate-400 hover:bg-slate-900/60 hover:text-slate-100'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-400' : 'text-slate-400'}`} />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Sidebar Footer / User Profile & Logout */}
        <div className="p-4 border-t border-slate-900 space-y-4">
          <div className="flex items-center gap-3 px-2">
            <div className="w-9 h-9 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-indigo-400 font-bold">
              {username.charAt(0).toUpperCase()}
            </div>
            <div className="min-w-0">
              <p className="text-xs font-semibold text-white truncate">{username}</p>
              <div className="flex items-center gap-1 mt-0.5">
                <ShieldCheck className="w-3 h-3 text-cyan-400" />
                <span className="text-[10px] font-medium text-slate-400 capitalize">{role}</span>
              </div>
            </div>
          </div>

          <button
            onClick={handleSignOut}
            className="w-full flex items-center justify-center gap-2.5 py-2 rounded-lg text-sm font-medium border border-slate-800 hover:border-slate-700 text-slate-400 hover:text-rose-400 transition-colors"
          >
            <LogOut className="w-4 h-4" />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>

      {/* Main Workspace Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header Bar */}
        <header className="h-16 bg-[#090e18]/45 border-b border-slate-900/60 flex items-center justify-between px-8 relative z-20">
          <div className="flex items-center gap-3">
            <div className="h-2 w-2 rounded-full bg-emerald-500 animate-ping" />
            <span className="text-xs font-semibold text-slate-300">
              MOOSY-32 Sensor Array Status:
            </span>
            <span className="text-xs font-bold text-emerald-400">
              ONLINE & DRIFT COMPENSATED
            </span>
          </div>
          
          <div className="text-xs text-slate-500">
            Clinical System Workspace v1.0.0
          </div>
        </header>

        {/* Child Router Content */}
        <main className="flex-1 overflow-y-auto p-8 relative z-10">
          {children}
        </main>
      </div>
    </div>
  );
}
