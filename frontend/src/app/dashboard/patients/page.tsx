'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { API_BASE } from '../../../config';
import { 
  Users, 
  UserPlus, 
  Trash2, 
  Search, 
  AlertCircle,
  TrendingUp, 
  ShieldAlert 
} from 'lucide-react';

interface Patient {
  id: number;
  patient_code: string;
  age: number;
  psa: number;
  volume: number;
  clinical_notes?: string;
  created_at: string;
}

export default function PatientsPage() {
  const router = useRouter();
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Form State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [patientCode, setPatientCode] = useState('');
  const [age, setAge] = useState('');
  const [psa, setPsa] = useState('');
  const [volume, setVolume] = useState('');
  const [notes, setNotes] = useState('');
  const [formError, setFormError] = useState('');
  const [formLoading, setFormLoading] = useState(false);

  const fetchPatients = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;
    
    try {
      const res = await fetch(`${API_BASE}/api/v1/patients/`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      if (Array.isArray(data)) {
        setPatients(data);
      }
    } catch (err) {
      console.error("Error fetching patients:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const handleRegisterPatient = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError('');
    setFormLoading(true);

    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      const res = await fetch(`${API_BASE}/api/v1/patients/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          patient_code: patientCode,
          age: parseInt(age),
          psa: parseFloat(psa),
          volume: parseFloat(volume),
          clinical_notes: notes
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to register patient profile.');
      }

      setIsModalOpen(false);
      // Reset form fields
      setPatientCode('');
      setAge('');
      setPsa('');
      setVolume('');
      setNotes('');
      
      // Refresh list
      fetchPatients();
    } catch (err: any) {
      setFormError(err.message || 'Error occurred.');
    } finally {
      setFormLoading(false);
    }
  };

  const handleDeletePatient = async (id: number) => {
    if (!confirm("Are you sure you want to delete this patient record? This will delete all prediction histories associated with this patient.")) {
      return;
    }

    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      const res = await fetch(`${API_BASE}/api/v1/patients/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        setPatients(patients.filter(p => p.id !== id));
      } else {
        const data = await res.json();
        alert(data.detail || "Error deleting patient.");
      }
    } catch (err) {
      console.error("Error deleting patient:", err);
    }
  };

  // Filter patients by search term
  const filteredPatients = patients.filter(p => 
    p.patient_code.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (p.clinical_notes && p.clinical_notes.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Patient Directory</h1>
          <p className="text-slate-400 mt-1.5 text-sm">
            Manage cohort demographics, clinical test results, and register new patients.
          </p>
        </div>
        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-semibold text-white transition-all shadow-md shadow-indigo-900/20"
        >
          <UserPlus className="w-4 h-4" />
          <span>Register Patient</span>
        </button>
      </div>

      {/* Directory Search & List */}
      <div className="space-y-4">
        <div className="relative max-w-md">
          <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="w-4 h-4 text-slate-500" />
          </span>
          <input
            type="text"
            placeholder="Search by patient code or notes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-[#090e18]/60 border border-slate-800/80 focus:border-indigo-500 rounded-lg py-2 pl-10 pr-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none transition-all"
          />
        </div>

        <div className="glass-card rounded-xl overflow-hidden border border-slate-800/80">
          {filteredPatients.length === 0 ? (
            <div className="p-8 text-center text-slate-500 text-sm">
              No matching patient records found.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="bg-slate-900/60 border-b border-slate-800/80 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                    <th className="py-3 px-6">Patient Code</th>
                    <th className="py-3 px-6">Age</th>
                    <th className="py-3 px-6">PSA (ng/mL)</th>
                    <th className="py-3 px-6">Prostate Vol (cc)</th>
                    <th className="py-3 px-6">Clinical Status</th>
                    <th className="py-3 px-6">Notes</th>
                    <th className="py-3 px-6 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPatients.map((pat) => (
                    <tr 
                      key={pat.id} 
                      className="border-b border-slate-900 hover:bg-slate-900/10 transition-colors"
                    >
                      <td className="py-3.5 px-6 font-semibold text-white">{pat.patient_code}</td>
                      <td className="py-3.5 px-6 text-slate-300">{pat.age}</td>
                      <td className="py-3.5 px-6 text-slate-300">
                        <span className={`font-semibold ${pat.psa >= 4.0 ? 'text-amber-400' : 'text-slate-300'}`}>
                          {pat.psa.toFixed(2)}
                        </span>
                      </td>
                      <td className="py-3.5 px-6 text-slate-300">{pat.volume.toFixed(1)}</td>
                      <td className="py-3.5 px-6">
                        {pat.psa >= 4.0 ? (
                          <span className="flex items-center gap-1 text-[10px] uppercase font-bold text-amber-400">
                            <ShieldAlert className="w-3.5 h-3.5" />
                            <span>Elevated PSA</span>
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-[10px] uppercase font-bold text-slate-400">
                            <TrendingUp className="w-3.5 h-3.5 text-slate-500" />
                            <span>Standard Reference</span>
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-6 text-slate-400 max-w-xs truncate" title={pat.clinical_notes}>
                        {pat.clinical_notes || '—'}
                      </td>
                      <td className="py-3.5 px-6 text-right space-x-2">
                        <button
                          onClick={() => router.push(`/dashboard/predict?patientId=${pat.id}`)}
                          className="px-2.5 py-1 bg-indigo-600/10 hover:bg-indigo-600/20 text-indigo-400 rounded text-xs font-semibold border border-indigo-500/20 transition-all"
                        >
                          Diagnose
                        </button>
                        <button
                          onClick={() => handleDeletePatient(pat.id)}
                          className="p-1 hover:bg-rose-500/10 text-slate-500 hover:text-rose-400 rounded transition-colors"
                          title="Delete Record"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Registration Modal Dialog */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fadeIn">
          <div className="w-full max-w-md bg-[#090e18] border border-slate-800 rounded-xl p-6 shadow-2xl space-y-6">
            <div className="flex items-center justify-between pb-3 border-b border-slate-900">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Users className="w-5 h-5 text-indigo-400" />
                <span>Register Patient Profile</span>
              </h3>
              <button 
                onClick={() => setIsModalOpen(false)}
                className="text-slate-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            {formError && (
              <div className="bg-rose-500/10 border border-rose-500/20 text-rose-300 p-3 rounded-lg flex items-center gap-2 text-xs">
                <AlertCircle className="w-4 h-4 text-rose-400" />
                <span>{formError}</span>
              </div>
            )}

            <form onSubmit={handleRegisterPatient} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1.5">
                  Patient Code *
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. PAT-9021"
                  value={patientCode}
                  onChange={(e) => setPatientCode(e.target.value)}
                  className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2 px-3 text-sm text-slate-100 focus:outline-none transition-all"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1.5">
                    Age *
                  </label>
                  <input
                    type="number"
                    required
                    min="1"
                    placeholder="e.g. 68"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2 px-3 text-sm text-slate-100 focus:outline-none transition-all"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1.5">
                    Serum PSA (ng/mL) *
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    required
                    min="0"
                    placeholder="e.g. 5.4"
                    value={psa}
                    onChange={(e) => setPsa(e.target.value)}
                    className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2 px-3 text-sm text-slate-100 focus:outline-none transition-all"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1.5">
                  Prostate Volume (cc) *
                </label>
                <input
                  type="number"
                  step="0.1"
                  required
                  min="1"
                  placeholder="e.g. 42.5"
                  value={volume}
                  onChange={(e) => setVolume(e.target.value)}
                  className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2 px-3 text-sm text-slate-100 focus:outline-none transition-all"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1.5">
                  Clinical Notes
                </label>
                <textarea
                  rows={3}
                  placeholder="Clinical history, comorbidities, or symptoms..."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2 px-3 text-sm text-slate-100 focus:outline-none transition-all resize-none"
                />
              </div>

              <button
                type="submit"
                disabled={formLoading}
                className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 focus:outline-none rounded-lg text-sm font-semibold text-white transition-all shadow-lg"
              >
                {formLoading ? 'Creating profile...' : 'Register Profile'}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
