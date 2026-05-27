'use client';

import React, { useEffect, useState } from 'react';
import { API_BASE } from '../../../config';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend 
} from 'recharts';
import { 
  LineChart as ChartIcon, 
  Settings2, 
  RefreshCw, 
  Grid, 
  HelpCircle, 
  TrendingUp, 
  Cpu 
} from 'lucide-react';

interface ModelBenchmark {
  id: number;
  model_name: string;
  run_id: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  confusion_matrix: number[][];
  classification_report?: {
    roc_points?: { fpr: number; tpr: number }[];
  };
  created_at: string;
}

export default function ResearchPage() {
  const [benchmarks, setBenchmarks] = useState<ModelBenchmark[]>([]);
  const [ablation, setAblation] = useState<any | null>(null);
  const [logs, setLogs] = useState<any[]>([]);
  
  const [loading, setLoading] = useState(true);
  const [ablationLoading, setAblationLoading] = useState(false);
  const [selectedModelForMatrix, setSelectedModelForMatrix] = useState<string>('hybrid_model');
  const [trainingState, setTrainingState] = useState<Record<string, string>>({}); // model_name -> status

  const fetchResearchData = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;
    
    try {
      const headers = { 'Authorization': `Bearer ${token}` };
      
      // 1. Fetch Benchmarks
      const benchRes = await fetch(`${API_BASE}/api/v1/metrics/benchmarks`, { headers });
      const benchData = await benchRes.json();
      if (Array.isArray(benchData)) {
        setBenchmarks(benchData);
      }
      
      // 2. Fetch Experiment logs for loss curves
      const logsRes = await fetch(`${API_BASE}/api/v1/metrics/experiment-logs?limit=5`, { headers });
      const logsData = await logsRes.json();
      if (Array.isArray(logsData)) {
        setLogs(logsData);
      }
    } catch (err) {
      console.error("Error loading research dashboard:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResearchData();
  }, []);

  const handleRunAblation = async () => {
    setAblationLoading(true);
    const token = localStorage.getItem('access_token');
    if (!token) return;
    
    try {
      const res = await fetch(`${API_BASE}/api/v1/metrics/ablation`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setAblation(data);
    } catch (err) {
      console.error("Error running ablation study:", err);
      alert("Failed to compute ablation study.");
    } finally {
      setAblationLoading(false);
    }
  };

  const handleRetrain = async (modelName: string) => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    setTrainingState(prev => ({ ...prev, [modelName]: 'training' }));
    
    try {
      const res = await fetch(`${API_BASE}/api/v1/metrics/train/${modelName}?weighted=true`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      if (res.ok) {
        setTrainingState(prev => ({ ...prev, [modelName]: 'initiated' }));
        // Poll for database update or prompt user
        alert(`Model training run (${data.run_id}) initiated in background. Refresh benchmarks in a few seconds.`);
        setTimeout(() => {
          fetchResearchData();
          setTrainingState(prev => ({ ...prev, [modelName]: 'done' }));
        }, 8000);
      } else {
        throw new Error(data.detail);
      }
    } catch (err: any) {
      alert("Error training model: " + err.message);
      setTrainingState(prev => ({ ...prev, [modelName]: 'failed' }));
    }
  };

  // Align ROC curve coordinates for Recharts multi-line plotting
  const getRocChartData = () => {
    // Generate 20 points from 0.0 to 1.0 for X axis (FPR)
    const chartPoints = Array.from({ length: 20 }, (_, i) => {
      const fpr = i / 19;
      return { fpr };
    });
    
    // Find closest TPR for each FPR index across models
    const getModelTprAtFpr = (modelName: string, targetFpr: number) => {
      const modelBench = benchmarks.find(b => b.model_name === modelName);
      const points = modelBench?.classification_report?.roc_points;
      if (!points || points.length === 0) {
        // Fallback mathematical curve shape
        const auc = modelBench?.roc_auc || 0.5;
        return Math.min(1.0, Math.pow(targetFpr, (1 - auc) / auc || 1));
      }
      
      // Find closest empirical coordinate
      let closest = points[0];
      let minDiff = Math.abs(points[0].fpr - targetFpr);
      for (const p of points) {
        const diff = Math.abs(p.fpr - targetFpr);
        if (diff < minDiff) {
          minDiff = diff;
          closest = p;
        }
      }
      return closest.tpr;
    };
    
    return chartPoints.map(p => ({
      fpr: parseFloat(p.fpr.toFixed(3)),
      'Hybrid Model': getModelTprAtFpr('hybrid_model', p.fpr),
      'Baseline DNN': getModelTprAtFpr('baseline_dnn', p.fpr),
      'Random Forest': getModelTprAtFpr('random_forest', p.fpr),
      'XGBoost': getModelTprAtFpr('xgboost', p.fpr),
    }));
  };

  // Prepare training history curves from logs
  const getTrainingHistoryData = () => {
    // Look for hybrid or dnn logs that contain epoch history
    const dnnLog = logs.find(l => l.model_name === 'baseline_dnn');
    const hybridLog = logs.find(l => l.model_name === 'hybrid_model');
    
    const activeLog = hybridLog || dnnLog;
    if (!activeLog?.training_metrics?.loss) return [];
    
    const loss = activeLog.training_metrics.loss;
    const valLoss = activeLog.training_metrics.val_loss || [];
    
    return loss.map((l: number, i: number) => ({
      epoch: i + 1,
      'Train Loss': l,
      'Val Loss': valLoss[i] || null
    }));
  };

  // Get active matrix info
  const activeBenchmark = benchmarks.find(b => b.model_name === selectedModelForMatrix);
  const matrix = activeBenchmark?.confusion_matrix || [[0, 0], [0, 0]];
  const tn = matrix[0]?.[0] || 0;
  const fp = matrix[0]?.[1] || 0;
  const fn = matrix[1]?.[0] || 0;
  const tp = matrix[1]?.[1] || 0;
  const totalCm = tn + fp + fn + tp || 1;

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
          <h1 className="text-3xl font-bold text-white tracking-tight">Research & Benchmarks</h1>
          <p className="text-slate-400 mt-1.5 text-sm">
            Evaluate empirical diagnostic performance, trigger training sessions, and audit model architectures.
          </p>
        </div>
        <button
          onClick={() => { setLoading(true); fetchResearchData(); }}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded-lg text-xs font-semibold border border-slate-800 transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {/* Model Benchmarking Table */}
      <div className="glass-card p-6 rounded-xl border border-slate-800/80 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Cpu className="w-5 h-5 text-indigo-400" />
          <span>Diagnostic Model Performance Benchmarks</span>
        </h2>
        
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="bg-slate-900/60 border-b border-slate-800/80 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                <th className="py-3 px-6">Model Name</th>
                <th className="py-3 px-6">Accuracy</th>
                <th className="py-3 px-6">Precision</th>
                <th className="py-3 px-6">Recall (Sensitivity)</th>
                <th className="py-3 px-6">F1-Score</th>
                <th className="py-3 px-6">ROC-AUC</th>
                <th className="py-3 px-6 text-right">Re-Train</th>
              </tr>
            </thead>
            <tbody>
              {['hybrid_model', 'baseline_dnn', 'random_forest', 'xgboost'].map((name) => {
                const b = benchmarks.find(item => item.model_name === name);
                const isTrainPending = trainingState[name] === 'training';
                return (
                  <tr key={name} className="border-b border-slate-900 hover:bg-slate-900/10 transition-colors">
                    <td className="py-3.5 px-6 font-semibold text-white capitalize">
                      {name === 'hybrid_model' ? '★ Proposed Hybrid CNN-GRU-Attn' : name.replace('_', ' ')}
                    </td>
                    <td className="py-3.5 px-6 text-slate-300">
                      {b ? `${(b.accuracy * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-3.5 px-6 text-slate-300">
                      {b ? `${(b.precision * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-3.5 px-6 text-slate-300">
                      {b ? `${(b.recall * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-3.5 px-6 text-slate-300">
                      {b ? `${(b.f1_score * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-3.5 px-6 text-slate-300">
                      {b ? `${(b.roc_auc).toFixed(3)}` : '—'}
                    </td>
                    <td className="py-3.5 px-6 text-right">
                      <button
                        onClick={() => handleRetrain(name)}
                        disabled={isTrainPending}
                        className={`px-3 py-1 rounded text-xs font-semibold border transition-all ${
                          isTrainPending 
                            ? 'bg-amber-500/10 border-amber-500/20 text-amber-400' 
                            : 'bg-indigo-600/10 hover:bg-indigo-600/20 border-indigo-500/20 text-indigo-400'
                        }`}
                      >
                        {isTrainPending ? 'Training...' : 'Retrain'}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Analytics Row: ROC Curves & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* ROC Curves Chart */}
        <div className="glass-card p-6 rounded-xl border border-slate-800/80 space-y-4">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <ChartIcon className="w-5 h-5 text-cyan-400" />
            <span>ROC Curves Comparison (Test Partition)</span>
          </h2>
          
          <div className="h-64 w-full bg-slate-950/40 p-2 rounded-lg border border-slate-900">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={getRocChartData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="fpr" label={{ value: 'False Positive Rate (FPR)', position: 'insideBottom', offset: -5 }} stroke="#64748b" fontSize={9} />
                <YAxis label={{ value: 'True Positive Rate (TPR)', angle: -90, position: 'insideLeft', offset: 5 }} stroke="#64748b" fontSize={9} />
                <Tooltip contentStyle={{ backgroundColor: '#090e18', borderColor: '#334155' }} />
                <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                <Line type="monotone" dataKey="Hybrid Model" stroke="#a78bfa" strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="Baseline DNN" stroke="#f43f5e" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="Random Forest" stroke="#34d399" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="XGBoost" stroke="#22d3ee" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix Heatmap */}
        <div className="glass-card p-6 rounded-xl border border-slate-800/80 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Grid className="w-5 h-5 text-emerald-400" />
              <span>Confusion Matrix Visualization</span>
            </h2>
            <select
              value={selectedModelForMatrix}
              onChange={(e) => setSelectedModelForMatrix(e.target.value)}
              className="bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-md py-1 px-2.5 text-xs text-slate-300 focus:outline-none transition-all"
            >
              <option value="hybrid_model">Hybrid Model</option>
              <option value="baseline_dnn">Baseline DNN</option>
              <option value="random_forest">Random Forest</option>
              <option value="xgboost">XGBoost</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4 h-64 items-center">
            {/* 2x2 Heatmap */}
            <div className="grid grid-cols-2 gap-2 max-w-[200px] mx-auto w-full aspect-square relative z-10">
              {/* TN */}
              <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg flex flex-col items-center justify-center p-3 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase">TN (Benign)</span>
                <span className="text-xl font-bold text-emerald-400 mt-1">{tn}</span>
                <span className="text-[9px] text-slate-400 mt-0.5">{((tn/totalCm)*100).toFixed(0)}%</span>
              </div>
              {/* FP */}
              <div className="bg-rose-500/5 border border-rose-500/10 rounded-lg flex flex-col items-center justify-center p-3 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase">FP</span>
                <span className="text-xl font-bold text-rose-400 mt-1">{fp}</span>
                <span className="text-[9px] text-slate-400 mt-0.5">{((fp/totalCm)*100).toFixed(0)}%</span>
              </div>
              {/* FN */}
              <div className="bg-rose-500/5 border border-rose-500/10 rounded-lg flex flex-col items-center justify-center p-3 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase">FN</span>
                <span className="text-xl font-bold text-rose-400 mt-1">{fn}</span>
                <span className="text-[9px] text-slate-400 mt-0.5">{((fn/totalCm)*100).toFixed(0)}%</span>
              </div>
              {/* TP */}
              <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg flex flex-col items-center justify-center p-3 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase">TP (Cancer)</span>
                <span className="text-xl font-bold text-emerald-400 mt-1">{tp}</span>
                <span className="text-[9px] text-slate-400 mt-0.5">{((tp/totalCm)*100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="space-y-3 text-xs pr-4">
              <p className="text-slate-400 leading-relaxed">
                Selected model: <strong className="text-white capitalize">{selectedModelForMatrix.replace('_', ' ')}</strong>
              </p>
              <div className="space-y-1.5">
                <div className="flex justify-between border-b border-slate-900 pb-1.5">
                  <span className="text-slate-500">True Positives (TP):</span>
                  <span className="font-semibold text-emerald-400">{tp}</span>
                </div>
                <div className="flex justify-between border-b border-slate-900 pb-1.5">
                  <span className="text-slate-500">True Negatives (TN):</span>
                  <span className="font-semibold text-emerald-400">{tn}</span>
                </div>
                <div className="flex justify-between border-b border-slate-900 pb-1.5">
                  <span className="text-slate-500">False Positives (FP):</span>
                  <span className="font-semibold text-rose-400">{fp}</span>
                </div>
                <div className="flex justify-between pb-1.5">
                  <span className="text-slate-500">False Negatives (FN):</span>
                  <span className="font-semibold text-rose-400">{fn}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Ablation Study */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Settings2 className="w-5 h-5 text-indigo-400" />
              <span>Proposed Model Ablation Matrix</span>
            </h2>
            <button
              onClick={handleRunAblation}
              disabled={ablationLoading}
              className="flex items-center gap-1.5 px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-semibold transition-colors"
            >
              {ablationLoading ? 'Evaluating...' : 'Compute Ablation'}
            </button>
          </div>

          <div className="glass-card p-6 rounded-xl border border-slate-800/80">
            {!ablation ? (
              <div className="text-slate-500 text-center py-8 text-sm">
                Ablation matrix not yet computed. Click **Compute Ablation** to fit models.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                      <th className="py-2.5 pb-3">Architecture Variant</th>
                      <th className="py-2.5 pb-3">Accuracy</th>
                      <th className="py-2.5 pb-3">Precision</th>
                      <th className="py-2.5 pb-3">Recall</th>
                      <th className="py-2.5 pb-3">ROC-AUC</th>
                      <th className="py-2.5 pb-3">Acc. Drop</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(ablation).map(([variant, val]: [string, any]) => {
                      const fullAcc = ablation['Full Model']?.accuracy || 1.0;
                      const drop = fullAcc - val.accuracy;
                      return (
                        <tr key={variant} className="border-b border-slate-900 py-2 hover:bg-slate-900/10">
                          <td className="py-2.5 font-semibold text-white">{variant}</td>
                          <td className="py-2.5 text-slate-300">{(val.accuracy * 100).toFixed(1)}%</td>
                          <td className="py-2.5 text-slate-300">{(val.precision * 100).toFixed(1)}%</td>
                          <td className="py-2.5 text-slate-300">{(val.recall * 100).toFixed(1)}%</td>
                          <td className="py-2.5 text-slate-300">{val.roc_auc.toFixed(3)}</td>
                          <td className="py-2.5">
                            {drop > 0 ? (
                              <span className="text-rose-400 font-medium">-{ (drop * 100).toFixed(1) }%</span>
                            ) : (
                              <span className="text-slate-500">Ref</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Training history curves */}
        <div className="space-y-4">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-violet-400" />
            <span>Loss Convergence</span>
          </h2>

          <div className="glass-card p-6 rounded-xl border border-slate-800/80">
            {getTrainingHistoryData().length === 0 ? (
              <div className="text-slate-500 text-center py-10 text-xs leading-relaxed">
                No active training histories logged in the system. Triggers background re-training above to stream losses.
              </div>
            ) : (
              <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={getTrainingHistoryData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="epoch" stroke="#64748b" fontSize={8} />
                    <YAxis stroke="#64748b" fontSize={8} />
                    <Tooltip contentStyle={{ backgroundColor: '#090e18', borderColor: '#334155' }} />
                    <Line type="monotone" dataKey="Train Loss" stroke="#22d3ee" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="Val Loss" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
