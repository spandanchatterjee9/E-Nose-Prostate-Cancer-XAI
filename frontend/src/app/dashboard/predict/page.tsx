'use client';

import React, { useEffect, useState, useRef } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { API_BASE } from '../../../config';
import { 
  Activity, 
  User, 
  Binary, 
  HelpCircle, 
  Upload, 
  Download, 
  CheckCircle2, 
  AlertTriangle,
  Play, 
  TrendingUp 
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell, 
  ReferenceLine 
} from 'recharts';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

interface Patient {
  id: number;
  patient_code: string;
  age: number;
  psa: number;
  volume: number;
}

export default function PredictPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pdfRef = useRef<HTMLDivElement>(null);
  
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [modelName, setModelName] = useState<string>('hybrid_model');
  const [simulatedIndex, setSimulatedIndex] = useState<string>('0');
  
  // Custom File state
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [parsedCsvData, setParsedCsvData] = useState<any[] | null>(null);
  
  // Results State
  const [prediction, setPrediction] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch initial patient list and check for query parameters
  useEffect(() => {
    const fetchInit = async () => {
      const token = localStorage.getItem('access_token');
      if (!token) return;
      
      try {
        const headers = { 'Authorization': `Bearer ${token}` };
        const res = await fetch(`${API_BASE}/api/v1/patients/`, { headers });
        const data = await res.json();
        if (Array.isArray(data)) {
          setPatients(data);
          
          // Check query parameters
          const queryPatId = searchParams.get('patientId');
          const queryPredId = searchParams.get('predictionId');
          
          if (queryPatId) {
            setSelectedPatientId(queryPatId);
          } else if (data.length > 0) {
            setSelectedPatientId(data[0].id.toString());
          }
          
          if (queryPredId) {
            setLoading(true);
            const predRes = await fetch(`${API_BASE}/api/v1/history/${queryPredId}`, { headers });
            const predData = await predRes.json();
            if (predRes.ok) {
              setPrediction(predData);
              setSelectedPatientId(predData.patient_id.toString());
              setModelName(predData.model_name);
            }
            setLoading(false);
          }
        }
      } catch (err) {
        console.error("Error loading predict data:", err);
      }
    };
    fetchInit();
  }, [searchParams]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setCsvFile(file);
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
        const headers = lines[0].split(',').map(h => h.trim());
        
        const records = lines.slice(1).map(line => {
          const values = line.split(',').map(v => v.trim());
          const obj: any = {};
          headers.forEach((h, i) => {
            if (h === 'Sensor') {
              obj[h] = values[i];
            } else {
              obj[h] = parseFloat(values[i]);
            }
          });
          return obj;
        });
        
        if (records.length !== 32) {
          alert("Error: E-Nose VOC session file must contain exactly 32 sensor rows.");
          setCsvFile(null);
          return;
        }
        
        setParsedCsvData(records);
      } catch (err) {
        alert("Failed to parse CSV file. Ensure valid structure with required headers.");
        setCsvFile(null);
      }
    };
    reader.readAsText(file);
  };

  const handlePredict = async () => {
    if (!selectedPatientId) {
      setError("Please select a patient profile.");
      return;
    }

    setError('');
    setPrediction(null);
    setLoading(true);

    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      const body: any = {
        patient_id: parseInt(selectedPatientId),
        model_name: modelName
      };

      if (parsedCsvData) {
        body.sensor_data = parsedCsvData;
      } else {
        body.simulated_run_index = parseInt(simulatedIndex);
      }

      const res = await fetch(`${API_BASE}/api/v1/predict/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(body)
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Prediction failed.');
      }

      setPrediction(data);
    } catch (err: any) {
      setError(err.message || 'Diagnostic failed.');
    } finally {
      setLoading(false);
    }
  };

  const downloadPDFReport = async () => {
    if (!prediction || !pdfRef.current) return;
    
    setLoading(true);
    try {
      const element = pdfRef.current;
      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        backgroundColor: '#090e18'
      });
      
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgWidth = 210; // A4 size width
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      
      pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
      pdf.save(`E-Nose_Diagnosis_Report_${prediction.patient_code}_${prediction.id}.pdf`);
    } catch (err) {
      console.error("PDF export failed:", err);
      alert("Failed to export PDF.");
    } finally {
      setLoading(false);
    }
  };

  // Prepare SHAP graph data
  const getShapData = () => {
    if (!prediction?.shap_values) return [];
    
    // Sort and slice top 12 features
    const entries = Object.entries(prediction.shap_values)
      .map(([name, val]) => ({ name, value: val as number }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 15);
      
    return entries;
  };

  // Prepare Attention graph data
  const getAttentionData = () => {
    if (!prediction?.attention_weights) return [];
    return prediction.attention_weights.map((w: number, i: number) => ({
      name: `S${i+1}`,
      weight: w
    }));
  };

  // Prepare Feature Saliency data
  const getSaliencyData = () => {
    if (!prediction?.feature_importance) return [];
    return Object.entries(prediction.feature_importance)
      .map(([name, val]) => ({ name, importance: val as number }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 15);
  };

  const selectedPatientObj = patients.find(p => p.id.toString() === selectedPatientId);
  const isHybrid = prediction?.model_name === 'hybrid_model';
  const isSequence = ['hybrid_model', 'cnn', 'gru', 'cnn_gru'].includes(prediction?.model_name);

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Predict & Explain (XAI)</h1>
          <p className="text-slate-400 mt-1.5 text-sm">
            Execute model inference and view explainability attributions.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Control Panel */}
        <div className="glass-card p-6 rounded-xl border border-slate-800/80 space-y-6 h-fit">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <User className="w-5 h-5 text-indigo-400" />
            <span>Diagnosis Controller</span>
          </h2>

          <div className="space-y-4">
            {/* Patient selection */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                Select Patient Profile
              </label>
              <select
                value={selectedPatientId}
                onChange={(e) => setSelectedPatientId(e.target.value)}
                className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2.5 px-3 text-sm text-slate-100 focus:outline-none transition-all"
              >
                {patients.map(p => (
                  <option key={p.id} value={p.id} className="bg-slate-950">
                    {p.patient_code} (Age: {p.age} | PSA: {p.psa.toFixed(2)})
                  </option>
                ))}
              </select>
            </div>

            {/* Model selection */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                Classification Model
              </label>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2.5 px-3 text-sm text-slate-100 focus:outline-none transition-all"
              >
                <option value="hybrid_model" className="bg-slate-950">Proposed Hybrid CNN-GRU-Attention</option>
                <option value="cnn" className="bg-slate-950">CNN Sequence Baseline</option>
                <option value="gru" className="bg-slate-950">GRU Sequence Baseline</option>
                <option value="cnn_gru" className="bg-slate-950">CNN-GRU Sequence Baseline</option>
                <option value="baseline_dnn" className="bg-slate-950">Baseline Dense Neural Network</option>
                <option value="random_forest" className="bg-slate-950">Random Forest Baseline</option>
                <option value="xgboost" className="bg-slate-950">XGBoost Baseline</option>
              </select>
            </div>

            {/* CSV File or Simulation Select */}
            <div className="border-t border-slate-900 pt-4 space-y-4">
              <span className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                E-Nose VOC Feature Input
              </span>

              {/* Upload choice toggles */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <button
                  onClick={() => { setCsvFile(null); setParsedCsvData(null); }}
                  className={`py-2 rounded border text-center transition-all ${
                    !csvFile ? 'bg-indigo-600/10 border-indigo-500 text-indigo-300' : 'border-slate-800 text-slate-400'
                  }`}
                >
                  Simulate Cohort Run
                </button>
                <label className={`py-2 rounded border text-center cursor-pointer transition-all ${
                  csvFile ? 'bg-indigo-600/10 border-indigo-500 text-indigo-300' : 'border-slate-800 text-slate-400'
                }`}>
                  Upload sensor CSV
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              </div>

              {!csvFile ? (
                <div>
                  <label className="block text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-1.5">
                    Select Test Run Index (0–399)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="399"
                    value={simulatedIndex}
                    onChange={(e) => setSimulatedIndex(e.target.value)}
                    className="w-full bg-slate-950/60 border border-slate-800 focus:border-indigo-500 rounded-lg py-2 px-3 text-sm text-slate-100 focus:outline-none transition-all"
                  />
                  <span className="text-[10px] text-slate-500 mt-1 block">
                    Reads 32 sensor rows from test dataset partition
                  </span>
                </div>
              ) : (
                <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-3 text-xs flex items-center justify-between text-slate-300">
                  <span className="truncate max-w-[180px]">{csvFile.name}</span>
                  <button 
                    onClick={() => { setCsvFile(null); setParsedCsvData(null); }}
                    className="text-rose-400 hover:text-rose-300"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full py-3 bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white rounded-lg text-sm font-semibold transition-all shadow-lg flex items-center justify-center gap-2 mt-4"
            >
              {loading ? 'Evaluating...' : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Execute Diagnosis</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Output Panel */}
        <div className="lg:col-span-2 space-y-6">
          {error && (
            <div className="bg-rose-500/10 border border-rose-500/20 text-rose-300 p-4 rounded-xl flex items-center gap-2 text-sm">
              <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {!prediction && !loading && (
            <div className="glass-card p-12 text-center text-slate-500 rounded-xl border border-slate-800/80">
              <Activity className="w-12 h-12 text-slate-600 mx-auto mb-4 animate-pulse" />
              <h3 className="font-bold text-slate-300 text-lg mb-1">Awaiting Diagnosis Execution</h3>
              <p className="text-sm max-w-sm mx-auto">
                Set the parameters in the controller on the left and click **Execute Diagnosis** to run prediction and calculate SHAP attributions.
              </p>
            </div>
          )}

          {loading && (
            <div className="glass-card p-12 text-center text-slate-500 rounded-xl border border-slate-800/80 flex flex-col items-center justify-center gap-4">
              <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <div>
                <h3 className="font-bold text-slate-300 text-lg mb-1">Evaluating Neural Network Model</h3>
                <p className="text-sm">Pre-processing VOC features & generating explanations...</p>
              </div>
            </div>
          )}

          {/* DIAGNOSTIC REPORT (EXPLAINABLE PANEL) */}
          {prediction && !loading && (
            <div className="space-y-6">
              {/* PDF Container Wrapper */}
              <div ref={pdfRef} className="p-8 bg-[#090e18] border border-slate-800/80 rounded-xl space-y-6">
                
                {/* PDF Header (Only visible in PDF export layout or styled nicely for screen) */}
                <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                  <div>
                    <h2 className="text-xl font-bold text-white">E-Nose Patient Diagnostic Report</h2>
                    <span className="text-[10px] uppercase font-semibold text-slate-500">
                      ID: {prediction.id} | Session Date: {new Date(prediction.created_at).toLocaleString()}
                    </span>
                  </div>
                  <button 
                    onClick={downloadPDFReport}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download Report</span>
                  </button>
                </div>

                {/* Patient / Model Summary */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 bg-slate-950/40 border border-slate-900 p-4 rounded-lg">
                  <div>
                    <span className="text-[10px] uppercase font-bold text-slate-500 block">Patient Code</span>
                    <span className="text-sm font-semibold text-white mt-1 block">{prediction.patient_code}</span>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase font-bold text-slate-500 block">Age / Serum PSA</span>
                    <span className="text-sm font-semibold text-white mt-1 block">
                      {selectedPatientObj?.age} yrs / {selectedPatientObj?.psa.toFixed(2)} ng/mL
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase font-bold text-slate-500 block">Prostate Volume</span>
                    <span className="text-sm font-semibold text-white mt-1 block">{selectedPatientObj?.volume.toFixed(1)} cc</span>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase font-bold text-slate-500 block">Diagnostic Model</span>
                    <span className="text-sm font-semibold text-white mt-1 block capitalize">
                      {prediction.model_name.replace('_', ' ')}
                    </span>
                  </div>
                </div>

                {/* Prediction Result Gauge */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
                  <div className="md:col-span-2 space-y-2">
                    <span className="text-[10px] uppercase font-bold text-slate-500">Inferred Classification</span>
                    <div className="flex items-center gap-3">
                      <h3 className={`text-3xl font-black ${
                        prediction.prediction_label === 'CaP' ? 'text-rose-400' : 'text-emerald-400'
                      }`}>
                        {prediction.prediction_label === 'CaP' 
                          ? 'Prostate Cancer (CaP)' 
                          : 'Benign Hyperplasia (HBP)'}
                      </h3>
                      {prediction.prediction_label === 'CaP' ? (
                        <AlertTriangle className="w-7 h-7 text-rose-500" />
                      ) : (
                        <CheckCircle2 className="w-7 h-7 text-emerald-500" />
                      )}
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {prediction.prediction_label === 'CaP' 
                        ? 'Alert: VOC biomarker concentration patterns indicate high association with adenocarcinoma. Transrectal ultrasound biopsy recommended.'
                        : 'Biomarker analysis indicates standard VOC concentration profiles typical of benign prostate hypertrophy. Monitor PSA kinetics.'}
                    </p>
                  </div>
                  
                  {/* Gauge */}
                  <div className="flex flex-col items-center justify-center p-4 bg-slate-950/30 border border-slate-900 rounded-lg">
                    <span className="text-[10px] uppercase font-bold text-slate-500 mb-2">Model Confidence</span>
                    <div className="relative w-24 h-24 flex items-center justify-center">
                      {/* Ring */}
                      <svg className="w-full h-full transform -rotate-90">
                        <circle cx="48" cy="48" r="40" stroke="#1e293b" strokeWidth="6" fill="transparent" />
                        <circle 
                          cx="48" cy="48" r="40" 
                          stroke={prediction.prediction_label === 'CaP' ? '#f43f5e' : '#34d399'} 
                          strokeWidth="6" 
                          fill="transparent" 
                          strokeDasharray={2 * Math.PI * 40}
                          strokeDashoffset={2 * Math.PI * 40 * (1 - prediction.confidence)}
                        />
                      </svg>
                      <span className="absolute text-xl font-bold text-white">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* VISUAL XAI ATTRIBUTIONS */}
                <div className="border-t border-slate-900 pt-6 space-y-6">
                  
                  {/* 1. Proposed Hybrid: Sensor Attention weights */}
                  {isHybrid && (
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-bold text-white text-sm flex items-center gap-1.5">
                          <TrendingUp className="w-4 h-4 text-violet-400" />
                          <span>MOOSY-32 Sensor Channel Attention Heatmap</span>
                        </h4>
                        <p className="text-[11px] text-slate-500 mt-0.5">
                          Attention layer weights (relative importance) assigned to each of the 32 MOS sensors.
                        </p>
                      </div>

                      <div className="h-48 w-full bg-slate-950/40 p-2 rounded-lg border border-slate-900">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={getAttentionData()}>
                            <XAxis dataKey="name" stroke="#64748b" fontSize={9} tickLine={false} />
                            <YAxis stroke="#64748b" fontSize={9} tickLine={false} />
                            <Tooltip 
                              contentStyle={{ backgroundColor: '#090e18', borderColor: '#334155' }}
                              labelClassName="text-white text-xs"
                            />
                            <Bar dataKey="weight" fill="#a78bfa">
                              {getAttentionData().map((entry: any, index: number) => {
                                const isHigh = entry.weight > 0.04;
                                return <Cell key={`cell-${index}`} fill={isHigh ? '#a78bfa' : '#475569'} />;
                              })}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}

                  {/* 2. Hybrid Feature importance or SHAP attributions */}
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-bold text-white text-sm">
                        {isSequence ? 'Feature Saliency Attribution' : 'SHAP Waterfall Attributions'}
                      </h4>
                      <p className="text-[11px] text-slate-500 mt-0.5">
                        {isSequence 
                          ? 'Sensitivity gradients showing the impact of VOC features on model output.' 
                          : 'Feature values pushing the model prediction towards CaP (positive, red) or HBP (negative, green).'}
                      </p>
                    </div>

                    <div className="h-64 w-full bg-slate-950/40 p-2 rounded-lg border border-slate-900">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          layout="vertical"
                          data={isSequence ? getSaliencyData() : getShapData()}
                          margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis type="number" stroke="#64748b" fontSize={9} />
                          <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={9} width={80} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#090e18', borderColor: '#334155' }}
                          />
                          {isSequence ? (
                            <Bar dataKey="importance" fill="#22d3ee" barSize={10} />
                          ) : (
                            <Bar dataKey="value" barSize={10}>
                              {getShapData().map((entry: any, index: number) => (
                                <Cell 
                                  key={`cell-${index}`} 
                                  fill={entry.value > 0 ? '#f43f5e' : '#34d399'} 
                                />
                              ))}
                            </Bar>
                          )}
                          {!isSequence && <ReferenceLine x={0} stroke="#475569" />}
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
